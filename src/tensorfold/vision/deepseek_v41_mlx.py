"""DeepSeek-V4.1-Flash's ViT and aligner on MLX, after DeepSeek's ``inference/vision.py`` (MIT) as oMLX ports it."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable

from tensorfold.vision.deepseek_v41_processing import (IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START,
                                                        DeepSeekV41ImageProcessor, PreparedDeepSeekVisionPrompt)

EPS = 1e-6
PREFIXES = ("vision.", "aligner.", "image_start", "image_end", "image_newline")


@dataclass(frozen=True)
class EncodedDeepSeekVisionPrompt:
    token_ids: tuple[int, ...]
    inputs_embeds: Any                  # [1, T, D] in the language model's embedding dtype
    image_spans: tuple[tuple[int, int], ...]
    image_hashes: tuple[str, ...]


def _rope(n_h: int, n_w: int, dim: int, theta: float) -> tuple[Any, Any]:
    """The ViT's 2D RoPE tables: height then width positions, each over ``dim // 2`` frequencies."""

    import mlx.core as mx

    inv = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    h = mx.broadcast_to(mx.arange(n_h)[:, None], (n_h, n_w))
    w = mx.broadcast_to(mx.arange(n_w)[None, :], (n_h, n_w))
    freqs = (mx.stack([h, w], axis=-1).reshape(-1, 2, 1).astype(mx.float32) * inv).reshape(n_h * n_w, -1)
    return mx.cos(freqs)[:, None, :], mx.sin(freqs)[:, None, :]


def _rotate(x: Any, cos: Any, sin: Any) -> Any:
    import mlx.core as mx

    x1, x2 = mx.split(x.astype(mx.float32), 2, axis=-1)
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


def _norm(x: Any, weight: Any) -> Any:
    """RMSNorm with fp32 statistics, rounded once (MLX's fused kernel, as mlx-vlm and oMLX run the tower)."""

    import mlx.core as mx

    return mx.fast.rms_norm(x, weight, EPS)


class DeepSeekV41Tower:
    """The ViT (full attention over one image, 2D RoPE) and the 3 x 3 aligner, from the checkpoint's own weights."""

    def __init__(self, w: dict[str, Any], vision: dict) -> None:
        self.w = w
        self.layers = int(vision["num_hidden_layers"])
        self.dim, self.heads = int(vision["hidden_size"]), int(vision["num_attention_heads"])
        self.head_dim = self.dim // self.heads
        self.rope_dim, self.theta = self.head_dim // 2, float(vision.get("rope_theta", 10000))
        self.ratio = int(vision["downsample_ratio"])

    def _linear(self, name: str, x: Any) -> Any:
        """x W^T + b with the bias added before the one rounding (a fused addmm, as the reference's GEMM does)."""

        import mlx.core as mx

        bias = self.w.get(f"{name}.bias")
        weight = self.w[f"{name}.weight"]
        return mx.addmm(bias, x, weight.T) if bias is not None else x @ weight.T

    def vit(self, patches: Any, n_h: int, n_w: int) -> Any:
        import mlx.core as mx

        n = n_h * n_w
        if int(patches.shape[0]) != n:
            raise ValueError(f"expected {n} ViT patches, got {int(patches.shape[0])}")
        x = self._linear("vision.patch_embed.proj", patches.reshape(n, -1))
        cos, sin = _rope(n_h, n_w, self.rope_dim, self.theta)
        for i in range(self.layers):
            p = f"vision.blocks.{i}"
            h = _norm(x, self.w[f"{p}.norm1.weight"])
            q, k, v = (t.reshape(n, self.heads, self.head_dim) for t in mx.split(self._linear(f"{p}.attn.wqkv", h), 3, -1))
            q, k = _rotate(q, cos, sin).transpose(1, 0, 2), _rotate(k, cos, sin).transpose(1, 0, 2)
            o = mx.fast.scaled_dot_product_attention(q[None], k[None], v.transpose(1, 0, 2)[None],
                                                     scale=self.head_dim ** -0.5)[0]
            x = x + self._linear(f"{p}.attn.wo", o.transpose(1, 0, 2).reshape(n, -1))
            gate, up = mx.split(self._linear(f"{p}.ffn.w1", _norm(x, self.w[f"{p}.norm2.weight"])), 2, -1)
            x = x + self._linear(f"{p}.ffn.w2", (gate * mx.sigmoid(gate)) * up)
        return _norm(x, self.w["vision.norm.weight"])

    def align(self, x: Any, n_h: int, n_w: int) -> Any:
        """3 x 3 pixel-unshuffle (F.unfold's channel, row, column order) then w2(gelu(w1(.)))."""

        import mlx.core as mx
        import mlx.nn as nn

        r, dim = self.ratio, int(x.shape[-1])
        x = x.reshape(n_h, n_w, dim)
        if -n_h % r or -n_w % r:
            x = mx.pad(x, ((0, -n_h % r), (0, -n_w % r), (0, 0)))
        out_h, out_w = math.ceil(n_h / r), math.ceil(n_w / r)
        x = x.reshape(out_h, r, out_w, r, dim).transpose(0, 2, 4, 1, 3).reshape(out_h * out_w, dim * r * r)
        return self._linear("aligner.w2", nn.gelu(self._linear("aligner.w1", x)))

    def features(self, patches: Any, n_h: int, n_w: int) -> Any:
        return self.align(self.vit(patches, n_h, n_w), n_h, n_w)


def _load(model_dir: Path) -> dict[str, Any]:
    import mlx.core as mx

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    names = [k for k in index if k.startswith(PREFIXES)]
    if not any(k.startswith("vision.") for k in names):
        raise ValueError("this DeepSeek-V4.1 checkpoint has no vision tower")
    out: dict[str, Any] = {}
    for shard in sorted({index[k] for k in names}):
        data = mx.load(str(model_dir / shard))
        out.update({k: data[k] for k in names if index[k] == shard})
    if any(k.endswith(".scales") for k in out):
        raise ValueError("a quantized DeepSeek-V4.1 vision tower is not supported: the tower must be bf16")
    return out


class DeepSeekV41VisionFrontend(DeepSeekV41ImageProcessor):
    """The checkpoint's ViT, aligner and span delimiters beside TensorFold's loaded language model."""

    def __init__(self, config: dict, tokenizer: Any, embed_tokens: Callable, weights: dict[str, Any],
                 allow_urls: bool = False) -> None:
        super().__init__(config, tokenizer)
        self.embed_tokens, self.allow_urls = embed_tokens, allow_urls
        self.tower = DeepSeekV41Tower(weights, config["vision_config"])
        self.delimiters = {IMAGE_START: weights["image_start"], IMAGE_NEW_LINE: weights["image_newline"],
                           IMAGE_END: weights["image_end"]}

    @classmethod
    def load(cls, model_dir: str | Path, tokenizer: Any, embed_tokens: Callable,
             allow_urls: bool = False) -> "DeepSeekV41VisionFrontend":
        import mlx.core as mx

        path = Path(model_dir).expanduser()
        config = json.loads((path / "config.json").read_text())
        if config.get("model_type") != "deepseek_v41" or not isinstance(config.get("vision_config"), dict):
            raise ValueError("DeepSeek-V4.1 image input needs a checkpoint with its vision_config and tower")
        weights = _load(path)
        mx.eval(list(weights.values()))
        front = cls(config, tokenizer, embed_tokens, weights, allow_urls)
        front.workspace_bytes = front.measure_workspace()
        return front

    def weight_bytes(self) -> int:
        return sum(int(a.nbytes) for a in self.tower.w.values())

    def measure_workspace(self) -> int:
        """Peak extra memory of the largest image (1,024 tokens) through the tower."""

        import mlx.core as mx

        g = self.geometry
        side = math.isqrt(g.max_n_token - 2)
        n = side * g.downsample_ratio
        patches = mx.zeros((n * n, 3, g.patch_size, g.patch_size), dtype=mx.bfloat16)
        mx.eval(patches)
        mx.synchronize()
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        mx.eval(self.tower.features(patches, n, n))
        peak = int(mx.get_peak_memory()) - int(base)
        mx.clear_cache()
        return max(0, peak)

    def encode(self, prepared: PreparedDeepSeekVisionPrompt) -> EncodedDeepSeekVisionPrompt:
        import mlx.core as mx

        tokens = mx.array([list(prepared.token_ids)], dtype=mx.uint32)
        embeddings = self.embed_tokens(tokens.reshape(-1)).reshape(1, len(prepared.token_ids), -1)
        for patches, (n_h, n_w), (begin, end), kinds in zip(prepared.patches, prepared.grids, prepared.image_spans,
                                                            prepared.types):
            features = self.tower.features(mx.array(patches).astype(mx.bfloat16), n_h, n_w)
            if kinds.count(IMAGE) != int(features.shape[0]) or end - begin != len(kinds):
                raise ValueError("DeepSeek-V4.1 image layout does not match the aligned features")
            rows, at = [], 0
            for kind in kinds:
                if kind == IMAGE:
                    rows.append(features[at])
                    at += 1
                else:
                    rows.append(self.delimiters[kind])
            embeddings[0, begin:end] = mx.stack(rows).astype(embeddings.dtype)
        mx.eval(embeddings)
        return EncodedDeepSeekVisionPrompt(prepared.token_ids, embeddings, prepared.image_spans, prepared.image_hashes)
