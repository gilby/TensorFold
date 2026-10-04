"""Qwen3.6 MoE EXL3 packs on a Mac: the trellis tiles as stored, read by tensorfold.kernels.exl3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from tensorfold.cuda.exl3 import format as fmt
from tensorfold.kernels.exl3 import experts as exl3_experts
from tensorfold.kernels.exl3 import rows, tiles
from tensorfold.kernels.exl3.linear import Exl3Linear

PREFIX = "model.language_model."


class Exl3MoeBlock(nn.Module):
    """The sparse MoE block: router, routed experts and the shared expert (the experts' last), all EXL3."""

    def __init__(self, ex: exl3_experts.Experts) -> None:
        super().__init__()
        self._ex = ex

    @property
    def experts(self) -> exl3_experts.Experts:
        return self._ex

    def __call__(self, x: mx.array) -> mx.array:
        from tensorfold.kernels.exl3 import prompt

        lead = x.shape[:-1]
        y = prompt.moe(self._ex, x.reshape(-1, int(x.shape[-1])))
        return y.reshape(*lead, int(x.shape[-1])).astype(x.dtype)


class _Reader:
    """Lazy MLX arrays of a checkpoint's shards, read by name."""

    def __init__(self, model_dir: Path, checkpoint: fmt.Checkpoint) -> None:
        self.dir, self.ck = model_dir, checkpoint
        self.where: dict[str, str] = {}
        for path in sorted(model_dir.glob("*.safetensors")):
            if path.name == "mtp.safetensors":
                continue
            for name in fmt.read_header(path):
                self.where[name] = path.name
        self.files: dict[str, dict[str, mx.array]] = {}

    def get(self, name: str, keep: bool = True) -> mx.array:
        """The tensor ``name`` (lazy); ``keep`` False forgets it here, so its buffer goes with its one copy."""

        file = self.where[name]
        if file not in self.files:
            self.files[file] = mx.load(str(self.dir / file))
        return self.files[file][name] if keep else self.files[file].pop(name)

    def words(self, prefix: str) -> mx.array:
        t = self.get(prefix + ".trellis", keep=False)
        return t.view(mx.uint32).reshape(int(t.shape[0]), int(t.shape[1]), -1)


def _group(ck: fmt.Checkpoint, prefix: str) -> fmt.Exl3Tensor:
    g = ck.groups.get(prefix)
    if g is None:
        raise ValueError(f"{prefix}: not an EXL3 group in this checkpoint ({ck.bad.get(prefix, 'missing')})")
    if g.in_scales != "suh" or g.out_scales != "svh":
        raise ValueError(f"{prefix}: packed sign scales (su/sv) are not read on a Mac yet")
    return g


def _stack(rd: _Reader, prefixes: list[str]) -> rows.Stack:
    """Projections reading the same rows, one stack (one width and codebook)."""

    groups = [_group(rd.ck, p) for p in prefixes]
    kinds = {(g.bits, g.codebook, g.k) for g in groups}
    if len(kinds) != 1:
        raise ValueError(f"{', '.join(prefixes)}: one launch reads one width and codebook, found {sorted(kinds)}")
    g0 = groups[0]
    words = mx.concatenate([rd.words(p) for p in prefixes], axis=1) if len(prefixes) > 1 else rd.words(prefixes[0])
    suh = mx.stack([rd.get(p + ".suh", keep=False) for p in prefixes])
    svh = mx.concatenate([rd.get(p + ".svh", keep=False) for p in prefixes])
    bias = None
    if any(g.bias for g in groups):
        bias = mx.concatenate([rd.get(p + ".bias") if g.bias else mx.zeros((g.n,), dtype=mx.float16)
                               for p, g in zip(prefixes, groups)]).astype(mx.float16)
    st = rows.Stack(mx.contiguous(words), suh, svh, g0.k, sum(g.n for g in groups), tiles.k2_of(g0.bits),
                    tiles.CODEBOOK_IDS[g0.codebook], tuple(g.n for g in groups), bias=bias, layout="stored")
    mx.eval(st.words, st.suh, st.svh, *([bias] if bias is not None else []))
    return st


def _experts(rd: _Reader, layer: int, args: Any) -> exl3_experts.Experts:
    """The routed experts and the shared expert of ``layer`` as one stack of E + 1."""

    base = f"{PREFIX}layers.{layer}.mlp."
    names = [f"{base}experts.{e}." for e in range(int(args.num_experts))] + [f"{base}shared_expert."]
    gu, dn = [], []
    for n in names:
        for kind in ("gate_proj", "up_proj", "down_proj"):
            g = _group(rd.ck, n + kind)
            (dn if kind == "down_proj" else gu).append((g.bits, g.codebook))
    if len(set(gu)) != 1 or len(set(dn)) != 1 or {c for _, c in gu} != {c for _, c in dn}:
        raise ValueError(f"layer {layer}: the Mac expert kernels read one width for gate/up and one for down, and one "
                         f"codebook; this checkpoint mixes them ({sorted(set(gu))}, {sorted(set(dn))})")
    gate_up = mx.stack([mx.stack([rd.words(n + "gate_proj"), rd.words(n + "up_proj")]) for n in names])
    down = mx.stack([rd.words(n + "down_proj") for n in names])
    get = lambda name: rd.get(name, keep=False)        # noqa: E731
    suh_gu = mx.stack([mx.stack([get(n + "gate_proj.suh"), get(n + "up_proj.suh")]) for n in names])
    svh_gu = mx.stack([mx.stack([get(n + "gate_proj.svh"), get(n + "up_proj.svh")]) for n in names])
    suh_d = mx.stack([get(n + "down_proj.suh") for n in names])
    svh_d = mx.stack([get(n + "down_proj.svh") for n in names])
    router = mx.concatenate([rd.get(base + "gate.weight"), rd.get(base + "shared_expert_gate.weight")]).astype(mx.float16)
    ex = exl3_experts.Experts(gate_up, suh_gu, svh_gu, down, suh_d, svh_d, router, int(args.hidden_size),
                              int(args.moe_intermediate_size), int(args.num_experts), int(args.num_experts_per_tok),
                              tiles.k2_of(gu[0][0]), tiles.k2_of(dn[0][0]), tiles.CODEBOOK_IDS[gu[0][1]],
                              bool(getattr(args, "norm_topk_prob", True)))
    if int(args.shared_expert_intermediate_size) != int(args.moe_intermediate_size):
        raise ValueError("the shared expert rides with the routed ones: their widths must match")
    mx.eval(ex.gate_up, ex.down, ex.suh_gu, ex.svh_gu, ex.suh_d, ex.svh_d, ex.router)
    return ex


def is_exl3(model_dir: str | Path) -> bool:
    return fmt.is_exl3(model_dir)


def load_model(model_dir: str | Path) -> tuple[Any, Any]:
    """The mlx_lm module tree with EXL3 layers in place of its linears (no float copy of a weight) and the tokenizer."""

    from mlx_lm.models import qwen3_5_moe
    from mlx_lm.utils import load_config, load_tokenizer

    root = Path(model_dir)
    config = load_config(root)
    model = qwen3_5_moe.Model(qwen3_5_moe.ModelArgs.from_dict(config))
    model.eval()                              # inference: the recurrence's Metal kernel, not its training loop
    lm = model.language_model
    args = lm.args
    ck = fmt.scan(root, read_markers=False)
    rd = _Reader(root, ck)
    for i, layer in enumerate(lm.model.layers):
        p = f"{PREFIX}layers.{i}."
        if layer.is_linear:
            gdn = layer.linear_attn
            stack = _stack(rd, [p + "linear_attn.in_proj_qkv", p + "linear_attn.in_proj_z"])
            gdn.in_proj_qkv, gdn.in_proj_z = Exl3Linear(stack, 0), Exl3Linear(stack, 1)
            gdn.out_proj = Exl3Linear(_stack(rd, [p + "linear_attn.out_proj"]))
            gdn._exl3_ba = mx.concatenate([rd.get(p + "linear_attn.in_proj_b.weight"),
                                           rd.get(p + "linear_attn.in_proj_a.weight")]).astype(mx.float16)
            mx.eval(gdn._exl3_ba)
        else:
            attn = layer.self_attn
            stack = _stack(rd, [p + "self_attn.q_proj", p + "self_attn.k_proj", p + "self_attn.v_proj"])
            attn.q_proj, attn.k_proj, attn.v_proj = (Exl3Linear(stack, j) for j in range(3))
            attn.o_proj = Exl3Linear(_stack(rd, [p + "self_attn.o_proj"]))
        layer.mlp = Exl3MoeBlock(_experts(rd, i, args))
    lm.lm_head = Exl3Linear(_stack(rd, ["lm_head"]))
    taken = {f"{g}.{part}" for g in ck.groups for part in fmt.PARTS}
    plain = {name: rd.get(name) for name in rd.where if name not in taken and "visual" not in name
             and ".mlp.gate.weight" not in name and "shared_expert_gate" not in name}
    plain = model.sanitize(plain)
    model.load_weights(list(plain.items()), strict=False)
    _check_loaded(model, set(plain))
    mx.eval(model.parameters())
    rd.files.clear()
    model._tensorfold_exl3 = True
    lm.model._tensorfold_exl3 = True
    tokenizer = load_tokenizer(root, eos_token_ids=config.get("eos_token_id"))
    return model, tokenizer


def _check_loaded(model: Any, loaded: set[str]) -> None:
    from mlx.utils import tree_flatten

    missing = [k for k, _ in tree_flatten(model.parameters()) if k not in loaded]
    if missing:
        raise ValueError(f"the EXL3 pack leaves {len(missing)} weights unset, e.g. {missing[:4]}")


__all__ = ["Exl3MoeBlock", "is_exl3", "load_model"]
