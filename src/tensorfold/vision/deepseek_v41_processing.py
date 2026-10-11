"""CPU image preparation for DeepSeek-V4.1-Flash, after DeepSeek's ``inference/image_processor.py`` (MIT)."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorfold.server.errors import CONTEXT_LIMIT

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


@dataclass(frozen=True)
class PreparedDeepSeekVisionPrompt:
    token_ids: tuple[int, ...]
    patches: tuple[np.ndarray, ...]             # each image's ViT patches [n_vit_h * n_vit_w, 3, p, p] in float32
    grids: tuple[tuple[int, int], ...]          # each image's ViT patch grid (n_vit_h, n_vit_w)
    image_spans: tuple[tuple[int, int], ...]    # each image's [begin, end) in token_ids
    types: tuple[tuple[int, ...], ...]          # each image span's token types (start, image, new line, end)
    image_hashes: tuple[str, ...]

    @property
    def visual_tokens(self) -> int:
        return sum(end - begin for begin, end in self.image_spans)


@dataclass(frozen=True)
class VisionGeometry:
    patch_size: int
    downsample_ratio: int
    max_n_token: int
    min_pixels: int
    max_wh_ratio: float | None

    @classmethod
    def from_config(cls, vision: dict) -> "VisionGeometry":
        ratio = vision.get("max_wh_ratio")
        return cls(int(vision["patch_size"]), int(vision["downsample_ratio"]), int(vision["max_image_tokens"]),
                   int(vision["min_pixels"]), None if ratio is None else float(ratio))


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int) -> tuple[int, int]:
    """The aligner's token grid for a patch grid of this pixel size."""

    return (math.ceil((best_height // patch_size) / downsample_ratio),
            math.ceil((best_width // patch_size) / downsample_ratio))


def solve_resize_ratio(height: float, width: float, patch_size: int, downsample_ratio: int,
                       max_n_token: int) -> tuple[int, int]:
    """The largest aspect-preserving pixel size whose token grid still fits ``max_n_token``."""

    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:                               # very tall: one column
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:                               # very wide: one row
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / patch_size) * patch_size, math.floor(width * beta / patch_size) * patch_size


def plan_image_grid(width: int, height: int, g: VisionGeometry, max_n_token: int) -> tuple[int, int, int, int]:
    """(n_llm_h, n_llm_w, best_height, best_width) for an image of this original size."""

    p = g.patch_size
    if g.max_wh_ratio is not None and width > height * g.max_wh_ratio:
        width = height * g.max_wh_ratio
    if 0 < width * height < g.min_pixels:
        ratio = (g.min_pixels / (width * height)) ** 0.5
        width, height = int(width * ratio), int(height * ratio)
    best_width, best_height = math.ceil(width / p) * p, math.ceil(height / p) * p
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, p, g.downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(height, width, p, g.downsample_ratio, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, p, g.downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        raise ValueError("an image does not fit its visual-token budget")
    return n_llm_h, n_llm_w, best_height, best_width


def image_patches(image: Any, g: VisionGeometry, max_n_token: int) -> tuple[np.ndarray, int, int, tuple[int, ...]]:
    """A PIL image's ViT patches [n, 3, p, p] (float32, scaled to [-1, 1]), its patch grid and its token types."""

    from PIL import ImageOps

    image = image.convert("RGB")
    n_llm_h, n_llm_w, height, width = plan_image_grid(image.width, image.height, g, max_n_token)
    if g.max_wh_ratio is not None and image.width >= g.max_wh_ratio * image.height:
        image = image.resize((width, height))
    else:
        image = ImageOps.pad(image, (width, height), color=(127, 127, 127))
    p = g.patch_size
    x = (np.asarray(image, dtype=np.float32) / 255 - 0.5) / 0.5
    patches = x.reshape(height // p, p, width // p, p, 3).transpose(0, 2, 4, 1, 3).reshape(-1, 3, p, p)
    types = (IMAGE_START,) + ((IMAGE,) * n_llm_w + (IMAGE_NEW_LINE,)) * n_llm_h + (IMAGE_END,)
    return np.ascontiguousarray(patches), height // p, width // p, types


class DeepSeekV41ImageProcessor:
    """Expands each image placeholder into its span of image tokens, the geometry DeepSeek's processor uses."""

    image_marker = IMAGE_PLACEHOLDER

    def __init__(self, config: dict, tokenizer: Any) -> None:
        self.config, self.tokenizer = config, tokenizer
        self.image_token_id = int(config.get("image_token_id", 129264))
        self.geometry = VisionGeometry.from_config(config["vision_config"])
        ids = list(tokenizer.encode(IMAGE_PLACEHOLDER, add_special_tokens=False))
        if ids != [self.image_token_id]:
            raise ValueError("The tokenizer's image placeholder does not match the DeepSeek-V4.1 vision configuration")

    @classmethod
    def from_directory(cls, model_dir: str | Path, tokenizer: Any) -> "DeepSeekV41ImageProcessor":
        config = json.loads((Path(model_dir).expanduser() / "config.json").read_text())
        if config.get("model_type") != "deepseek_v41" or not isinstance(config.get("vision_config"), dict):
            raise ValueError("Image preprocessing requires a DeepSeek-V4.1-Flash checkpoint with its vision tower")
        return cls(config, tokenizer)

    def prepare(self, rendered_prompt: str, images: Sequence[Any], *, max_visual_tokens: int = 4096,
                max_prompt_tokens: int | None = None) -> PreparedDeepSeekVisionPrompt:
        if not images or max_visual_tokens < 1:
            raise ValueError("DeepSeek-V4.1 image preprocessing needs images and a positive visual-token budget")
        ids = [int(t) for t in self.tokenizer.encode(rendered_prompt, add_special_tokens=False)]
        if ids.count(self.image_token_id) != len(images):
            raise ValueError(f"The rendered prompt has {ids.count(self.image_token_id)} image placeholders for "
                             f"{len(images)} images")
        cap = min(self.geometry.max_n_token, max_visual_tokens // len(images))
        if cap < 5:
            raise ValueError("The visual-token budget is too small for this many images")
        tokens: list[int] = []
        patches, grids, spans, types = [], [], [], []
        it = iter(images)
        for token in ids:
            if token != self.image_token_id:
                tokens.append(token)
                continue
            image = next(it)
            limit = min(cap, 256) if getattr(image, "detail", "auto") == "low" else cap
            patch, n_h, n_w, kinds = image_patches(image.to_pil(), self.geometry, limit)
            spans.append((len(tokens), len(tokens) + len(kinds)))
            tokens.extend([self.image_token_id] * len(kinds))
            patches.append(patch)
            grids.append((n_h, n_w))
            types.append(kinds)
        if max_prompt_tokens is not None and len(tokens) > max_prompt_tokens:   # OpenAI's context_length_exceeded
            raise ValueError(f"{CONTEXT_LIMIT} {max_prompt_tokens} tokens: the expanded image prompt has "
                             f"{len(tokens)} tokens, which exceeds the context window; reduce image resolution or "
                             "prompt length")
        for p in patches:
            p.setflags(write=False)
        return PreparedDeepSeekVisionPrompt(tuple(tokens), tuple(patches), tuple(grids), tuple(spans), tuple(types),
                                            tuple(image.content_hash for image in images))

    def estimate_workspace_bytes(self, prepared: PreparedDeepSeekVisionPrompt) -> int:
        vision = self.config["vision_config"]
        biggest = max((h * w for h, w in prepared.grids), default=0)
        hidden, inter, heads = int(vision["hidden_size"]), int(vision["intermediate_size"]), int(vision["num_attention_heads"])
        measured = int(getattr(self, "workspace_bytes", 0) or 0)
        activation = biggest * (8 * hidden + 3 * inter) * 4 + heads * biggest * biggest * 4
        embeddings = len(prepared.token_ids) * 5120 * 8
        return int(sum(p.nbytes for p in prepared.patches) * 2 + max(measured, activation) + embeddings)
