"""Gemma 4 unified (12B) on the lane engine: mlx_lm's forward for prompts, row-exact dense kernels for every decode row.

The unified checkpoint's text model is Gemma 4's dense decoder (no MoE block, no per-layer inputs, no shared-KV layers
in the 12B): Gemma 4's caches, attention and RoPE kernels with a dense MLP, at whatever affine width oQ wrote.
Its ``gemma4_unified_assistant`` drafter is an MTP head that reads the target's last sliding and full layers' keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tensorfold.families.gemma4.assisted import AssistantRounds
from tensorfold.families.gemma4.model import Gemma4, realize
from tensorfold.kernels.gemma.dense.v1.decode import DenseDecode


class Gemma4Unified(AssistantRounds, Gemma4):
    """mlx_lm's Gemma 4 dense text model decoded through ``DenseDecode``; drafts from its MTP assistant, if given."""

    def __init__(self, model: Any, *, backend: str | None = None, head_backend: str | None = None, check: bool = True,
                 tokenizer: Any = None, drafter: Any = None, drafts: int = 0) -> None:
        realize(model)
        self.model = model
        self.text = getattr(model, "language_model", model)
        self.backbone = self.text.model
        self.args = self.text.args
        self.decode = DenseDecode(self.text, backend or "rows", head_backend)
        self.head_drafts = None
        self.mtp = None
        self.drafts = 0
        self._last: dict[int, tuple[int, int]] = {}
        if drafter is not None:
            self.bind_assistant(drafter, drafts)
        self.exact_width, self.window_costs = (1, {})
        self.shared_costs: dict[int, float] = {}
        if check:
            self.exact_width, self.window_costs = self.check_windows(tokenizer)
        self.multi_row_exact = self.exact_width >= 2
        if check and self.multi_row_exact and not self.check_streams(tokenizer):
            self.max_streams = 1
            print("[gemma4_unified] a forward over several streams' rows does not reproduce each stream's own call "
                  "here: one stream a forward", flush=True)
        if check and self.multi_row_exact and self.max_streams > 1:
            self.shared_costs = self.time_shared_rows(tokenizer)
        if check:
            timing = ", ".join(f"{w}: {ms:.1f}" for w, ms in sorted(self.window_costs.items()))
            shared = ", ".join(f"{w}: {ms:.1f}" for w, ms in sorted(self.shared_costs.items()))
            print(f"[gemma4_unified] {self.decode.backend} matmul; windows of up to {self.exact_width} rows reproduce "
                  f"one-row steps here (ms by rows {timing}; shared forwards {shared or 'none'})", flush=True)


def load_target(model_dir: Path) -> tuple[Any, Any]:
    """mlx_lm's Gemma 4 (``gemma4``: the language model, vision/audio inputs dropped) on the unified checkpoint."""

    from mlx_lm.models import gemma4
    from mlx_lm.utils import load_model, load_tokenizer

    class Unified(gemma4.Model):
        """gemma4's sanitize, which also drops the unified checkpoint's patch embedder (``vision_embedder``)."""

        def sanitize(self, weights: dict) -> dict:
            return super().sanitize({k: v for k, v in weights.items()
                                     if not k.removeprefix("model.").startswith("vision_embedder")})

    # the unified text model is gemma4_text's dense decoder: the oQ and mlx-vlm layouts both sanitize to it
    model, config = load_model(Path(model_dir), get_model_classes=lambda config: (Unified, gemma4.ModelArgs))
    tokenizer = load_tokenizer(Path(model_dir), eos_token_ids=config.get("eos_token_id", None))
    return model, tokenizer


def load(model_dir: Path, *, backend: str | None = None, check: bool = True, drafter: str = "",
         drafts: int = 0) -> tuple[Gemma4Unified, Any]:
    model, tokenizer = load_target(Path(model_dir))
    realize(model)
    draft = None
    if drafter:
        from tensorfold.families.gemma4_unified.assistant import Assistant

        draft = Assistant.load(Path(drafter))
        print(f"[tensorfold] drafter {drafter} (Gemma 4 MTP assistant, {draft.describe()})", flush=True)
    return Gemma4Unified(model, backend=backend, check=check, tokenizer=tokenizer, drafter=draft,
                         drafts=drafts), tokenizer


__all__ = ["Gemma4Unified", "load", "load_target"]
