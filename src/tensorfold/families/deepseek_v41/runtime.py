"""DeepSeek-V4.1-Flash on V4's lane runtime, with the checkpoint's own DSpark drafter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tensorfold.families.deepseek_v41.config import DECODE_ROWS, INDEX_QUERIES
from tensorfold.families.deepseek_v4.runtime import DeepSeekFlash, DSparkFlash, materialize


def _workspace_per_token(args: Any) -> int:
    from tensorfold.kernels.deepseek.v41 import rows as KV

    if KV.index_fits(int(args.index_head_dim)):
        return INDEX_QUERIES * 16
    return INDEX_QUERIES * args.index_n_heads * 8


class DeepSeekV41Flash(DeepSeekFlash):
    """The backbone behind the lane protocol, without a draft head: one row a step."""

    tag = "deepseek_v41"
    fused_rows = DECODE_ROWS            # V4.1's decode window: checked exact at load like every width below it
    max_streams = DECODE_ROWS

    @property
    def prefill_workspace_per_token(self) -> int:
        """Prefill bytes a position of context: an index query block's scores over every key (Metal's fused kernels
        keep [block, keys] fp32 and the sort's copies, measured ~0.9 KB; the unfused path a [block, heads, keys] pair)."""

        return _workspace_per_token(self.args)


def _prefetch_prompt(self: Any, tokens: Any, begin: int, end: int) -> None:
    """The next prompt rows' Engram pages, read while this chunk computes."""

    self.model.prefetch_prompt(tokens, begin, end)


DeepSeekV41Flash.prefetch_prompt = _prefetch_prompt


class DSparkV41Flash(DSparkFlash):
    """DSpark drafts after a round is read: the kept rows' taps go into its rings, one pass drafts a block."""

    tag = "deepseek_v41"
    fused_rows = DECODE_ROWS
    max_streams = DECODE_ROWS

    @property
    def prefill_workspace_per_token(self) -> int:
        return _workspace_per_token(self.args)

    def draft_streams(self, caches: list[list[Any]], follows: list[list[int]], rows: list[list[int]],
                      positions: list[int], samplings: list[Any], depths: list[int]) -> list[Any]:
        """Each stream's rings take its kept rows, then one pass of the drafter drafts for every stream that drafts."""

        import mlx.core as mx

        from tensorfold.engine.gpu_sampling import sample as gpu_sample

        ringsets = [cache[self.layer_count:] for cache in caches]
        for rings, kept in zip(ringsets, rows):
            self.dspark.absorb(mx.take(self._rows, mx.array([int(r) for r in kept], dtype=mx.int32), axis=0), rings)
        live = [i for i, d in enumerate(depths) if d > 0]
        out: list[Any] = [[] for _ in caches]
        if live:
            tokens = [mx.array([int(follows[i][-1])], dtype=mx.uint32) for i in live]
            logits = self.dspark.logits_streams(self.model, tokens, [ringsets[i] for i in live])
            draws = [(lambda row, j, s=samplings[i], k=positions[i]: gpu_sample(row, s, [k + j])) for i in live]
            drafted = self.dspark.draw_streams(logits, tokens, [min(int(depths[i]), self.dspark.size) for i in live],
                                               draws)
            for i, d in zip(live, drafted):
                out[i] = d
        mx.async_eval(*[d for d in out if isinstance(d, mx.array)])
        return out


DSparkV41Flash.prefetch_prompt = _prefetch_prompt


def load(model_dir: Path, *, drafter: str = "", mtp_drafts: int | None = None, check: bool = True,
         **_: Any) -> tuple[DeepSeekFlash, Any]:
    """The runtime and tokenizer; ``mtp_drafts`` 0 serves without the checkpoint's DSpark drafter."""

    from tensorfold.families.deepseek_v41 import dspark
    from tensorfold.families.deepseek_v41.engram import build_compressed_token_map
    from tensorfold.families.deepseek_v41.prompts import DeepSeekV41Tokenizer
    from tensorfold.families.deepseek_v41.weights import load_backbone
    from tensorfold.families.tokenizer import load_tokenizer

    model = load_backbone(Path(model_dir))
    inner = load_tokenizer(Path(model_dir), eos_token_ids=model.args.eos_token_id or None)
    if model.args.engram_layer_ids:
        hf = getattr(inner, "_tokenizer", inner)
        model.set_token_map(build_compressed_token_map(hf)[0])
    drafts = 5 if mtp_drafts is None else int(mtp_drafts)
    w = model.weights
    runtime: DeepSeekFlash
    if drafts > 0 and dspark.has_drafter(w.raw, w.where):
        runtime, kind = DSparkV41Flash(model, dspark.load(model, w), check=check), "DSpark"
    else:
        runtime, kind = DeepSeekV41Flash(model, None, drafts=0, check=check), "none"
    w.release()
    model.weights = None
    materialize(runtime)
    print(f"[deepseek_v41] exact window {runtime.exact_width} rows, forward ms by width {runtime.window_costs}, "
          f"drafter {kind}, step {runtime.mtp_step_ms} ms", flush=True)
    return runtime, DeepSeekV41Tokenizer(inner)
