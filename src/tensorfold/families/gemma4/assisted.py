"""Gemma 4 with an MTP assistant (``gemma4_assistant``): drafts one chain a round from the target's caches.

The assistant (``families.gemma4_unified.assistant``) reads the target's last sliding and full layers' keys and the
last kept row's final hidden state, so the rounds keep the last forward's rows. Drafts only propose: the target
verifies every one, so the replies are the serial ones whatever the assistant drafts.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.families.gemma4.model import Gemma4


class AssistantRounds:
    """The draft-head protocol over an ``Assistant``: one MTP chain a round, read after the verify (late speculation).

    Mixed in before a ``Gemma4`` model class, whose forwards it wraps to keep their final-normed rows.
    """

    # the assistant reads only the kept row's final hidden state: earlier prompt chunks skip MLX's last layer
    draft_reads_hidden = False
    _hidden: mx.array | None = None                  # the last forward's final-normed rows [1, N, D]

    def bind_assistant(self, drafter: Any, drafts: int = 0) -> None:
        drafter.bind(self)
        self.mtp, self.head_drafts = drafter, drafter
        self.drafts = int(drafts or drafter.default_drafts)

    # -- forwards: the model's own, keeping the rows for the assistant ---------------------------------------------
    def prefill(self, inputs: Any, cache: list[Any]) -> mx.array:
        out = super().prefill(inputs, cache)
        self._hidden = out
        return out

    def hidden(self, inputs: Any, cache: list[Any], parents: Any = None) -> mx.array:
        out = super().hidden(inputs, cache, parents)
        self._hidden = out
        return out

    def hidden_rows(self, windows: list[Any], caches_: list[list[Any]], parents: Any = None) -> mx.array:
        out = super().hidden_rows(windows, caches_, parents)
        self._hidden = out
        return out

    # -- draft-head protocol ----------------------------------------------------------------------------------------
    def absorb_draft_context(self, hidden: mx.array, next_tokens: Any, cache: list[Any], start: int = 0) -> None:
        """Nothing to absorb: the assistant reads the target's caches and one row's hidden state when it drafts."""

    def speculate(self, cache: list[Any], tokens: Any, position: int, sampling: Any, start: int = 0,
                  last_only: bool = False, rows: Any = None) -> mx.array:
        """Remember the last kept row of the last forward (its hidden state, position) and the token after it."""

        follow = [int(t) for t in (tokens.reshape(-1).tolist() if isinstance(tokens, mx.array) else tokens)]
        kept = [int(r) for r in rows] if rows is not None else list(range(start, start + len(follow)))
        first_position, first_row = self._last.get(id(cache), (int(position), 0))
        row = kept[-1]
        slot = cache[-1]
        slot.hidden = self._hidden[:, row:row + 1, :]
        slot.position = int(first_position) + row - int(first_row)
        slot.token = follow[-1]
        return mx.array([0], dtype=mx.uint32)

    def settle(self, cache: list[Any], keep: int, first: Any, position: int, sampling: Any, count: int) -> Any:
        if count <= 0:
            return []
        slot = cache[-1]
        return self.head_drafts.draft(self._layers(cache), slot.hidden, slot.token, slot.position, int(position),
                                      sampling, int(count))

    def draft_streams(self, caches_: list[list[Any]], follows: list[list[int]], rows: list[list[int]],
                      positions: list[int], samplings: list[Any], depths: list[int]) -> list[Any]:
        """Every stream of a shared round: its kept row read, then one batched chain for those that draft."""

        for cache, follow, kept, sampling in zip(caches_, follows, rows, samplings):
            self.speculate(cache, follow, 0, sampling, rows=kept)
        live = [i for i, d in enumerate(depths) if int(d) > 0]
        out: list[Any] = [[] for _ in caches_]
        if live:
            drafted = self.head_drafts.draft_many(
                [self._layers(caches_[i]) for i in live], [caches_[i][-1] for i in live],
                [int(positions[i]) for i in live], [samplings[i] for i in live], [int(depths[i]) for i in live])
            for i, d in zip(live, drafted):
                out[i] = d
        return out

    def unspeculate(self, cache: list[Any]) -> None:
        pass


class Gemma4Assisted(AssistantRounds, Gemma4):
    """Gemma 4 (26B-A4B MoE or 31B dense) decoded through ``RowDecode``, drafting from its MTP assistant."""

    def __init__(self, model: Any, *, drafter: Any, drafts: int = 0, **options: Any) -> None:
        super().__init__(model, **options)
        self.bind_assistant(drafter, drafts)


__all__ = ["AssistantRounds", "Gemma4Assisted"]
