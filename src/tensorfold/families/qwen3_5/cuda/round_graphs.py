"""The 27B's concurrent tree-verify rounds on several ranks as CUDA graphs: one per (rows, GDN slot class).

``multi_graphs.MultiGraphs`` replays the chain rounds of Qwen3.6; DFlash2 drafts trees, whose GDN kernel is chosen by
the round's live-state count (``tree_graphs.SLOTS``' classes, the same instantiation the eager forward picks). Here a
graph covers ``width`` rows in ``lanes`` + 1 stream slots (``forward.MultiStaged``: the round's streams first, laid out
as the eager forward lays them, a padding chain to the width, empty slots after) for one slot class, and captures
the drafter taps and the row-parallel sums with the forward. Every kernel computes a row from its own stream's
window and state, so a replay gives each real row the eager forward's bits; a round that fits no graph runs eagerly.

Every rank captures and replays the same graphs in step (the sums meet across ranks); which widths replay is timed
at startup on rank 0 and shared, so every rank takes the same path each round.
"""

from __future__ import annotations

import dataclasses
import gc
import time
from typing import Callable, Sequence

import torch

from tensorfold.cuda.kernels import gdn as deltanet

from .forward import MultiStaged, State, multi_tree_forward, staged_fits
from .multi_graphs import WIDTHS, best_ms, spread
from .tree_graphs import SLOTS, slot_class


class RoundGraphs:
    """Verify graphs for 2 to ``lanes`` streams of tree windows over at most ``context`` keys each, on every rank."""

    def __init__(self, w, lanes: int, context: int, *, tp: bool, taps: bool, full_logits: bool,
                 widths: Sequence[int] = WIDTHS, classes: Sequence[int] = SLOTS[:-1],
                 log: Callable[[str], None] | None = None) -> None:
        c = w.config
        self.w, self.lanes, self.context = w, lanes, context
        self.tp, self.taps, self.full_logits = tp, taps, full_logits
        self.widths = tuple(sorted(x for x in widths if x <= 16 * lanes))   # a window holds 16 rows at most
        self.classes = tuple(classes)                # a stream's window of 16 rows needs 16 live states at most
        self.pool = torch.cuda.graph_pool_handle()
        self.pad_rec = torch.zeros((c.v_heads, c.dv, c.dk), dtype=torch.float32, device=w.norm.device)
        self.entries: dict[tuple[int, int], tuple] = {}
        self.outputs: list[torch.Tensor] | None = None    # every graph's outputs, the widest's rows (one copy)
        self.slower: set[int] = set()                 # widths whose replay lost to the eager forward at startup
        self.timings: dict[int, tuple[float, float]] = {}
        self.replays = 0
        self.log = log

    def key(self, streams) -> tuple[int, int] | None:
        rows = sum(len(t) for t, _, _ in streams)
        width = next((x for x in self.widths if x >= rows), None)
        if width is None or width in self.slower or len(streams) < 2:
            return None
        cls = slot_class(max(deltanet.schedule(p)[1] for _, p, _ in streams))
        if cls not in self.classes or not staged_fits(streams, self.lanes, width, self.context, trees=True):
            return None
        return width, cls

    def _flat(self, out) -> list[torch.Tensor]:
        logits, record, taps = out[0], out[1], out[2] if self.taps else None
        flat = [logits] + [getattr(r, f.name) for r in record for f in dataclasses.fields(r)]
        return flat + ([taps] if taps is not None else [])

    def _unflat(self, views: list[torch.Tensor], record: list):
        out, k = [], 1
        for r in record:
            n = len(dataclasses.fields(r))
            out.append(type(r)(*views[k:k + n]))
            k += n
        return views[0], out, (views[k] if self.taps else None)

    def _run(self, staged: MultiStaged):
        return staged.forward(full_logits=self.full_logits, tp=self.tp, capture_taps=self.taps)

    def capture(self, key: tuple[int, int], streams) -> None:
        """``key``'s graph, staged with these streams and captured once (every rank at once)."""

        if key in self.entries:
            return
        width, cls = key
        staged = MultiStaged(self.w, width, self.lanes, self.context, self.pad_rec, slots=cls)
        staged.refresh(streams)
        out = self._run(staged)                       # compiles each kernel, sizes the scratch, meets the ranks' sums
        if self.outputs is None:
            most = self.widths[-1]
            self.outputs = [t.new_empty((most, *t.shape[1:])) for t in self._flat(out)]
        record = out[1]
        del out
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        enabled_gc = gc.isenabled()
        gc.disable()                                  # collecting an old graph mid-capture invalidates it
        try:
            with torch.cuda.graph(g, pool=self.pool):
                views = [t[:width] for t in self.outputs]
                torch._foreach_copy_(views, self._flat(self._run(staged)))
        finally:
            if enabled_gc:
                gc.enable()
        self.entries[key] = (g, staged, self._unflat(views, record))

    @torch.no_grad()
    def capture_all(self, st: State) -> None:
        """Every (width, class) graph now, widest first (the scratch sizes once): no request waits for a capture."""

        t0 = time.perf_counter()
        for width in reversed(self.widths):
            for cls in self.classes:
                self.capture((width, cls), self._windows(width, st))
        if self.log is not None:
            self.log(f"[tensorfold] {len(self.entries)} concurrent-round verify graphs captured "
                     f"({len(self.widths)} widths x {len(self.classes)} GDN slot classes) in "
                     f"{time.perf_counter() - t0:.1f} s")

    def _windows(self, rows: int, st: State):
        """Chains of ``rows`` rows in all over two streams at least (a graph serves two or more), 16 rows at most each."""

        rows = max(2, rows)
        n = min(self.lanes, max(2, -(-rows // 16)))
        sizes = [rows // n + (i < rows % n) for i in range(n)]
        return [([0] * k, list(range(-1, k - 1)), st) for k in sizes]

    @torch.no_grad()
    def forward(self, streams) -> tuple | None:
        """(logits, record, taps, starts) of a round by a replay, logits and record padded to the graph's width (the
        streams' rows first, at ``starts``); None when the round fits no graph (every rank decides alike)."""

        key = self.key(streams)
        if key is None:
            return None
        return self._replay(key, streams)

    def _replay(self, key, streams):
        if key not in self.entries:
            self.capture(key, streams)
        g, staged, (logits, record, taps) = self.entries[key]
        starts = staged.refresh(streams)
        g.replay()
        self.replays += 1
        return logits, record, taps, starts

    @torch.no_grad()
    def calibrate(self, st: State, rank: int, world: int) -> None:
        """Each width's replay timed against the eager forward of the fewest rows it serves, on rank 0; a width that
        is not faster stays eager on every rank (rank 0's verdict is shared)."""

        lo, verdict = 0, []
        for width in self.widths:
            windows = self._windows(lo + 1, st)
            eager = best_ms(lambda: multi_tree_forward(self.w, windows, full_logits=self.full_logits, tp=self.tp,
                                                       capture_taps=self.taps))
            key = (width, slot_class(0))
            replay = best_ms(lambda: self._replay(key, windows))
            self.timings[width] = (round(eager, 2), round(replay, 2))
            verdict.append(int(replay >= eager))
            lo = width
        flags = torch.tensor(verdict, dtype=torch.int32, device=self.w.norm.device)
        if world > 1:
            import torch.distributed as dist

            dist.broadcast(flags, 0)
        self.slower = {w for w, slow in zip(self.widths, flags.tolist()) if slow}
        self.replays = 0
        if self.log is not None:
            self.log("[tensorfold] concurrent-round graphs, eager vs replay ms by width: " + ", ".join(
                f"{w}: {e:.1f}/{r:.1f}" for w, (e, r) in self.timings.items()) +
                (f"; eager at {sorted(self.slower)}" if self.slower else ""))

    def used(self) -> tuple[int, ...]:
        return tuple(x for x in self.widths if x not in self.slower)

    @torch.no_grad()
    def trimmed(self, st: State) -> "RoundGraphs | None":
        """Only the widths that replay faster, captured afresh in a new pool (the others' memory goes back); None when
        none is faster. Every rank calls it alike (``slower`` is shared)."""

        used = self.used()
        if len(used) == len(self.widths):
            return self
        self.entries.clear()
        self.outputs = None
        gc.collect()
        torch.cuda.empty_cache()
        if not used:
            return None
        fresh = RoundGraphs(self.w, self.lanes, self.context, tp=self.tp, taps=self.taps,
                            full_logits=self.full_logits, widths=used, classes=self.classes, log=self.log)
        fresh.capture_all(st)
        fresh.timings = self.timings
        return fresh
