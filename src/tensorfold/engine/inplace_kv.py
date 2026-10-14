"""KV buffers that decode writes in place with a Metal kernel, so pipelined steps need no second copy of the cache."""

from __future__ import annotations

import weakref
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from tensorfold.kernels.inputs import MIN_ELEMENTS, ints

# thread (word, row, head): rows [AT, AT + n) of DK and DV take rows [FIRST, FIRST + n) of SK and SV, 16 bytes a thread
_WRITE = r"""
  const uint w = thread_position_in_grid.x;
  const int r = int(thread_position_in_grid.y), h = int(thread_position_in_grid.z);
  const int AT = META[0], FIRST = META[1];
  const size_t to = (size_t(h) * DK_shape[2] + AT + r) * W + w;
  const size_t from = (size_t(h) * SK_shape[2] + FIRST + r) * W + w;
  ((device uint4*)DK)[to] = ((const device uint4*)SK)[from];
  ((device uint4*)DV)[to] = ((const device uint4*)SV)[from];
  if (w == 0 && r == 0 && h == 0) DONE[0] = AT;
"""

_kernels: dict[int, Any] = {}
_none: list[mx.array] = []


def _kernel(words: int) -> Any:
    found = _kernels.get(words)
    if found is None:
        found = _kernels[words] = mx.fast.metal_kernel(
            name=f"tf_kv_write_{words}", input_names=["DK", "DV", "SK", "SV", "META", "AFTER"],
            output_names=["DONE"], source=f"  constexpr int W = {words};\n" + _WRITE)
    return found


def no_write() -> mx.array:
    """The marker that stands for no earlier write (an evaluated input orders nothing)."""

    if not _none:
        _none.append(mx.zeros((MIN_ELEMENTS,), dtype=mx.int32))
        mx.eval(_none[0])
    return _none[0]


def write_rows(dk: mx.array, dv: mx.array, sk: mx.array, sv: mx.array, at: int, first: int, count: int,
               after: mx.array | None) -> mx.array:
    """Rows [first, first + count) of sk, sv into rows at .. of dk, dv in place, after ``after``: the write's marker."""

    if dk.shape != dv.shape or sk.shape != sv.shape or dk.shape[0::3] != sk.shape[0::3] or dk.shape[1] != sk.shape[1]:
        raise ValueError(f"write_rows: buffers {dk.shape} {dv.shape}, rows {sk.shape} {sv.shape}")
    if not (dk.dtype == dv.dtype == sk.dtype == sv.dtype) or (int(dk.shape[3]) * dk.dtype.size) % 16:
        raise ValueError("write_rows: one dtype, and rows of whole 16-byte words")
    if at + count > int(dk.shape[2]) or first + count > int(sk.shape[2]):
        raise ValueError("write_rows: rows past a buffer's end")
    words = int(dk.shape[3]) * dk.dtype.size // 16
    heads = int(dk.shape[0]) * int(dk.shape[1])
    return _kernel(words)(inputs=[dk, dv, sk, sv, ints((at, first)), no_write() if after is None else after],
                          grid=(words, count, heads), threadgroup=(min(words, 256), 1, 1),
                          output_shapes=[(MIN_ELEMENTS,)], output_dtypes=[mx.int32])[0]


class InplaceKVCache(KVCache):
    """Decode rows go into the cache's own buffers in place; a pending step reads only rows below every later write.

    A step's attention reads its earlier keys (rows below the write's first row) from the buffers and its new rows
    from the write's inputs, so rows a step in flight reads are never rewritten while the next write starts at or
    above them (``_floor``). A write starting lower (a rollback past the last write, or rows a copy holds) and a
    buffer this cache did not make (a copy, a stored prefix) are copied into new buffers first.
    """

    grow = 2048                     # capacity added at a time
    _owner: Any = None              # a weak reference to the cache whose buffers these are (a shallow copy's differs)
    _done: mx.array | None = None   # the last in-place write's marker: readers and the next write take it as an input
    _floor = 0                      # rows below are read by a step in flight or held by a copy
    # left out of prefix snapshots: a stored or loaded cache owns nothing
    transient = ("_owner", "_done", "_floor")

    def write(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array, mx.array | None]:
        """Decode rows [B, Hk, n, D] at offset ..: (keys, values, marker); readers of the buffers take the marker."""

        if mx.default_device() != mx.gpu:          # the write is a Metal kernel: elsewhere an ordinary update
            self.update_and_fetch(keys, values)
            return self.keys, self.values, None
        start, rows = self.offset, int(keys.shape[2])
        if not self._owns() or start < self._floor or int(self.keys.shape[2]) < start + rows:
            self._own(keys, values, start, start + rows)
        self._done = write_rows(self.keys, self.values, keys, values, start, 0, rows, self._done)
        self._floor, self.offset = start, start + rows
        return self.keys, self.values, self._done

    def _owns(self) -> bool:
        return self.keys is not None and self._owner is not None and self._owner() is self

    def _own(self, keys: mx.array, values: mx.array, start: int, end: int) -> None:
        """New buffers of whole growth steps holding rows [0, start), this cache's own."""

        like_k, like_v = (keys, values) if self.keys is None else (self.keys, self.values)
        capacity = -(-end // self.grow) * self.grow
        batch, heads = int(like_k.shape[0]), int(like_k.shape[1])
        new_k = mx.zeros((batch, heads, capacity, int(like_k.shape[3])), dtype=like_k.dtype)
        new_v = mx.zeros((batch, heads, capacity, int(like_v.shape[3])), dtype=like_v.dtype)
        if start and self.keys is not None:
            self._done = write_rows(new_k, new_v, self.keys, self.values, 0, 0, start, self._done)
        self.keys, self.values = new_k, new_v
        self._owner, self._floor = weakref.ref(self), 0

    def _settle(self) -> None:
        """Wait for the last in-place write: MLX ops read the buffers without its marker."""

        if self._done is not None:
            mx.eval(self._done)
            self._done = None

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        """mlx_lm's write (prompt chunks): a new or donated buffer nothing else holds, read below the offset."""

        self._settle()
        out = super().update_and_fetch(keys, values)
        self._owner, self._floor = weakref.ref(self), self.offset
        return out

    def materialize(self) -> None:
        """Before a copy or a store shares the buffers: every write landed, and rows below the offset stay as they are."""

        self._settle()
        self._floor = max(self._floor, self.offset)

    @property
    def state(self) -> tuple:
        return super().state

    @state.setter
    def state(self, v: tuple) -> None:
        self._settle()
        KVCache.state.fset(self, v)      # (keys, values) under mlx-lm 0.31, (keys, values, offset) under 0.32
        self._owner, self._floor = None, 0

    def drop_spare(self) -> None:
        """No spare buffer to drop (``drop_spares`` callers)."""


__all__ = ["InplaceKVCache", "no_write", "write_rows"]
