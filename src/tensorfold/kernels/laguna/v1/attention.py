"""Decode attention over each row's own keys: Gemma's kernels, with as many simdgroups a head as the group allows.

Laguna's sliding layers have 72 query heads over 8 key heads (9 a group): Gemma's 4 simdgroups a query head would
need 1,152 threads a threadgroup. The split is fixed by the layer's shape, never by the rows, so every row keeps its
bits at every row count and stream mix.
"""

from __future__ import annotations

import mlx.core as mx

from tensorfold.kernels.gemma.v1 import attention as gemma
from tensorfold.kernels.gemma.v1.attention import Rows
from tensorfold.kernels.inputs import MIN_ELEMENTS

MAX_THREADS = 1024


def split_for(heads: int, kv_heads: int, dims: int) -> int:
    """Simdgroups a query head: Gemma's for the head size, halved until a threadgroup fits."""

    group = heads // kv_heads
    split = gemma.SHAPES.get(dims, gemma.OTHER)[1]
    while split > 1 and 32 * group * split > MAX_THREADS:
        split //= 2
    if 32 * group * split > MAX_THREADS:
        raise ValueError(f"attend: {group} query heads a key head do not fit a threadgroup")
    return split


def attend(q: mx.array, keys: mx.array, values: mx.array, rows: Rows, new_keys: mx.array, new_values: mx.array,
           scale: float = 1.0, after: mx.array | None = None) -> mx.array:
    """q [R, H, D] over its rows' new keys [Hk, R, D] and the earlier ones in [1, Hk, CAP, D] buffers: [R, H, D].

    ``after``: the marker of an in-place write to the buffers, which the partials wait for."""

    count, heads, dims = (int(s) for s in q.shape)
    kv_heads = int(keys.shape[1])
    if dims % 32 or heads % kv_heads or rows.count != count:
        raise ValueError(f"attend: head dim a multiple of 32, heads a multiple of key heads, {count} rows")
    chunk, _, block = gemma.SHAPES.get(dims, gemma.OTHER)
    split = split_for(heads, kv_heads, dims)
    group = heads // kv_heads
    consts = (("D", dims), ("G", group), ("HK", kv_heads), ("CK", chunk), ("S", split), ("BLK", block),
              ("SCALE", float(scale)))
    slots = heads * count * rows.chunks
    inputs = [q, keys, values, new_keys, new_values, rows.positions, rows.lows, rows.meta]
    partial = gemma._partial if after is None else gemma._partial_after
    pm, pl, po = partial(consts, inputs=inputs if after is None else inputs + [after],
                         grid=(32 * group * split * rows.chunks, kv_heads, count),
                         threadgroup=(32 * group * split, 1, 1),
                         output_shapes=[(max(slots, MIN_ELEMENTS),), (max(slots, MIN_ELEMENTS),), (slots, dims)],
                         output_dtypes=[mx.float32, mx.float32, mx.float32])
    return gemma._merge((("D", dims), ("H", heads)), inputs=[pm, pl, po, rows.meta],
                        grid=(32, heads, count), threadgroup=(32, 1, 1),
                        output_shapes=[(count, heads, dims)], output_dtypes=[mx.bfloat16])[0]


__all__ = ["Rows", "attend", "split_for"]
