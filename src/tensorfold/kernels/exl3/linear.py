"""EXL3 projections as MLX modules: decode rows through ``rows``, prompt chunks through ``prompt``."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from tensorfold.kernels.exl3 import rows, tiles


class Exl3Linear(nn.Module):
    """Projection ``index`` of an EXL3 stack (its own stack when it reads its rows alone)."""

    def __init__(self, stack: rows.Stack, index: int = 0) -> None:
        super().__init__()
        self._stack, self._index = stack, int(index)
        self._own = len(stack.sizes) == 1

    @property
    def stack(self) -> rows.Stack:
        return self._stack

    @property
    def outputs(self) -> int:
        return self._stack.sizes[self._index]

    def _columns(self, y: mx.array) -> mx.array:
        if self._own:
            return y
        start = sum(self._stack.sizes[:self._index])
        return y[..., start:start + self.outputs]

    def project_rows(self, x: mx.array) -> mx.array:
        """Row-exact rows [..., K] -> [..., N] (bf16): every row's bits are its one-row bits."""

        lead = x.shape[:-1]
        x2 = x.reshape(-1, int(x.shape[-1]))
        y = rows.project(self._stack, tiles.rotate(x2, self._stack.suh))
        return self._columns(y).reshape(*lead, self.outputs)

    def __call__(self, x: mx.array) -> mx.array:
        """Prompt rows: the stack's projections in one launch, kept for its other members reading the same ``x``."""

        from tensorfold.kernels.exl3 import prompt

        held = getattr(self._stack, "_prompt", None)
        if held is None or held[0] is not x or self._index in held[2]:
            y = prompt.matmul(self._stack, x.reshape(-1, int(x.shape[-1])))
            held = (x, y, set())
        held[2].add(self._index)
        self._stack._prompt = None if len(held[2]) == len(self._stack.sizes) else held
        return self._columns(held[1]).reshape(*x.shape[:-1], self.outputs).astype(x.dtype)

    def dequantized_rows(self, ids: np.ndarray, chunk: int = 8192) -> mx.array:
        """W.T rows [len(ids), K] fp32 for output columns ``ids`` (a draft head's rows)."""

        st = self._stack
        h = tiles.hadamard()
        base = sum(st.sizes[:self._index])
        wanted = np.asarray(ids, dtype=np.int64)
        out = []
        for n0 in range(0, self.outputs, chunk):
            nw = min(chunk, self.outputs - n0)
            sel = wanted[(wanted >= n0) & (wanted < n0 + nw)] - n0
            if not sel.size:
                continue
            wq = tiles.unpack(st.words, st.k, st.k2, st.cb, st.strides, base + n0, nw).astype(mx.float32)
            k = st.k
            w = (h @ wq.reshape(k // tiles.HAD, tiles.HAD, nw)).reshape(k, nw)          # H_K along the inputs
            w = w * st.suh[self._index].astype(mx.float32)[:, None]
            w = (w.reshape(k, nw // tiles.HAD, tiles.HAD) @ h).reshape(k, nw)           # H_N along the outputs
            w = w * st.svh[base + n0:base + n0 + nw].astype(mx.float32)[None, :]
            out.append(mx.take(w, mx.array(sel.astype(np.int32)), axis=1).T)
            mx.eval(out[-1])
        return mx.concatenate(out, axis=0)

    def draft_head(self, ids: np.ndarray, bits: int = 4, group_size: int = 64) -> nn.QuantizedLinear:
        """The rows of ``ids`` as an MLX affine head of their own (drafts only; verification keeps this layer)."""

        ids = np.asarray(ids)
        if np.any(np.diff(ids) <= 0):
            raise ValueError("draft head ids must be ascending")
        rows_ = self.dequantized_rows(ids).astype(mx.bfloat16)
        head = nn.QuantizedLinear(int(rows_.shape[1]), int(rows_.shape[0]), bias=False, group_size=group_size, bits=bits)
        head.weight, head.scales, head.biases = mx.quantize(rows_, group_size=group_size, bits=bits)
        mx.eval(head.weight, head.scales, head.biases)
        return head


__all__ = ["Exl3Linear"]
