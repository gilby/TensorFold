"""The EXL3 prompt matmul (``cuda/exl3/prefill.py``) on a GPU, on synthetic layers: every path a row can take gives
the same bits whatever the row count, and the paths that replace one another (tile changes, kernels that rebuild the
same decoded weights) give the bits of the path they replace."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorfold.cuda.exl3 import format as fmt
from tensorfold.cuda.exl3 import linear, prefill

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

ROWS = (1, 3, 16, 17, 33, 48, 49, 64, 128, 129, 300)


def _layer(codebook: str, bits: float, k: int, n: int, seed: int = 0, bias: bool = False) -> linear.Exl3Linear:
    """A synthetic group: random trellis words, small scales; K and N multiples of 128."""

    rng = np.random.default_rng(seed)
    trellis = torch.from_numpy(rng.integers(-2**15, 2**15, size=(k // 16, n // 16, fmt.tile_words(bits)))
                               .astype(np.int16))
    suh = torch.from_numpy((rng.standard_normal(k) * 0.05).astype(np.float16))
    svh = torch.from_numpy((rng.standard_normal(n) * 0.05).astype(np.float16))
    b = torch.from_numpy((rng.standard_normal(n) * 0.05).astype(np.float16)) if bias else None
    return linear.Exl3Linear.from_tensors(trellis, suh, svh, codebook, b)


def _rows(m: int, k: int, seed: int = 1) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((m, k), generator=gen, device="cuda").to(torch.bfloat16)


def _run(layer: linear.Exl3Linear, x: torch.Tensor, ws: prefill.Workspace) -> torch.Tensor:
    out = torch.empty((x.shape[0], layer.n), dtype=torch.bfloat16, device=x.device)
    return prefill.matmul(layer, x, out, ws)


@pytest.mark.parametrize("codebook,bits,k,n", [("mul1", 4, 5120, 6144), ("mul1", 3, 1024, 2048),
                                                ("3inst", 2, 512, 1024), ("mcg", 6, 1024, 512)])
def test_the_k_step_leaves_the_bits_unchanged(monkeypatch, codebook, bits, k, n):
    """K steps of 64 sum the same products in the same order as steps of 32: the same output bits."""

    layer = _layer(codebook, bits, k, n, bias=True)
    x = _rows(300, k)
    now = _run(layer, x, prefill.Workspace())
    monkeypatch.setattr(prefill, "tiles", lambda k, n: (128, 32, 8, 4, 8))
    assert torch.equal(_run(layer, x, prefill.Workspace()), now)
