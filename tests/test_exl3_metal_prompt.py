"""EXL3 prompt matmuls on the M5 tensor units: the format's layer, and a row's bits whatever the chunking."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from tensorfold.cuda.exl3 import format as fmt  # noqa: E402
from tensorfold.kernels.exl3 import experts, prompt, rows  # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def _tensor_units() -> bool:
    from tensorfold.kernels import device

    return device.tensor_units()


needs_units = pytest.mark.skipif(not _tensor_units(), reason="the prompt kernels use Metal 4 tensor units (M5)")


@needs_units
@pytest.mark.parametrize("bits,cb", [(4, "mcg"), (6, "mcg"), (3, "3inst"), (2.5, "mul1")])
def test_prompt_matmul_matches_the_format_at_any_chunking(bits, cb):
    rng = np.random.default_rng(int(4 * bits))
    k, sizes = 1024, [512, 256]
    trs = [rng.integers(-32768, 32767, size=(k // 16, n // 16, int(16 * bits)), dtype=np.int16) for n in sizes]
    suh = (rng.standard_normal((2, k)) * 0.5).astype(np.float16)
    svh = (rng.standard_normal(768) * 0.5).astype(np.float16)
    words = mx.array(np.concatenate(trs, axis=1)).view(mx.uint32).reshape(k // 16, 48, -1)
    st = rows.Stack(words, mx.array(suh), mx.array(svh), k, 768, int(2 * bits), {"3inst": 0, "mcg": 1, "mul1": 2}[cb],
                    tuple(sizes), layout="stored")
    x = mx.array(rng.standard_normal((150, k)).astype(np.float32)).astype(mx.bfloat16)
    y = prompt.matmul(st, x, out_dtype=mx.float32)
    xf = np.array(x.astype(mx.float32))
    ref = np.concatenate([fmt.forward(xf, trs[0], suh[0], svh[:512], bits, cb),
                          fmt.forward(xf, trs[1], suh[1], svh[512:], bits, cb)], axis=1)
    assert np.abs(np.array(y) - ref).max() <= 1e-3 * np.abs(ref).max()
    parts = mx.concatenate([prompt.matmul(st, x[a:b], out_dtype=mx.float32) for a, b in ((0, 1), (1, 64), (64, 150))])
    assert bool(mx.array_equal(parts, y).item())
    from tensorfold.kernels.exl3.linear import Exl3Linear

    first, second = Exl3Linear(st, 0), Exl3Linear(st, 1)
    xb = x[None]
    ya, yb = first(xb), second(xb)                       # one launch for both members of the stack
    assert bool(mx.array_equal(yb[0], prompt.matmul(st, x)[:, 512:]).item()) and ya.shape == (1, 150, 512)


@needs_units
def test_prompt_moe_is_the_decode_moe_and_chunk_invariant():
    E, D, I, TOPK = 256, 512, 256, 8
    rng = np.random.default_rng(4)
    words = lambda *s: mx.array(rng.integers(0, 2**32, size=s, dtype=np.uint64).astype(np.uint32))  # noqa: E731
    f16 = lambda *s: mx.array((rng.standard_normal(s) * 0.3).astype(np.float16))                     # noqa: E731
    ex = experts.Experts(words(E + 1, 2, D // 16, I // 16, 32), f16(E + 1, 2, D), f16(E + 1, 2, I),
                         words(E + 1, I // 16, D // 16, 32), f16(E + 1, I), f16(E + 1, D),
                         mx.array((rng.standard_normal((E + 1, D)) * 0.05).astype(np.float16)), D, I, E, TOPK, 8, 8, 1)
    x = mx.array(rng.standard_normal((130, D)).astype(np.float32)).astype(mx.bfloat16)
    y = prompt.moe(ex, x)
    logits = prompt.router_logits(x, ex)
    plain = np.array(x.astype(mx.float32)) @ np.array(ex.router.astype(mx.float32)).T
    assert np.abs(np.array(logits) - plain).max() <= 1e-4
    slots, weights, units = experts.route(logits[:20], ex)
    decode = experts.down(experts.gate_up(x[:20], slots, units, ex), slots, weights, units, ex).sum(axis=1)
    assert float(mx.abs(y[:20] - decode).max()) <= 1e-3 * float(mx.abs(decode).max())
    parts = mx.concatenate([prompt.moe(ex, x[a:b]) for a, b in ((0, 1), (1, 77), (77, 130))])
    assert bool(mx.array_equal(parts, y).item())
