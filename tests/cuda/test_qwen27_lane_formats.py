"""The 27B's projections in every lane format: tiling, shapes, row slices and dispatch give the shared kernels' bits."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.build import volta  # noqa: E402
from tensorfold.cuda.kernels import qmm  # noqa: E402
from tensorfold.families.qwen3_5.cuda import qmm_fast  # noqa: E402
from tensorfold.families.qwen3_5.cuda.qmm import dequantize  # noqa: E402
from tensorfold.families.qwen3_5.cuda.weights import LANE_FORMATS, QLinear  # noqa: E402

if volta():
    pytest.skip("sm_70 tiles 4-bit groups of 64 only", allow_module_level=True)

FORMATS = sorted(LANE_FORMATS)


def _stored(n: int, k: int, bits: int, gs: int, seed: int) -> QLinear:
    g = torch.Generator(device="cuda").manual_seed(seed)
    codes = torch.randint(0, 1 << bits, (n, k), generator=g, device="cuda", dtype=torch.int32)
    scales = (torch.rand((n, k // gs), generator=g, device="cuda") * 0.02 + 0.001).bfloat16()
    biases = (torch.randn((n, k // gs), generator=g, device="cuda") * 0.05).bfloat16()
    return QLinear(qmm.words_of(codes, bits), scales, biases, gs=gs, bits=bits)


@pytest.mark.parametrize("bits,gs", FORMATS)
def test_tile_keeps_shape_and_round_trips(bits, gs):
    q = _stored(200, 2048, bits, gs, bits * gs)
    t = qmm_fast.tile(q)
    assert t.layout == "tiled" and (t.n, t.k, t.bits, t.gs) == (q.n, q.k, bits, gs)
    back = qmm_fast.untile(t)
    assert all(torch.equal(a, b) for a, b in zip((back.weight, back.scales, back.biases), (q.weight, q.scales, q.biases)))
    assert (back.bits, back.gs) == (bits, gs)


@pytest.mark.parametrize("bits,gs", FORMATS)
def test_matmul_reads_any_group_sums(bits, gs):
    """Callers pass sums of 64-input groups; a 32-input format computes its own, with the same bits as passing them."""

    q = qmm_fast.tile(_stored(1024, 5120, bits, gs, 3 + bits))
    x = torch.randn((16, 5120), device="cuda").bfloat16()
    want = qmm.matmul(x, q, qmm.group_sums(x, gs))
    assert torch.equal(qmm_fast.matmul(x, q, qmm.group_sums(x, 64)), want)
    assert torch.equal(qmm_fast.matmul(x, q), want)
    assert torch.equal(qmm_fast.matmul_partial(x, q, qmm.group_sums(x, 64)), qmm.matmul(x, q, f32=True))
    u = qmm_fast.untile(q)
    ref = x.float() @ dequantize(u.weight, u.scales, u.biases, bits, gs).T
    assert (want.float() - ref).abs().max().item() <= ref.abs().max().item() * 2 ** -7


@pytest.mark.parametrize("bits,gs", FORMATS)
def test_head_rows_are_the_stacked_bits(bits, gs):
    q = qmm_fast.tile(_stored(1024, 2048, bits, gs, 17 + bits))
    x = torch.randn((8, 2048), device="cuda").bfloat16()
    full = qmm_fast.matmul(x, q)
    for a, b in ((0, 256), (128, 512), (100, 300), (960, 1024)):
        part = qmm_fast.rows(q, a, b)
        assert (part.n, part.k, part.bits, part.gs) == (b - a, q.k, bits, gs)
        assert torch.equal(qmm_fast.matmul_rows(x, [part]), qmm.matmul(x, part, qmm.group_sums(x, gs),
                                                                      sk=qmm.split_k(b - a, q.k, gs)))
        assert torch.equal(qmm_fast.untile(part).weight, qmm_fast.untile(q).weight[a:b])
    assert torch.equal(full, qmm_fast.matmul(x, q))


def test_group_of_mixed_lane_formats():
    k = 5120
    parts = [qmm_fast.tile(_stored(n, k, b, 64, n + b)) for n, b in ((6144, 4), (2048, 5), (96, 6), (64, 8))]
    x = torch.randn((9, k), device="cuda").bfloat16()
    xs = qmm.group_sums(x, 64)
    got = qmm_fast.matmul_group(x, parts, xs)
    assert all(torch.equal(y, qmm_fast.matmul(x, q, xs)) for q, y in zip(parts, got))
    g32 = qmm_fast.tile(_stored(256, k, 4, 32, 5))
    got = qmm_fast.matmul_group(x, [parts[0], g32], xs)
    assert torch.equal(got[1], qmm_fast.matmul(x, g32))


def test_prompt_rows_take_the_prompt_kernel():
    from tensorfold.families.qwen3_5.cuda.prefill import _mm

    x = torch.randn((300, 5120), device="cuda").bfloat16()
    for bits, gs in ((5, 64), (8, 64), (6, 32)):
        q = qmm_fast.tile(_stored(1024, 5120, bits, gs, bits))
        assert torch.equal(_mm(x, q), qmm.prefill_matmul(x, q))
        chunks = torch.cat([_mm(x[a:a + 77], q) for a in range(0, 300, 77)])
        assert torch.equal(_mm(x, q), chunks)


def test_only_4_bit_groups_of_64_take_the_fp8_prompt_path():
    q4, q5 = _stored(64, 512, 4, 64, 1), _stored(64, 512, 5, 64, 2)
    assert q4.fast and q4.nibble and q5.fast and not q5.nibble
    assert not _stored(64, 512, 4, 32, 3).nibble
