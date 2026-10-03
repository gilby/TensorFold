"""The lane matmul at 5, 6 and 8 bits: exact values, rows independent of the row count, and 4-bit's arithmetic."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.kernels import qmm  # noqa: E402
from tensorfold.families.qwen3_5.cuda.qmm import dequantize  # noqa: E402

ROWS = [1, 2, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 128, 129, 200]
WIDTHS = [(5, 64), (6, 64), (8, 64), (5, 32), (6, 32), (8, 32)]


def _weights(n: int, k: int, gs: int, bits: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randint(0, 1 << bits, (n, k), generator=g, device="cuda", dtype=torch.int32)
    scales = (torch.rand((n, k // gs), generator=g, device="cuda") * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((n, k // gs), generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    return qmm.words_of(q, bits), scales, biases


@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_codes_read_the_mlx_bit_stream(bits):
    words, scales, biases = _weights(64, 512, 64, bits, bits)
    ones, zeros = torch.ones_like(scales), torch.zeros_like(biases)
    assert torch.equal(qmm.codes(words, bits).float(), dequantize(words, ones, zeros, bits, 64))
    assert torch.equal(qmm.words_of(qmm.codes(words, bits), bits), words)


@pytest.mark.parametrize("bits,gs", WIDTHS)
@pytest.mark.parametrize("n,k", [(48, 5120), (1024, 2048), (200, 1024)])
def test_pack_round_trip(bits, gs, n, k):
    w = _weights(n, k, gs, bits, n + k + bits)
    q = qmm.pack(*w, gs, bits=bits)
    assert q.weight.numel() == -(-n // 128) * 128 * k * bits // 32
    back = qmm.unpack(q)
    assert all(torch.equal(a, b) for a, b in zip(back, w))


@pytest.mark.parametrize("bits,gs", WIDTHS)
def test_every_value_decodes_exactly(bits, gs):
    """One-hot rows read back each stored value (scale 1, bias 0) in the decode, grouped and prompt kernels."""

    n, k = 256, 2048
    words = _weights(n, k, gs, bits, 29)[0]
    ones = torch.ones((n, k // gs), device="cuda").bfloat16()
    zeros = torch.zeros_like(ones)
    q = qmm.pack(words, ones, zeros, gs, bits=bits)
    want = dequantize(words, ones, zeros, bits, gs).T.contiguous()
    assert set(want.unique().tolist()) == set(range(1 << bits))
    eye = torch.eye(k, device="cuda").bfloat16()
    for f32 in (False, True):
        assert torch.equal(qmm.matmul(eye, q, f32=f32).float(), want), f32
        assert torch.equal(qmm.prefill_matmul(eye, q, f32=f32).float(), want), f32
        for m in (1, 8, 16, 40):
            assert torch.equal(qmm.matmul(eye[:m], q, f32=f32).float(), want[:m]), (f32, m)


@pytest.mark.parametrize("bits,gs", WIDTHS)
@pytest.mark.parametrize("n,k", [(48, 5120), (1024, 5120), (5120, 6144)])
def test_rows_do_not_depend_on_row_count(bits, gs, n, k):
    q = qmm.pack(*_weights(n, k, gs, bits, 3 * n + k), gs, bits=bits)
    x = torch.randn((max(ROWS), k), generator=torch.Generator(device="cuda").manual_seed(7), device="cuda").bfloat16()
    alone = torch.cat([qmm.matmul(x[r:r + 1], q) for r in range(max(ROWS))])
    for m in ROWS:
        assert torch.equal(qmm.matmul(x[:m], q), alone[:m]), f"{n}x{k}: rows differ at M={m}"
    perm = torch.randperm(max(ROWS), generator=torch.Generator().manual_seed(3)).cuda()
    assert torch.equal(qmm.matmul(x[perm], q), alone[perm])
    chunks = torch.cat([qmm.prefill_matmul(x[a:a + 37], q) for a in range(0, max(ROWS), 37)])
    assert torch.equal(qmm.prefill_matmul(x, q), chunks)


@pytest.mark.parametrize("bits,gs", WIDTHS)
def test_accuracy_matches_fp32_reference(bits, gs):
    n, k = 1024, 5120
    w = _weights(n, k, gs, bits, 11 * n + k + bits)
    x = torch.randn((16, k), device="cuda").bfloat16()
    q = qmm.pack(*w, gs, bits=bits)
    ref = x.float() @ dequantize(*w, bits, gs).T
    for y in (qmm.matmul(x, q).float(), qmm.prefill_matmul(x, q).float()):
        assert (y - ref).abs().max().item() <= ref.abs().max().item() * 2 ** -7


def test_same_values_same_bits_at_any_width():
    """Values below 16 stored at 5, 6 or 8 bits give the 4-bit kernel's bits: one arithmetic for every width."""

    n, k, gs = 1024, 5120, 64
    g = torch.Generator(device="cuda").manual_seed(5)
    codes = torch.randint(0, 16, (n, k), generator=g, device="cuda", dtype=torch.int32)
    scales = (torch.rand((n, k // gs), generator=g, device="cuda") * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((n, k // gs), generator=g, device="cuda") * 0.05).to(torch.bfloat16)
    x = torch.randn((40, k), device="cuda").bfloat16()
    base = qmm.pack(qmm.words_of(codes, 4), scales, biases, gs)
    for bits in (5, 6, 8):
        q = qmm.pack(qmm.words_of(codes, bits), scales, biases, gs, bits=bits)
        for m in (1, 16, 40):
            assert torch.equal(qmm.matmul(x[:m], q), qmm.matmul(x[:m], base)), (bits, m)
            assert torch.equal(qmm.matmul(x[:m], q, f32=True), qmm.matmul(x[:m], base, f32=True)), (bits, m)
        assert torch.equal(qmm.prefill_matmul(x, q), qmm.prefill_matmul(x, base)), bits


def test_group_of_mixed_widths_matches_each_alone():
    k = 5120
    parts = [qmm.pack(*_weights(n, k, 64, b, n + b), 64, bits=b) for n, b in ((6144, 4), (2048, 5), (128, 8), (96, 6))]
    x = torch.randn((33, k), device="cuda").bfloat16()
    for m in (1, 8, 33):
        got = qmm.matmul_group(x[:m], parts)
        for q, y in zip(parts, got):
            assert torch.equal(y, qmm.matmul(x[:m], q)), (q.bits, m)
