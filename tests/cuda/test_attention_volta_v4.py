"""sm_70 verify attention, v4 tile (coalesced row-major values, fp16 int8 dequant): a row's bits do not depend on the
window's other rows or on other streams, and its error against a float64 reference over the values the int8 rows hold
is no worse than the v2 kernel's."""

from __future__ import annotations

import os

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7:
    pytest.skip("sm_70 only", allow_module_level=True)

from tensorfold.cuda.kernels import attention as ta  # noqa: E402
from tensorfold.cuda.kernels import kvpack  # noqa: E402

H, HK, D = 6, 1, 256


def _exact(packed: torch.Tensor) -> torch.Tensor:
    codes = packed[..., :D].double()
    scales = packed[..., D:].contiguous().view(torch.float16).double()
    rot = (codes.view(*codes.shape[:-1], D // 32, 32) + 0.5) * scales[..., None] / 128
    i = torch.arange(32, device=packed.device)
    bits = (i[:, None] & i[None, :]).to(torch.int64)
    pop = sum((bits >> b) & 1 for b in range(5))
    h = (1 - 2 * (pop % 2)).double() / 32 ** 0.5
    return (rot @ h).reshape(*codes.shape[:-1], D)


def _case(p: int, rows: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    k = kvpack.pack((torch.randn(p + 64, HK, D, generator=g, device="cuda") * 0.5).to(torch.bfloat16))
    v = kvpack.pack(torch.randn(p + 64, HK, D, generator=g, device="cuda").to(torch.bfloat16))
    q = (torch.randn(rows, H, D, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    kn = kvpack.pack((torch.randn(rows, HK, D, generator=g, device="cuda") * 0.5).to(torch.bfloat16))
    vn = kvpack.pack(torch.randn(rows, HK, D, generator=g, device="cuda").to(torch.bfloat16))
    return k, v, q, kn, vn


def _attend(k, v, q, kn, vn, p, version, streams=1):
    os.environ["TF_DECODE_ATTN"] = version
    try:
        rows = q.shape[0]
        plan = ta.plan([list(range(-1, rows - 1))] * streams, [p] * streams, H // HK, "cuda", tile=ta.tile_for("int8"))
        offs = torch.tensor(ta.offsets([(k, v)] * streams, "cuda"), dtype=torch.int64, device="cuda").view(streams, 2)
        rep = lambda t: torch.cat([t] * streams)                          # noqa: E731
        return ta.attention(rep(q), rep(kn), rep(vn), offs, plan, scale=D ** -0.5)
    finally:
        os.environ.pop("TF_DECODE_ATTN", None)


@pytest.mark.parametrize("p", [13, 2047, 5000])
def test_a_rows_bits_do_not_depend_on_other_rows_or_streams(p):
    k, v, q, kn, vn = _case(p, 12, 1)
    window = _attend(k, v, q, kn, vn, p, "v4")
    alone = torch.cat([_attend(k, v, q[:1], kn[:1], vn[:1], p, "v4")])
    assert torch.equal(window[:1], alone)                    # the first row alone == among the window's twelve
    many = _attend(k, v, q, kn, vn, p, "v4", streams=3)
    assert torch.equal(many[:12], window) and torch.equal(many[24:], window)


@pytest.mark.parametrize("p", [600, 5000])
def test_error_against_float64_no_worse_than_v2(p):
    k, v, q, kn, vn = _case(p, 8, 2)
    keys = torch.cat([_exact(k[:p]), _exact(kn)])[:, 0]       # (p + rows, D) in the model's basis
    vals = torch.cat([_exact(v[:p]), _exact(vn)])[:, 0]
    ref = torch.empty(8, H, D, dtype=torch.float64, device="cuda")
    for r in range(8):
        s = (q[r].double() @ keys[:p + r + 1].T) * D ** -0.5
        ref[r] = s.softmax(-1) @ vals[:p + r + 1]
    e4 = (_attend(k, v, q, kn, vn, p, "v4").double() - ref).abs()
    e2 = (_attend(k, v, q, kn, vn, p, "v2").double() - ref).abs()
    print(f"p {p}: v4 mean {float(e4.mean()):.3e} max {float(e4.max()):.3e}; v2 mean {float(e2.mean()):.3e} "
          f"max {float(e2.max()):.3e}")
    assert e4.mean() <= e2.mean() and e4.max() < 2e-2
