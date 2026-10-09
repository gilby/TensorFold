"""sm_70 fused causal prompt attention: any chunking or row tiling gives the same bits (bf16 and packed int8 caches),
and its error against a float64 reference is no worse than the GEMM-blocked kernel's (or, for int8, than attention on
the caches unpacked to bf16 through that kernel)."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7:
    pytest.skip("sm_70 only", allow_module_level=True)

from tensorfold.cuda.kernels import kvpack  # noqa: E402
from tensorfold.cuda.kernels import prefill_attention as pa  # noqa: E402

SCALE = 256 ** -0.5


def _inputs(n, h=24, hk=4, d=256, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(n, h, d, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    k = (torch.randn(n, hk, d, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    v = torch.randn(n, hk, d, generator=g, device="cuda").to(torch.bfloat16)
    return q, k, v


def _reference(q, k, v):
    n, h, d = q.shape
    hk = k.shape[1]
    kk = k.double().repeat_interleave(h // hk, dim=1)
    vv = v.double().repeat_interleave(h // hk, dim=1)
    s = torch.einsum("qhd,khd->hqk", q.double(), kk) * SCALE
    s = s.masked_fill(torch.ones(n, n, device="cuda").triu(1).bool(), float("-inf"))
    return torch.einsum("hqk,khd->qhd", s.softmax(-1), vv)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("cuts", [[0, 3000], [0, 1, 31, 33, 3000], [0, 1500, 2048, 2049, 3000]])
def test_any_chunking_gives_the_same_bits(packed, cuts):
    q, k, v = _inputs(3000, seed=1)
    if packed:
        k, v = kvpack.pack(k), kvpack.pack(v)
    whole = pa.fused_attention(q, k, v, 0, scale=SCALE)
    parts = torch.cat([pa.fused_attention(q[a:b], k, v, a, scale=SCALE) for a, b in zip(cuts, cuts[1:])])
    assert torch.equal(whole, parts)


def test_bf16_error_no_worse_than_blocked():
    q, k, v = _inputs(2600, seed=3)
    ref = _reference(q, k, v)
    fused = (pa.fused_attention(q, k, v, 0, scale=SCALE).double() - ref).abs()
    blocked = (pa.blocked_attention(q, k, v, 0, scale=SCALE, hk=4).double() - ref).abs()
    assert fused.max() < 2e-2
    assert fused.mean() <= 1.05 * blocked.mean()


def _exact(packed: torch.Tensor, d: int = 256) -> torch.Tensor:
    """The values packed int8 rows hold, in float64 and the model's basis: (code + 0.5) * s / 128, then H32."""

    codes = packed[..., :d].double()
    scales = packed[..., d:].contiguous().view(torch.float16).double()
    rot = ((codes.view(*codes.shape[:-1], d // 32, 32) + 0.5) * scales[..., None] / 128)
    i = torch.arange(32, device=packed.device)
    bits = (i[:, None] & i[None, :]).to(torch.int64)
    pop = torch.zeros_like(bits)
    for b in range(5):
        pop += (bits >> b) & 1
    h = (1 - 2 * (pop % 2)).double() / 32 ** 0.5
    return (rot @ h).reshape(*codes.shape[:-1], d)


def test_int8_read_packed_no_worse_than_unpacked():
    q, k, v = _inputs(2600, seed=4)
    pk, pv = kvpack.pack(k), kvpack.pack(v)
    ref = _reference(q, _exact(pk), _exact(pv))               # the values the int8 rows hold, exactly
    fused = (pa.fused_attention(q, pk, pv, 0, scale=SCALE).double() - ref).abs()
    uk, uv = kvpack.unpack(pk, 256), kvpack.unpack(pv, 256)
    blocked = (pa.blocked_attention(q, uk, uv, 0, scale=SCALE, hk=4).double() - ref).abs()
    print("fused", float(fused.mean()), float(fused.max()), "blocked", float(blocked.mean()), float(blocked.max()))
    assert fused.max() < 2e-2
    assert fused.mean() <= 1.05 * blocked.mean()
