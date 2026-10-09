"""sm_70 prompt attention as GEMMs over absolute key blocks: chunking never changes bits; values match a reference."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 7:
    pytest.skip("sm_70 only", allow_module_level=True)

from tensorfold.cuda.kernels import prefill_attention as pa  # noqa: E402


def _inputs(n, h=24, hk=4, d=256, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(n, h, d, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    k = (torch.randn(n, hk, d, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    v = torch.randn(n, hk, d, generator=g, device="cuda").to(torch.bfloat16)
    return q, k, v


@pytest.mark.parametrize("cuts", [[0, 5000], [0, 1, 2047, 2049, 5000], [0, 3000, 4096, 5000]])
def test_any_chunking_gives_the_same_bits(cuts):
    q, k, v = _inputs(5000)
    hk = k.shape[1]
    whole = pa.blocked_attention(q, k, v, 0, scale=256 ** -0.5, hk=hk)
    parts = torch.cat([pa.blocked_attention(q[a:b], k, v, a, scale=256 ** -0.5, hk=hk) for a, b in zip(cuts, cuts[1:])])
    assert torch.equal(whole, parts)


def test_matches_a_float64_reference():
    q, k, v = _inputs(2600, seed=3)
    n, h, d = q.shape
    hk = k.shape[1]
    got = pa.blocked_attention(q, k, v, 0, scale=d ** -0.5, hk=hk).double()
    kk = k.double().repeat_interleave(h // hk, dim=1)
    vv = v.double().repeat_interleave(h // hk, dim=1)
    s = torch.einsum("qhd,khd->hqk", q.double(), kk) * d ** -0.5
    s = s.masked_fill(torch.ones(n, n, device="cuda").triu(1).bool(), float("-inf"))
    ref = torch.einsum("hqk,khd->qhd", s.softmax(-1), vv)
    assert (got - ref).abs().max() < 2e-2
    memeff = pa.volta_attention(q, k, v, 0, scale=d ** -0.5, hk=hk).double()
    assert (got - ref).abs().mean() <= 1.5 * (memeff - ref).abs().mean() + 1e-4


@pytest.mark.parametrize("cuts", [[0, 5000], [0, 1, 2047, 2049, 5000], [0, 3000, 4096, 5000]])
def test_half_scores_any_chunking_gives_the_same_bits(cuts):
    q, k, v = _inputs(5000)
    hk = k.shape[1]
    whole = pa.blocked_attention(q, k, v, 0, scale=256 ** -0.5, hk=hk, half_scores=True)
    parts = torch.cat([pa.blocked_attention(q[a:b], k, v, a, scale=256 ** -0.5, hk=hk, half_scores=True)
                       for a, b in zip(cuts, cuts[1:])])
    assert torch.equal(whole, parts)


def test_half_scores_error_against_float64():
    q, k, v = _inputs(2600, seed=3)
    n, h, d = q.shape
    hk = k.shape[1]
    kk = k.double().repeat_interleave(h // hk, dim=1)
    vv = v.double().repeat_interleave(h // hk, dim=1)
    s = torch.einsum("qhd,khd->hqk", q.double(), kk) * d ** -0.5
    s = s.masked_fill(torch.ones(n, n, device="cuda").triu(1).bool(), float("-inf"))
    ref = torch.einsum("hqk,khd->qhd", s.softmax(-1), vv)
    half = (pa.blocked_attention(q, k, v, 0, scale=d ** -0.5, hk=hk, half_scores=True).double() - ref).abs()
    full = (pa.blocked_attention(q, k, v, 0, scale=d ** -0.5, hk=hk).double() - ref).abs()
    memeff = (pa.volta_attention(q, k, v, 0, scale=d ** -0.5, hk=hk).double() - ref).abs()
    print("half", float(half.mean()), float(half.max()), "fp32 scores", float(full.mean()), float(full.max()),
          "memeff", float(memeff.mean()), float(memeff.max()))
    assert half.max() < 2e-2
    assert half.mean() <= 1.5 * memeff.mean()
