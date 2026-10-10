"""Runtime LoRA adapters on the 27B's prompt path: products, row invariance and rank shards."""

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen3_5.cuda import adapters  # noqa: E402
from tensorfold.families.qwen3_5.cuda.distributed import output_rows  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _factors(out, inp, rank, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(rank, inp, generator=g) * inp ** -0.5).to(torch.bfloat16)
    b = (torch.randn(out, rank, generator=g) * 0.02).to(torch.bfloat16)
    return a, b


def _reference(x, a, b):
    mid = (x.float() @ a.float().t()).to(torch.bfloat16)
    return mid.float() @ b.float().t()


@pytest.mark.parametrize("volta", [False, True])
@cuda
def test_lora_matches_fp32(volta):
    if volta and torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("the sm_70 path")
    a, b = _factors(1024, 5120, 256)
    x = (torch.randn(300, 5120) * 0.5).to(torch.bfloat16)
    lora = adapters.Lora(a.cuda(), b.cuda(), volta)
    got = lora(x.cuda()).cpu()
    ref = _reference(x, a, b)
    assert ((got - ref).norm() / ref.norm()).item() < 2e-3


@pytest.mark.parametrize("volta", [False, True])
@cuda
def test_lora_rows_independent_of_count(volta):
    if volta and torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("the sm_70 path")
    a, b = _factors(512, 5120, 256, seed=1)
    x = (torch.randn(1100, 5120) * 0.5).to(torch.bfloat16).cuda()
    lora = adapters.Lora(a.cuda(), b.cuda(), volta)
    whole = lora(x)
    for lo, hi in ((0, 1), (3, 260), (257, 1100), (1000, 1100)):
        assert torch.equal(lora(x[lo:hi]), whole[lo:hi])


def test_rank_shards_add_up():
    world = 4
    a, b = _factors(2048 + 2048 + 6144, 5120, 256, seed=2)
    x = torch.randn(8, 5120)
    full = x @ a.float().t() @ b.float().t()
    # column-parallel (GDN q|k|v segments): each rank's B rows, A whole
    cols = [output_rows(b.shape[0], r, world, segments=(2048, 2048, 6144)) for r in range(world)]
    parts = [x @ a.float().t() @ b.float()[rows].t() for rows in cols]
    order = torch.cat(cols)
    assert torch.allclose(torch.cat(parts, dim=1), full[:, order], atol=1e-5)
    # row-parallel (an out projection): each rank's A columns, B whole; the rank sum is the whole product
    a2, b2 = _factors(5120, 6144, 256, seed=3)
    x2 = torch.randn(8, 6144)
    share = 6144 // world
    total = sum(x2[:, r * share:(r + 1) * share] @ a2.float()[:, r * share:(r + 1) * share].t() @ b2.float().t()
                for r in range(world))
    assert torch.allclose(total, x2 @ a2.float().t() @ b2.float().t(), atol=1e-4)


def test_applied_is_exclusive():
    ad = adapters.Adapter("x", {}, [], {})
    with adapters.applied(ad):
        assert adapters.ACTIVE is ad
        with pytest.raises(RuntimeError):
            with adapters.applied(ad):
                pass
    assert adapters.ACTIVE is None
    with adapters.applied(None):
        assert adapters.ACTIVE is None
