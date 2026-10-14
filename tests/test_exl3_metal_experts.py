"""EXL3 routed experts on Metal: the route, gate/up and down match a float64 MoE; each row equals its solo run."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from tensorfold.cuda.exl3 import format as fmt  # noqa: E402
from tensorfold.kernels.exl3 import experts, glue, tiles  # noqa: E402
from tensorfold.kernels.qwen.dense.v1 import row_glue  # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

E, D, I, TOPK = 32, 512, 256, 4


def _words(a: np.ndarray) -> mx.array:
    return mx.array(a).view(mx.uint32).reshape(*a.shape[:-1], -1)


@pytest.fixture(scope="module")
def layer():
    rng = np.random.default_rng(3)
    tr = lambda k, n: rng.integers(-32768, 32767, size=(k // 16, n // 16, 64), dtype=np.int16)  # noqa: E731
    gate, up, dn = ([tr(D, I) for _ in range(E + 1)], [tr(D, I) for _ in range(E + 1)],
                    [tr(I, D) for _ in range(E + 1)])
    suh_gu = (rng.standard_normal((E + 1, 2, D)) * 0.3).astype(np.float16)
    svh_gu = (rng.standard_normal((E + 1, 2, I)) * 0.3).astype(np.float16)
    suh_d = (rng.standard_normal((E + 1, I)) * 0.3).astype(np.float16)
    svh_d = (rng.standard_normal((E + 1, D)) * 0.3).astype(np.float16)
    router = (rng.standard_normal((E + 1, D)) * 0.05).astype(np.float16)
    ex = experts.Experts(_words(np.stack([np.stack(p) for p in zip(gate, up)])), mx.array(suh_gu), mx.array(svh_gu),
                         _words(np.stack(dn)), mx.array(suh_d), mx.array(svh_d), mx.array(router), D, I, E, TOPK, 8, 8, 1)
    return ex, (gate, up, dn, suh_gu, svh_gu, suh_d, svh_d, router)


def _reference(x: np.ndarray, raw) -> np.ndarray:
    gate, up, dn, suh_gu, svh_gu, suh_d, svh_d, router = raw
    out = []
    for row in x:
        lg = row.astype(np.float64) @ router.astype(np.float64).T
        p = np.exp(lg[:E] - lg[:E].max())
        p /= p.sum()
        ids = list(np.argsort(-p, kind="stable")[:TOPK])
        w = list(p[ids] / p[ids].sum()) + [1 / (1 + np.exp(-lg[E]))]
        y = 0.0
        for e, wt in zip(ids + [E], w):
            g = fmt.forward(row[None], gate[e], suh_gu[e, 0], svh_gu[e, 0], 4, "mcg")
            u = fmt.forward(row[None], up[e], suh_gu[e, 1], svh_gu[e, 1], 4, "mcg")
            y = y + wt * fmt.forward(g / (1 + np.exp(-g)) * u, dn[e], suh_d[e], svh_d[e], 4, "mcg")
        out.append(y[0])
    return np.stack(out)


def _moe(ex, x, logits=None):
    if logits is None:
        logits = x.astype(mx.float32) @ ex.router.astype(mx.float32).T
    slots, weights, units = experts.route(logits, ex)
    return experts.down(experts.gate_up(x, slots, units, ex), slots, weights, units, ex), slots


def test_experts_match_a_float64_moe(layer):
    ex, raw = layer
    rng = np.random.default_rng(5)
    x = mx.array(rng.standard_normal((7, D)).astype(np.float32)).astype(mx.bfloat16)
    ye, slots = _moe(ex, x)
    ref = _reference(np.array(x.astype(mx.float32)), raw)
    assert np.abs(np.array(ye).sum(axis=1) - ref).max() <= 2e-3 * np.abs(ref).max()
    assert (np.array(slots)[:, -1] == E).all()


def test_each_row_equals_its_solo_run(layer):
    ex, _ = layer
    rng = np.random.default_rng(9)
    x = mx.array(rng.standard_normal((12, D)).astype(np.float32)).astype(mx.bfloat16)
    x = mx.concatenate([x, x[:3]])                       # repeated rows share every expert
    logits = mx.array(np.array(x.astype(mx.float32)) @ np.array(ex.router.astype(mx.float32)).T)
    ye, _ = _moe(ex, x, logits)
    for i in range(int(x.shape[0])):
        solo, _ = _moe(ex, x[i:i + 1], logits[i:i + 1])
        assert bool(mx.array_equal(solo[0], ye[i]).item()), i


def test_norm_reads_the_experts_and_rotates_the_next_inputs():
    rng = np.random.default_rng(5)
    R = 5
    h = mx.array(rng.standard_normal((R, 2048)).astype(np.float32)).astype(mx.bfloat16)
    w = mx.array((1 + 0.1 * rng.standard_normal(2048)).astype(np.float32)).astype(mx.bfloat16)
    ye = mx.array((rng.standard_normal((R, 9, 2048)) * 0.1).astype(np.float32))
    suh = mx.array((rng.standard_normal((2, 2048)) * 0.5).astype(np.float16))
    ho, xo, xh = glue.norm_rotate(h, w, 1e-6, ye=ye, suh=suh)
    ref = np.array(h.astype(mx.float32)) + np.array(ye).sum(axis=1)
    assert np.abs(np.array(ho.astype(mx.float32)) - ref).max() <= 2 ** -6 * np.abs(ref).max()
    assert bool(mx.array_equal(xh, tiles.rotate(xo, suh)).item())
    router = mx.array((rng.standard_normal((257, 2048)) * 0.05).astype(np.float16))
    _, xo2, lg = glue.norm_router(h, w, 1e-6, router, ye=ye)
    assert bool(mx.array_equal(xo2, xo).item())
    plain = np.array(xo.astype(mx.float32)) @ np.array(router.astype(mx.float32)).T
    assert np.abs(np.array(lg) - plain).max() <= 1e-4
    for i in range(R):
        assert bool(mx.array_equal(glue.norm_router(h[i:i + 1], w, 1e-6, router, ye=ye[i:i + 1])[2][0], lg[i]).item())


def test_gdn_post_rotate_is_gdn_post_then_rotate():
    rng = np.random.default_rng(2)
    W, nv, dv = 3, 4, 128
    rec = mx.array(rng.standard_normal((1, W, nv, dv)).astype(np.float32)).astype(mx.bfloat16)
    y = mx.array(rng.standard_normal((1, W, 2 * nv * dv)).astype(np.float32)).astype(mx.bfloat16)
    nw = mx.array((1 + 0.1 * rng.standard_normal(dv)).astype(np.float32)).astype(mx.bfloat16)
    suh = mx.array((rng.standard_normal((1, nv * dv)) * 0.5).astype(np.float16))
    fused = glue.gdn_post_rotate(rec, y, nw, 1e-6, suh, zo=nv * dv)
    plain = row_glue.gdn_post(rec, y, nw, 1e-6, zo=nv * dv)
    assert bool(mx.array_equal(fused, tiles.rotate(plain[0], suh)).item())


def test_routing_inside_gate_up_is_the_route_launch(layer):
    ex, _ = layer
    rng = np.random.default_rng(13)
    x = mx.array(rng.standard_normal((6, D)).astype(np.float32)).astype(mx.bfloat16)
    logits = mx.array(np.array(x.astype(mx.float32)) @ np.array(ex.router.astype(mx.float32)).T)
    gu, slots, weights, units = experts.route_gate_up(x, logits, ex)
    s2, w2, u2 = experts.route(logits, ex)
    assert bool(mx.array_equal(slots, s2).item()) and bool(mx.array_equal(weights, w2).item())
    count = int(u2[0].item())
    assert bool(mx.array_equal(units[:count + 1], u2[:count + 1]).item())
    assert bool(mx.array_equal(gu, experts.gate_up(x, s2, u2, ex)).item())


def test_norm_router_launch_is_norm_then_router():
    rng = np.random.default_rng(8)
    R = 7
    h = mx.array(rng.standard_normal((R, 2048)).astype(np.float32)).astype(mx.bfloat16)
    w = mx.array((1 + 0.1 * rng.standard_normal(2048)).astype(np.float32)).astype(mx.bfloat16)
    z = mx.array((rng.standard_normal((4, R, 2048)) * 0.1).astype(np.float32))
    svh = mx.array(rng.standard_normal(2048).astype(np.float16))
    router = mx.array((rng.standard_normal((257, 2048)) * 0.05).astype(np.float16))
    ho, xo, lg = glue.norm_router(h, w, 1e-6, router, zp=(z, svh))
    h2, x2 = glue.norm_rotate(h, w, 1e-6, zp=(z, svh))
    assert bool(mx.array_equal(ho, h2).item()) and bool(mx.array_equal(xo, x2).item())
    assert bool(mx.array_equal(lg, glue.router_logits(x2, router)).item())
    for i in (0, 5):
        one = glue.norm_router(h[i:i + 1], w, 1e-6, router, zp=(z[:, i:i + 1], svh))[2]
        assert bool(mx.array_equal(one[0], lg[i]).item())
