"""EXL3 on Metal: tiles decode bit for bit, projections match the format's float64 layer, rows equal one-row launches."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
from tensorfold.cuda.exl3 import format as fmt  # noqa: E402
from tensorfold.kernels.exl3 import rows, tiles  # noqa: E402

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

COMBOS = [(b, cb) for b in fmt.BITS for cb in fmt.CODEBOOKS if float(b).is_integer() or cb == "mul1"]


def _words(tr: np.ndarray) -> mx.array:
    return mx.array(tr).view(mx.uint32).reshape(tr.shape[0], tr.shape[1], -1)


def _stack(rng, k: int, sizes: list[int], bits: float, cb: str):
    trs = [rng.integers(-32768, 32767, size=(k // 16, n // 16, int(16 * bits)), dtype=np.int16) for n in sizes]
    suh = (rng.standard_normal((len(sizes), k)) * 0.5).astype(np.float16)
    svh = (rng.standard_normal(sum(sizes)) * 0.5).astype(np.float16)
    st = rows.Stack(_words(np.concatenate(trs, axis=1)), mx.array(suh), mx.array(svh), k, sum(sizes), tiles.k2_of(bits),
                    tiles.CODEBOOK_IDS[cb], tuple(sizes), layout="stored")
    return st, trs, suh, svh


def _reference(x: np.ndarray, trs, suh, svh, sizes, bits, cb) -> np.ndarray:
    outs, at = [], 0
    for p, n in enumerate(sizes):
        outs.append(fmt.forward(x, trs[p], suh[p], svh[at:at + n], bits, cb))
        at += n
    return np.concatenate(outs, axis=1)


@pytest.mark.parametrize("bits,cb", COMBOS)
def test_tiles_decode_bit_for_bit(bits, cb):
    rng = np.random.default_rng(int(2 * bits) * 3 + fmt.CODEBOOKS.index(cb))
    tr = rng.integers(-32768, 32767, size=(16, 24, int(16 * bits)), dtype=np.int16)
    tw = tiles.tile_words(tiles.k2_of(bits))
    wq = tiles.unpack(_words(tr), 256, tiles.k2_of(bits), tiles.CODEBOOK_IDS[cb], (24 * tw, 8 * tw), 0, 384)
    assert (np.array(wq).view(np.uint16) == fmt.unpack(tr, bits, cb).view(np.uint16)).all()


def test_rotation_is_the_layers_and_in_lane_order():
    rng = np.random.default_rng(1)
    x = mx.array(rng.standard_normal((3, 512)).astype(np.float32)).astype(mx.bfloat16)
    suh = (rng.standard_normal((2, 512)) * 0.5).astype(np.float16)
    xh = np.array(tiles.rotate(x, mx.array(suh)).astype(mx.float32))
    natural = np.empty_like(xh)
    natural[..., tiles.lane_order(512)] = xh
    xf = np.array(x.astype(mx.float32)).astype(np.float64)
    for p in range(2):
        ref = fmt.rotate(xf * suh[p].astype(np.float64), -1)
        assert np.abs(natural[p] - ref).max() <= 2e-3 * np.abs(ref).max()


@pytest.mark.parametrize("bits,cb", [(4, "mcg"), (6, "mcg"), (2, "3inst"), (3.5, "mul1"), (5, "mul1"), (8, "3inst")])
def test_projection_matches_the_format(bits, cb):
    rng = np.random.default_rng(7)
    sizes = [256, 384]
    st, trs, suh, svh = _stack(rng, 512, sizes, bits, cb)
    x = mx.array(rng.standard_normal((5, 512)).astype(np.float32)).astype(mx.bfloat16)
    y = np.array(rows.project(st, tiles.rotate(x, st.suh), out_dtype=mx.float32))
    ref = _reference(np.array(x.astype(mx.float32)), trs, suh, svh, sizes, bits, cb)
    assert np.abs(y - ref).max() <= 1e-3 * np.abs(ref).max()
    z = np.array(rows.finish_split(rows.project_split(st, tiles.rotate(x, st.suh)), st.svh))
    assert np.abs(z - ref).max() <= 1e-3 * np.abs(ref).max()


@pytest.mark.parametrize("split", [False, True])
def test_every_row_equals_its_one_row_launch(split):
    rng = np.random.default_rng(11)
    st, *_ = _stack(rng, 1024, [256, 128], 4, "mcg")
    x = mx.array(rng.standard_normal((64, 1024)).astype(np.float32)).astype(mx.bfloat16)
    run = rows.project_split if split else rows.project
    alone = [run(st, tiles.rotate(x[i:i + 1], st.suh)) for i in range(64)]
    for width in (2, 3, 5, 8, 9, 16, 33, 64):
        for start in (0, 64 - width):
            got = run(st, tiles.rotate(x[start:start + width], st.suh))
            for i in range(width):
                row = got[:, i] if split else got[i]
                one = alone[start + i][:, 0] if split else alone[start + i][0]
                assert bool(mx.array_equal(row, one).item()), (width, start, i)


def test_plain_columns_follow_the_exl3_ones():
    rng = np.random.default_rng(3)
    st, trs, suh, svh = _stack(rng, 2048, [512], 4, "mcg")
    x = mx.array(rng.standard_normal((3, 2048)).astype(np.float32)).astype(mx.bfloat16)
    wp = mx.array((rng.standard_normal((64, 2048)) * 0.1).astype(np.float16))
    y = np.array(rows.project(st, tiles.rotate(x, st.suh), plain=(x, wp), out_dtype=mx.float32))
    ref = np.array(x.astype(mx.float32)) @ np.array(wp.astype(mx.float32)).T
    assert y.shape == (3, 576) and np.abs(y[:, 512:] - ref).max() <= 1e-5 * np.abs(ref).max()
    one = np.array(rows.project(st, tiles.rotate(x[1:2], st.suh), plain=(x[1:2], wp), out_dtype=mx.float32))
    assert (one[0] == y[1]).all()
