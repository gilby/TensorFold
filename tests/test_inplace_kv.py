"""InplaceKVCache holds what mlx_lm's KVCache holds, and no pending reader sees a row a later write puts in place."""

import random

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
KVCache = pytest.importorskip("mlx_lm.models.cache").KVCache
if not mx.metal.is_available():
    pytest.skip("the in-place write is a Metal kernel", allow_module_level=True)

from tensorfold.engine.family_common import cache_contents  # noqa: E402
from tensorfold.engine.inplace_kv import InplaceKVCache  # noqa: E402
from tensorfold.engine.lane_engine import LaneEngine  # noqa: E402

# thread (word, row, head): OUT takes rows [0, P) of K after AFTER, late (a busy loop first)
_READ = r"""
  const uint w = thread_position_in_grid.x;
  const int r = int(thread_position_in_grid.y), h = int(thread_position_in_grid.z);
  const int P = META[0], CAP = K_shape[2], W = K_shape[3];
  if (r >= P) return;
  float x = float(w + r);
  for (int i = 0; i < REPEAT; i++) x = metal::sin(x);
  OUT[(size_t(h) * P + r) * W + w] = float(K[(size_t(h) * CAP + r) * W + w]) + (x > 100.0f ? 1.0f : 0.0f);
"""
_reader = mx.fast.metal_kernel(name="test_inplace_read", input_names=["K", "META", "AFTER"], output_names=["OUT"],
                               source="  constexpr int REPEAT = 4000;\n" + _READ)


def _read(keys, rows, after):
    """Rows [0, rows) of ``keys`` as float32, read slowly by a kernel that waits for ``after``."""

    from tensorfold.engine.inplace_kv import no_write
    from tensorfold.kernels.inputs import ints

    heads, dims = int(keys.shape[1]), int(keys.shape[3])
    return _reader(inputs=[keys, ints((rows,)), no_write() if after is None else after],
                   grid=(dims, max(rows, 1), heads), threadgroup=(min(dims, 256), 1, 1),
                   output_shapes=[(1, heads, max(rows, 1), dims)], output_dtypes=[mx.float32])[0][:, :, :rows]


def _rows(rng, count, dims=16):
    return mx.array(rng.standard_normal((1, 2, count, dims)).astype(np.float32)).astype(mx.bfloat16)


def _same(a, b):
    return all(bool(mx.array_equal(x, y).item()) for x, y in zip(a, b))


def test_matches_kv_cache_under_random_writes_trims_prompts_and_copies():
    rng = np.random.default_rng(5)
    pick = random.Random(5)
    for _ in range(6):
        ref, cache = KVCache(), InplaceKVCache()
        copies = []
        for _ in range(120):
            prompt = pick.random() < 0.08
            rows = pick.choice([20, 300]) if prompt else pick.randint(1, 16)
            k, v = _rows(rng, rows), _rows(rng, rows)
            ref.update_and_fetch(k, v)
            if prompt or cache.keys is None:
                cache.update_and_fetch(k, v)
            else:
                cache.write(k, v)
            if rows > 1 and pick.random() < 0.5:
                drop = pick.randint(1, rows - 1) if pick.random() < 0.9 else pick.randint(1, cache.offset - 1)
                assert ref.trim(drop) == cache.trim(drop)
            if pick.random() < 0.1:
                copies.append((LaneEngine.copy_single_cache([ref])[0], LaneEngine.copy_single_cache([cache])[0]))
            assert ref.offset == cache.offset
            cache._settle()
            assert _same(cache_contents(ref), cache_contents(cache))
        for r, c in copies:
            assert _same(cache_contents(r), cache_contents(c))


def test_pending_readers_never_see_a_later_write():
    rng = np.random.default_rng(7)
    pick = random.Random(7)
    ref, cache = KVCache(), InplaceKVCache()
    first = _rows(rng, 40)
    ref.update_and_fetch(first, first)
    cache.update_and_fetch(first, first)
    pending = []
    for step in range(200):
        rows = pick.randint(1, 16)
        k = _rows(rng, rows)
        start = cache.offset
        ref.update_and_fetch(k, k)
        keys, _, done = cache.write(k, k)
        # the reader of this step's earlier rows shares a command buffer with the next steps' writes
        pending.append((_read(keys, start, done), np.array(ref.keys[..., :start, :].astype(mx.float32))))
        if step % 8 == 7:
            mx.async_eval(*(read for read, _ in pending[-8:]))
        if rows > 1 and pick.random() < 0.6:
            drop = pick.randint(1, rows - 1) if pick.random() < 0.85 else min(cache.offset - 1, pick.randint(1, 40))
            ref.trim(drop)
            cache.trim(drop)
    for read, want in pending:
        assert np.array_equal(np.array(read), want)


def test_a_copy_keeps_its_rows_while_the_original_writes_and_rolls_back():
    rng = np.random.default_rng(9)
    cache = InplaceKVCache()
    k = _rows(rng, 30)
    cache.update_and_fetch(k, k)
    for _ in range(5):
        cache.write(_rows(rng, 4), _rows(rng, 4))
    copy = LaneEngine.copy_single_cache([cache])[0]
    held = [np.array(a.astype(mx.float32)) for a in cache_contents(copy)]
    owned = cache.keys
    for _ in range(4):
        cache.trim(6)
        cache.write(_rows(rng, 3), _rows(rng, 3))
    assert cache.keys is not owned                           # writing below the copy's rows took new buffers
    copy.write(_rows(rng, 2), _rows(rng, 2))                 # the copy writes into buffers of its own
    copy.trim(2)
    copy.materialize()
    assert all(np.array_equal(np.array(a.astype(mx.float32)), b) for a, b in zip(cache_contents(copy), held))


def test_decode_writes_stay_in_one_buffer_until_it_grows():
    rng = np.random.default_rng(11)
    cache = InplaceKVCache()
    k = _rows(rng, 10)
    cache.update_and_fetch(k, k)
    while cache.offset < 300:                                # past the prompt's buffer: whole growth steps from here
        cache.write(_rows(rng, 1), _rows(rng, 1))
    buffer = cache.keys
    assert int(buffer.shape[2]) == InplaceKVCache.grow
    while cache.offset < InplaceKVCache.grow:
        cache.write(_rows(rng, 1), _rows(rng, 1))
    assert cache.keys is buffer and cache.nbytes == KVCache.nbytes.fget(cache)
    cache.write(_rows(rng, 1), _rows(rng, 1))
    assert cache.keys is not buffer and int(cache.keys.shape[2]) == 2 * InplaceKVCache.grow
