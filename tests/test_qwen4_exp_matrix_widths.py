"""The matrix kernels Flash Next picks by shape before M5: a window matches its one-row steps, near MLX."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tensorfold.families.qwen4_exp import decode  # noqa: E402


def _linear(n, k, bits, group, seed):
    mx.random.seed(seed)
    holder = nn.Sequential(nn.Linear(k, n, bias=False))
    holder.set_dtype(mx.bfloat16)
    nn.quantize(holder, group_size=group, bits=bits)
    mx.eval(holder.parameters())
    return holder.layers[0]


@pytest.mark.parametrize("bits, group", [(5, 64), (6, 64), (8, 64), (5, 128), (8, 128), (4, 64), (4, 32)])
def test_matrix_rows_equal_one_row_steps(bits, group):
    linear = _linear(512, 1024, bits, group, seed=bits * 1000 + group)
    x = (mx.random.normal((16, 1024)) * 0.5).astype(mx.bfloat16)
    try:
        window = decode._matrix_project(x, linear)
        steps = mx.concatenate([decode._matrix_project(x[r:r + 1], linear) for r in range(16)])
        mx.eval(window, steps)
    except RuntimeError as exc:                      # no Metal matrix kernels here
        pytest.skip(str(exc).splitlines()[0][:80])
    assert bool(mx.array_equal(window, steps).item())
    ref = linear(x).astype(mx.float32)
    err = float(mx.max(mx.abs(window.astype(mx.float32) - ref)).item())
    assert err <= 0.02 * float(mx.max(mx.abs(ref)).item())


class _Shape:
    def __init__(self, n, k, bits, group):
        self.bits = bits
        self.group_size = group
        self.weight = type("W", (), {"shape": (n, k * bits // 32)})()


def test_default_matrix_is_chosen_by_format_and_shape(monkeypatch):
    monkeypatch.setattr(decode, "DENSE", "rows")
    for shape in [(16480, 2560, 4, 64), (10240, 2560, 4, 64), (16480, 2560, 4, 128), (16480, 2560, 8, 64),
                  (2560, 6144, 5, 128), (6144, 2560, 5, 128), (1280, 2560, 8, 128), (12288, 2560, 6, 64)]:
        assert decode._default_matrix(_Shape(*shape))
        assert decode.choose(shape[2], shape[3], shape[0], shape[1])
    assert not decode._default_matrix(_Shape(10240, 2560, 4, 32))         # 4-bit groups of 32: MLX's one-row bits
    assert not decode._default_matrix(_Shape(10244, 2560, 8, 64))         # outputs not in eights
    assert not hasattr(decode, "_checked")                                 # no switch: no simd_qmm-only path
    monkeypatch.setattr(decode, "DENSE", "lane")
    assert not decode._default_matrix(_Shape(16480, 2560, 4, 64))


def test_native_g128_four_bit_reads_the_same_values_as_two_groups_of_64():
    from tensorfold.kernels.qwen.dense.v1 import simd_qmm_bits

    linear = _linear(1024, 2048, 4, 128, seed=4128)
    x = (mx.random.normal((8, 2048)) * 0.5).astype(mx.bfloat16)
    try:
        assert simd_qmm_bits.check(linear.weight, linear.scales, linear.biases, 4, 128)
        native = decode._matrix_project(x, linear)
        halves = simd_qmm_bits.qmm(x, linear.weight, mx.repeat(linear.scales, 2, axis=1),
                                   mx.repeat(linear.biases, 2, axis=1), 4, 64)
        mx.eval(native, halves)
    except RuntimeError as exc:                      # no Metal matrix kernels here
        pytest.skip(str(exc).splitlines()[0][:80])
    assert decode._matrix[id(linear)][4] is True    # read natively, one scale load a group
    assert bool(mx.array_equal(native, halves).item())


def test_project_routes_the_stacked_shape_to_matrix(monkeypatch):
    monkeypatch.setattr(decode, "DENSE", "rows")
    linear = _linear(16480, 2560, 4, 64, seed=16480)
    seen = {}

    def fake(x, got):
        seen["linear"] = got
        return x

    monkeypatch.setattr(decode, "_matrix_project", fake)
    x = mx.zeros((2, 2560), dtype=mx.bfloat16)
    out = decode.project(x, linear)
    mx.eval(out)
    assert seen["linear"] is linear
    assert decode._default_matrix(linear)
