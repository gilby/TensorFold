"""EXL3 tiles on Metal (format after ExLlamaV3, MIT, Copyright (c) 2025 Turboderp): decoding, rotations and their sources."""

from __future__ import annotations

import hashlib
from typing import Any

import mlx.core as mx
import numpy as np

CODEBOOK_IDS = {"3inst": 0, "mcg": 1, "mul1": 2}
HAD = 128
HAD_SCALE = 0.08838834764831845          # 1 / sqrt(128)


def k2_of(bits: float) -> int:
    """Half-bits a value (2 * bits): 8 for 4-bit, 7 for 3.5-bit."""

    return int(round(2 * float(bits)))


def tile_words(k2: int) -> int:
    """32-bit words a 16x16 tile: 256 values of k2 / 2 bits."""

    return 4 * k2


def stream_end(k2: int, p: int) -> int:
    """E(p): one past the last stream bit of value p's 16-bit window (``format.stream_ends``)."""

    return ((p + 1) * k2 - ((p + 1) & 1)) // 2 if k2 & 1 else (p + 1) * (k2 // 2)


def lane_words(k2: int) -> int:
    """Words a lane reads for its 8 values: the most any lane's windows span (2 or 3)."""

    most = 0
    for lane in range(32):
        first = (stream_end(k2, 8 * lane) - 16 + 128 * k2) % 32
        most = max(most, first + stream_end(k2, 8 * lane + 7) - stream_end(k2, 8 * lane) + 16)
    return (most + 31) // 32


HEADER = r"""
// the high word of (hi:lo) << s, s in 0..31
inline uint exl3_fsl(uint lo, uint hi, uint s) { return s ? (hi << s) | (lo >> (32u - s)) : hi; }

// bits [d, d + 16) of the 96-bit stream hi:mid:lo (d a constant after unrolling)
inline uint exl3_window(uint hi, uint mid, uint lo, int d) {
  if (d <= 16) return (hi >> (16 - d)) & 0xffffu;
  if (d < 32) return exl3_fsl(mid, hi, uint(d)) >> 16;
  if (d <= 48) return (mid >> (48 - d)) & 0xffffu;
  return exl3_fsl(lo, mid, uint(d - 32)) >> 16;
}

// a 16-bit state's codebook value (3inst 0, mcg 1, mul1 2), rounded once to fp16 as the format defines
inline half exl3_value(uint s, int cb) {
  if (cb == 2) {
    const uint x = s * 0x83DCD12Du;
    const uint p = (x & 0x00FF00FFu) + ((x >> 8) & 0x00FF00FFu);
    const ushort h = ushort(0x6400u + (p & 0xFFFFu) + (p >> 16));
    return fma(as_type<half>(h), as_type<half>(ushort(0x1EEE)), as_type<half>(ushort(0xC931)));
  }
  uint x = cb == 1 ? s * 0xCBAC1FEDu : s * 89226354u + 64248484u;
  x = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
  const half2 h = as_type<half2>(x);
  return h.x + h.y;
}

// unscaled Walsh-Hadamard transform of 128 values, 4 a lane (index 4 lane + j), in a fixed butterfly order
inline void exl3_fwht128(thread float (&v)[4], uint lane) {
  const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
  v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
  for (ushort m = 1; m < 32; m <<= 1) {
    for (int j = 0; j < 4; j++) {
      const float o = simd_shuffle_xor(v[j], m);
      v[j] = (lane & m) ? o - v[j] : v[j] + o;
    }
  }
}

// a 128-block's slot in lane order of its index k: lane L of a decode reads its tiles' inputs as one run
inline int exl3_slot(int k) {
  const int t = k / 16, r = k % 16;
  return 32 * (t / 2) + 8 * ((r % 8) / 2) + 4 * (t % 2) + 2 * (r / 8) + (r % 2);
}
"""


def lane_states(k2: int, lane: str = "lane") -> str:
    """Metal statements: the 8 states of lane ``lane``'s values (8 lane + j) from its words ``w[]`` into ``st[]``."""

    ends = [stream_end(k2, j) - stream_end(k2, 0) for j in range(8)]
    lines = [
        f"    const uint off_ = uint(4 * int({lane}) * {k2} + {k2 // 2} - 16 + {128 * k2}) & 31u;",
        "    const uint c_ = " + ("w[2]" if lane_words(k2) > 2 else "0u") + ";",
        "    const uint hi_ = exl3_fsl(w[1], w[0], off_);",
        "    const uint mid_ = exl3_fsl(c_, w[1], off_);",
        "    const uint lo_ = c_ << off_;",
    ]
    lines += [f"    st[{j}] = exl3_window(hi_, mid_, lo_, {d});" for j, d in enumerate(ends)]
    return "\n".join(lines) + "\n"


def first_word(k2: int, lane: str = "lane") -> str:
    """Metal expression: the index in its tile of lane ``lane``'s first word."""

    return f"((uint(4 * int({lane}) * {k2} + {k2 // 2} - 16 + {128 * k2}) >> 5) % {tile_words(k2)}u)"


def lane_order(k: int) -> np.ndarray:
    """slot -> input index: XH's lane order over ``k`` inputs (the inverse of ``exl3_slot`` in each 128-block)."""

    out = np.empty(k, dtype=np.int64)
    for kk in range(k):
        b, r = divmod(kk, HAD)
        t, m = divmod(r, 16)
        out[b * HAD + 32 * (t // 2) + 8 * ((m % 8) // 2) + 4 * (t % 2) + 2 * (m // 8) + (m % 2)] = kk
    return out


_ROTATE = r"""
  // XH[p][r] = fp16(H(x[r] * suh[p]) / sqrt(128)) in lane order: a simdgroup a (row, projection, 128-block)
  const uint lane = thread_index_in_simdgroup;
  const int unit = int(threadgroup_position_in_grid.x) * 8 + int(simdgroup_index_in_threadgroup);
  const int R = X_shape[0];
  constexpr int KB = K / 128;
  if (unit >= R * P * KB) return;
  const int r = unit / (P * KB), p = (unit / KB) % P, kb = unit % KB;
  float v[4];
  for (int j = 0; j < 4; j++) {
    const int k = kb * 128 + 4 * int(lane) + j;
    v[j] = float(X[size_t(r) * K + k]) * float(SUH[p * K + k]);
  }
  exl3_fwht128(v, lane);
  for (int j = 0; j < 4; j++)
    XH[(size_t(p) * R + r) * K + kb * 128 + exl3_slot(4 * int(lane) + j)] = half(v[j] * HSCALE);
"""

# W_q [K, N] fp16 (the rotated-domain weight) from tiles: one simdgroup a tile; columns [N0, N0 + NW) of it.
_UNPACK = r"""
  const uint lane = thread_index_in_simdgroup;
  const int tile = int(threadgroup_position_in_grid.x) * 8 + int(simdgroup_index_in_threadgroup);
  constexpr int NTW = NW / 16;
  if (tile >= (K / 16) * NTW) return;
  const int kt = tile / NTW, nt = N0 / 16 + tile % NTW;
  const device uint* t = (const device uint*)T + size_t(nt / 8) * NBSTRIDE + size_t(nt % 8) * TW + size_t(kt) * KSTRIDE;
  const uint w0 = FIRST_WORD;
  uint w[3];
  for (int q = 0; q < LW; q++) w[q] = t[(w0 + q) % uint(TW)];
  uint st[8];
LANE_STATES
  for (int j = 0; j < 8; j++) {
    const int row = 2 * (int(lane) % 4) + (j & 1) + 8 * ((j >> 1) & 1), col = int(lane) / 4 + 8 * (j >> 2);
    WQ[size_t(kt * 16 + row) * NW + (nt - N0 / 16) * 16 + col] = exl3_value(st[j], CB);
  }
"""

_ROTATE_NATURAL = _ROTATE.replace("exl3_slot(4 * int(lane) + j)", "4 * int(lane) + j").replace(" in lane order", "")

# attention's output gated before its projection, bf16(o * bf16(sigmoid(gate))), then rotated in lane order; the
# gate is read in place from the q projection's rows (head h: D queries, then D gates)
_ROTATE_GATED = _ROTATE.replace(
    "v[j] = float(X[size_t(r) * K + k]) * float(SUH[p * K + k]);",
    "const float g = float(Q[size_t(r) * QS + (k / HD) * 2 * HD + HD + k % HD]);\n"
    "    v[j] = float(bfloat(float(X[size_t(r) * K + k]) * float(bfloat(1.0f / (1.0f + metal::exp(-g)))))) * float(SUH[p * K + k]);")

_kernels: dict[Any, Any] = {}


def kernel(base: str, body: str, inputs: list[str], outputs: list[str], consts: dict[str, int], header: str = "") -> Any:
    """A Metal kernel with integer constants baked into its source; MLX caches compiled kernels by the hashed name."""

    key = (base, body, tuple(inputs), tuple(outputs), tuple(sorted(consts.items())), header)
    hit = _kernels.get(key)
    if hit is None:
        source = "".join(f"  constexpr int {n} = {int(v)};\n" for n, v in sorted(consts.items()))
        source += f"  constexpr float HSCALE = {HAD_SCALE!r}f;\n" + body
        head = HEADER + header
        name = f"{base}_" + hashlib.sha256((head + source).encode()).hexdigest()[:16]
        hit = _kernels[key] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs,
                                                   source=source, header=head)
    return hit


def rotate(x: mx.array, suh: mx.array, lanes: bool = True) -> mx.array:
    """x [R, K] (any float type) and scales suh [P, K] fp16 -> XH [P, R, K] fp16, in lane order when ``lanes``."""

    R, K = int(x.shape[0]), int(x.shape[1])
    P = int(suh.shape[0])
    units = R * P * (K // HAD)
    run = kernel("exl3_rotate", _ROTATE if lanes else _ROTATE_NATURAL, ["X", "SUH"], ["XH"], {"K": K, "P": P})
    return run(inputs=[x, suh], grid=(-(-units // 8) * 256, 1, 1), threadgroup=(256, 1, 1),
               output_shapes=[(P, R, K)], output_dtypes=[mx.float16])[0]


def rotate_gated(x: mx.array, q: mx.array, head_dim: int, suh: mx.array) -> mx.array:
    """XH [1, R, K] (lane order) of bf16(x * sigmoid(gate)), the gate read from the rows q [R, QS] (per head D
    queries then D gates) in place."""

    R, K = int(x.shape[0]), int(x.shape[1])
    units = R * (K // HAD)
    run = kernel("exl3_rotate_gated", _ROTATE_GATED, ["X", "Q", "SUH"], ["XH"],
                 {"K": K, "P": 1, "QS": int(q.shape[-1]), "HD": int(head_dim)})
    return run(inputs=[x, q, suh], grid=(-(-units // 8) * 256, 1, 1), threadgroup=(256, 1, 1),
               output_shapes=[(1, R, K)], output_dtypes=[mx.float16])[0]


def unpack(words: mx.array, k: int, k2: int, cb: int, strides: tuple[int, int], n0: int, nw: int) -> mx.array:
    """Columns [n0, n0 + nw) of W_q [K, N] fp16 (128-aligned) from tile words with (k-tile, 128-block) word strides."""

    body = _UNPACK.replace("LANE_STATES", lane_states(k2)).replace("FIRST_WORD", first_word(k2))
    run = kernel("exl3_unpack", body, ["T"], ["WQ"], {"K": k, "K2": k2, "CB": cb, "TW": tile_words(k2),
                                                     "LW": lane_words(k2), "KSTRIDE": strides[0],
                                                     "NBSTRIDE": strides[1], "N0": n0, "NW": nw})
    tiles_ = (k // 16) * (nw // 16)
    return run(inputs=[words], grid=(-(-tiles_ // 8) * 256, 1, 1), threadgroup=(256, 1, 1),
               output_shapes=[(k, nw)], output_dtypes=[mx.float16])[0]


def hadamard() -> mx.array:
    """The 128 x 128 Sylvester Hadamard matrix over sqrt(128), fp32."""

    i = np.arange(HAD)
    parity = np.array([bin(v).count("1") & 1 for v in range(HAD)])[i[:, None] & i[None, :]]
    return mx.array(((1.0 - 2.0 * parity) * HAD_SCALE).astype(np.float32))


def sources() -> list[str]:
    """Every source this module contributes to a kernel fingerprint."""

    return [HEADER, _ROTATE, _ROTATE_NATURAL, _ROTATE_GATED, _UNPACK]


__all__ = ["CODEBOOK_IDS", "HAD", "HAD_SCALE", "HEADER", "first_word", "hadamard", "k2_of", "kernel", "lane_order",
           "lane_states", "lane_words", "rotate", "sources", "stream_end", "tile_words", "unpack"]
