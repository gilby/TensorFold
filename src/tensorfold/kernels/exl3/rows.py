"""Row-exact EXL3 projections for decode rows: one launch a projection group, every row its own fixed fma order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import mlx.core as mx

from tensorfold.kernels.exl3 import tiles

KS = 2                   # K slices a threadgroup (split launches and very wide heads take 1), added in slice order
WIDE = 65536             # outputs from which a launch takes one K slice
SPLIT_BLOCKS = 64        # a split launch takes enough K splits to give this many threadgroups
DT = {mx.bfloat16: "bfloat", mx.float16: "half", mx.float32: "float"}

# A threadgroup takes one 128-column block (one output Hadamard block) of one projection; its units (n-tile, K
# slice) walk their K ranges (_CORE) and the slices are added in slice order before H, so a row's bits depend on
# neither the rows beside it nor how many simdgroups the device gives the threadgroup.
_MAIN = r"""
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int nb = int(threadgroup_position_in_grid.x);
  const int split = int(threadgroup_position_in_grid.y);
  const int R = XH_shape[1];
  constexpr int KB = K / 128, NT = N / 16, KBS = KB / (KS * GS), BW = WPT * 8;
  threadgroup float red[KS][RB][128];
  //PLAIN_BEGIN
  if (nb >= N / 128) {
    const int c = (nb - N / 128) * SGS + sg;
    if (c < NP) {
PLAIN_COLUMN
    }
    return;
  }
  //PLAIN_END
  const int proj = PROJ_OF(nb);
  for (int r0 = 0; r0 < R; r0 += RB) {
    const int rn = min(RB, R - r0);
    for (int u = sg; u < 8 * KS; u += SGS) {
      const int nt = nb * 8 + (u % 8), ks = u / 8;
      const int kb0 = (split * KS + ks) * KBS;
      const device uint* tbase = (const device uint*)T + size_t(nt / 8) * NBSTRIDE + size_t(nt % 8) * TW
                                 + size_t(kb0) * 8 * KSTRIDE;
      #define XPTR(r) ((const device uint4*)(XH + (size_t(proj) * R + r0 + (r)) * K + kb0 * 128) + (lane % 4))
CORE_LOOP      #undef XPTR
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int r = sg; r < rn; r += SGS) {
      float v[4];
      for (int j = 0; j < 4; j++) {
        float s = red[0][r][4 * lane + j];
        for (int q = 1; q < KS; q++) s += red[q][r][4 * lane + j];
        v[j] = s;
      }
      exl3_fwht128(v, lane);
      for (int j = 0; j < 4; j++) {
        const int n = nb * 128 + 4 * int(lane) + j;
        STORE
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
"""

# One unit's walk of its K range, shared by every EXL3 kernel here: the next 128-block's words in flight while this
# one is decoded into registers once; then every row of the pass runs its two columns' fmas over it (k order in
# the tile, tiles in k order); the quad's lanes are added by a fixed tree into red[ks][row][column]. XPTR(r) is
# row r's inputs in lane order (device or threadgroup memory), from the unit's first block.
_CORE = r"""
      float acc[RB][2];
      for (int r = 0; r < RB; r++) { acc[r][0] = 0.0f; acc[r][1] = 0.0f; }
      uint cur[BW];
      FIRST_LOAD
      for (int b = 0; b < KBS; b++) {
        for (int i = 0; i < BW; i++) cur[i] = nxt[i];
        if (b + 1 < KBS) { LOAD_BLOCK(nxt, b + 1) }
        half wv[8][8];
        for (int t = 0; t < 8; t++) {
          uint w[3];
          LANE_WORDS
          uint st[8];
LANE_STATES
          for (int j = 0; j < 8; j++) wv[t][j] = exl3_value(st[j], CB);
        }
        for (int r = 0; r < RB; r++) {
          if (r < rn) {
            auto xp = XPTR(r) + b * 16;
            uint4 xq[4];
            for (int i = 0; i < 4; i++) xq[i] = xp[4 * i];
            float a0 = acc[r][0], a1 = acc[r][1];
            for (int t = 0; t < 8; t++) {
              const uint2 h = (t & 1) ? xq[t / 2].zw : xq[t / 2].xy;
              const half2 xa = as_type<half2>(h.x), xb = as_type<half2>(h.y);
              a0 = fma(float(xa.x), float(wv[t][0]), a0); a0 = fma(float(xa.y), float(wv[t][1]), a0);
              a0 = fma(float(xb.x), float(wv[t][2]), a0); a0 = fma(float(xb.y), float(wv[t][3]), a0);
              a1 = fma(float(xa.x), float(wv[t][4]), a1); a1 = fma(float(xa.y), float(wv[t][5]), a1);
              a1 = fma(float(xb.x), float(wv[t][6]), a1); a1 = fma(float(xb.y), float(wv[t][7]), a1);
            }
            acc[r][0] = a0; acc[r][1] = a1;
          }
        }
      }
      for (int r = 0; r < RB; r++)
        for (int h = 0; h < 2; h++) {
          float v = acc[r][h];
          v = v + simd_shuffle_xor(v, ushort(1));
          v = v + simd_shuffle_xor(v, ushort(2));
          if (lane % 4 == 0 && r < rn) red[ks][r][(u % 8) * 16 + int(lane) / 4 + 8 * h] = v;
        }
"""

# Plain fp16 column c of rows x [R, K] bf16 (WP [NP, K]): a simdgroup a column, its weights read once for every
# row; lanes take K in 8-element runs in a fixed order and a fixed tree adds them. PLAIN_OUT(r, c, value) stores.
_PLAIN = r"""
      const device uint4* wr = (const device uint4*)(WP + size_t(c) * K);
      for (int r0 = 0; r0 < R; r0 += 8) {
        const int rn = min(8, R - r0);
        float a[8];
        for (int r = 0; r < 8; r++) a[r] = 0.0f;
        for (int i = int(lane); i < K / 8; i += 32) {
          const uint4 wv = wr[i];
          float wf[8];
          for (int e = 0; e < 8; e++) wf[e] = float(as_type<half2>(wv[e / 2])[e % 2]);
          for (int r = 0; r < 8; r++) {
            if (r < rn) {
              const uint4 xv = ((const device uint4*)(XP + size_t(r0 + r) * K))[i];
              for (int e = 0; e < 8; e++) {
                const uint xw = xv[e / 2];
                a[r] = fma(as_type<float>((e % 2) ? (xw & 0xFFFF0000u) : (xw << 16)), wf[e], a[r]);
              }
            }
          }
        }
        for (int r = 0; r < 8; r++) {
          float v = a[r];
          for (ushort m = 16; m >= 1; m >>= 1) v = v + simd_shuffle_xor(v, m);
          if (lane == 0 && r < rn) PLAIN_OUT(r0 + r, c, v);
        }
      }
"""

_STORE_EPI = "Y[size_t(r0 + r) * NY + n] = YT(v[j] * HSCALE * float(SVH[n]) BIAS_ADD);"
_STORE_SPLIT = "Z[(size_t(split) * R + r0 + r) * N + n] = v[j] * HSCALE;"


def _loads(k2: int) -> tuple[str, str, int]:
    """(LOAD_BLOCK macro, LANE_WORDS statements, words a lane a tile): 4-bit lanes read one word, the one before by shuffle."""

    if k2 == 8:
        load = ("#define LOAD_BLOCK(dst, bb) for (int t_ = 0; t_ < 8; t_++) "
                "dst[t_] = tbase[(size_t(bb) * 8 + t_) * KSTRIDE + lane];\n")
        words = "          w[1] = cur[t]; w[0] = simd_shuffle(cur[t], ushort((lane + 31) & 31));\n"
        return load, words, 1
    lw, tw = tiles.lane_words(k2), tiles.tile_words(k2)
    load = (f"#define LOAD_BLOCK(dst, bb) for (int t_ = 0; t_ < 8; t_++) for (int q_ = 0; q_ < {lw}; q_++) "
            f"dst[t_ * {lw} + q_] = tbase[(size_t(bb) * 8 + t_) * KSTRIDE + ({tiles.first_word(k2)} + q_) % {tw}u];\n")
    return load, f"          for (int q = 0; q < {lw}; q++) w[q] = cur[t * {lw} + q];\n", lw


def proj_of(blocks: Sequence[int]) -> str:
    """Metal expression: the projection of column block ``nb`` in a stack of ``blocks`` blocks each."""

    terms, acc = [], 0
    for b in blocks[:-1]:
        acc += b
        terms.append(f"int(nb >= {acc})")
    return "(" + (" + ".join(terms) if terms else "0") + ")"


def source(k2: int, split: bool, plain: bool, bias: bool) -> tuple[str, str]:
    """(body, header) of the projection kernel for one width and output mode."""

    import re

    load, words, _ = _loads(k2)
    body = _MAIN.replace("CORE_LOOP", _CORE.replace("FIRST_LOAD", "uint nxt[BW];\n      LOAD_BLOCK(nxt, 0)")).replace("LANE_WORDS", words).replace("LANE_STATES", tiles.lane_states(k2))
    body = body.replace("STORE", _STORE_SPLIT if split else _STORE_EPI)
    body = body.replace("BIAS_ADD", "+ float(BIAS[n])" if bias else "")
    if not plain:
        body = re.sub(r"//PLAIN_BEGIN.*?//PLAIN_END", "", body, flags=re.S)
    body = body.replace("PLAIN_COLUMN", _PLAIN.replace("PLAIN_OUT(r0 + r, c, v)", "Y[size_t(r0 + r) * NY + N + c] = YT(v)"))
    return body, load


@dataclass
class Stack:
    """EXL3 projections reading the same rows, stacked along their outputs (one launch)."""

    words: mx.array                 # uint32 [K/16, N/16, TW] (stored) or [N/128, K/16, 8, TW] (strips)
    suh: mx.array                   # fp16 [P, K]: each projection's input scales
    svh: mx.array                   # fp16 [N]
    k: int
    n: int
    k2: int
    cb: int
    sizes: tuple[int, ...]          # each projection's outputs (multiples of 128)
    bias: mx.array | None = None    # fp16 [N]
    layout: str = "strips"

    @property
    def strides(self) -> tuple[int, int]:
        """(words between k tiles of one n tile, words between 128-column blocks)."""

        tw = tiles.tile_words(self.k2)
        if self.layout == "strips":
            return 8 * tw, (self.k // 16) * 8 * tw
        return (self.n // 16) * tw, 8 * tw

    @property
    def blocks(self) -> tuple[int, ...]:
        return tuple(s // tiles.HAD for s in self.sizes)

    def nbytes(self) -> int:
        return int(self.words.nbytes)


def splits(k: int, blocks: int, ks: int = 1) -> int:
    """K splits over the grid for a split launch: the shape's alone, so a row keeps one order at every width."""

    kb, gs = k // tiles.HAD, 1
    while blocks * gs < SPLIT_BLOCKS and kb % (ks * gs * 2) == 0:
        gs *= 2
    return gs


def _ks(k: int, split: bool = False, n: int = 0) -> int:
    """K slices a threadgroup: the launch kind's and shape's alone."""

    ks = 1 if split or n >= WIDE else KS
    while (k // tiles.HAD) % ks:
        ks //= 2
    return ks


_plans: dict[Any, tuple[Any, tuple[int, int], int, tuple[int, ...], Any]] = {}


def _plan(stack: Stack, R: int, split: bool, np_: int, out_dtype: Any, dep: bool) -> tuple:
    """(kernel, grid, threadgroup size, output shape, dtype) for a call shape, built once."""

    rb = 8 if R >= 8 else (4 if R >= 4 else (2 if R >= 2 else 1))
    key = (stack.k, stack.n, stack.k2, stack.cb, stack.blocks, stack.layout, stack.bias is not None, rb, split, np_,
           out_dtype, dep)
    hit = _plans.get(key)
    if hit is not None:
        return hit
    ks = _ks(stack.k, split, stack.n)
    gs = splits(stack.k, sum(stack.blocks), ks) if split else 1
    kstride, nbstride = stack.strides
    consts = {"K": stack.k, "N": stack.n, "K2": stack.k2, "CB": stack.cb, "KS": ks, "GS": gs, "RB": rb,
              "TW": tiles.tile_words(stack.k2), "WPT": _loads(stack.k2)[2], "KSTRIDE": kstride, "NBSTRIDE": nbstride,
              "NP": np_, "NY": stack.n + np_, "SGS": min(32, 8 * ks)}
    body, load = source(stack.k2, split, np_ > 0, stack.bias is not None)
    header = f"\n#define PROJ_OF(nb) {proj_of(stack.blocks)}\n#define YT {DT[out_dtype]}\n" + load
    inputs = ["XH", "T", "SVH"] + (["BIAS"] if stack.bias is not None else []) + (["XP", "WP"] if np_ else [])
    inputs += ["DEP"] if dep else []
    run = tiles.kernel("exl3_rows", body, inputs, ["Z" if split else "Y"], consts, header)
    nb = sum(stack.blocks) + -(-np_ // consts["SGS"])
    shape = (gs, R, stack.n) if split else (R, stack.n + np_)
    hit = _plans[key] = (run, (nb, gs), 32 * consts["SGS"], shape, mx.float32 if split else out_dtype)
    return hit


def _launch(stack: Stack, xh: mx.array, *, split: bool, plain: tuple[mx.array, mx.array] | None,
            out_dtype: Any, dep: mx.array | None) -> mx.array:
    R = int(xh.shape[1])
    np_ = int(plain[1].shape[0]) if plain is not None else 0
    run, (nb, gs), size, shape, dtype = _plan(stack, R, split, np_, out_dtype, dep is not None)
    arrays = [xh, stack.words, stack.svh] + ([stack.bias] if stack.bias is not None else []) + (list(plain) if plain else [])
    if dep is not None:
        arrays.append(dep)
    shape = (shape[0], R, shape[2]) if split else (R, shape[1])
    return run(inputs=arrays, grid=(nb * size, gs, 1), threadgroup=(size, 1, 1), output_shapes=[shape],
               output_dtypes=[dtype])[0]


def project(stack: Stack, xh: mx.array, *, plain: tuple[mx.array, mx.array] | None = None,
            out_dtype: Any = mx.bfloat16, dep: mx.array | None = None) -> mx.array:
    """Y [R, N (+ plain columns)] from XH [P, R, K] (``tiles.rotate``): H, svh and any bias applied; ``plain``
    (x [R, K] bf16, w [NP, K] fp16) appends x @ w.T."""

    return _launch(stack, xh, split=False, plain=plain, out_dtype=out_dtype, dep=dep)


def project_split(stack: Stack, xh: mx.array, *, dep: mx.array | None = None) -> mx.array:
    """Z [GS, R, N] fp32: the K splits' outputs after H, before svh; their sum in split order, times svh, is the layer's."""

    if stack.bias is not None:
        raise ValueError("a split projection leaves its bias to the reader")
    return _launch(stack, xh, split=True, plain=None, out_dtype=mx.float32, dep=dep)


def finish_split(z: mx.array, svh: mx.array) -> mx.array:
    """The layer's output (fp32) from a split launch, in the order the fused readers add it (tests and prompts)."""

    out = z[0]
    for g in range(1, int(z.shape[0])):
        out = out + z[g]
    return out * svh.astype(mx.float32)


def sources() -> list[str]:
    return [_MAIN, _CORE, _PLAIN, _STORE_EPI, _STORE_SPLIT, *(_loads(k2)[0] for k2 in (8, 12))]


__all__ = ["KS", "Stack", "finish_split", "proj_of", "project", "project_split", "sources", "splits"]
