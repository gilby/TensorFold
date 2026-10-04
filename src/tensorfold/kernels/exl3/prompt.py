"""EXL3 prompt matmuls on Metal tensor units: tiles decoded into threadgroup memory, fixed tiles, H epilogue."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.exl3 import experts, rows, tiles

BM = 64                  # rows a threadgroup takes (32 for narrow layers); a row's bits are the same at either
GBM = 32                 # rows of one expert a grouped threadgroup takes
GU_TILES = (32, 4)       # gate/up launches: (K step, simdgroups)
DOWN_TILES = (32, 4)     # down launches
BK = 32                  # K a step: two k-tiles decoded per simdgroup step, double buffered
SG = 8                   # simdgroups a threadgroup: 16 of the block's 128 columns each
ES = 32                  # epilogue rows at a time (threadgroup memory)

_MPP = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

# One threadgroup: BM rows x one 128-column block. Simdgroup s owns columns 16 s .. 16 s + 15: it decodes its
# n-tile's BK x 16 values of the next step into its own threadgroup buffer (column-major, half2 stores) while the
# tensor units multiply the current one; K steps in order and fp32 accumulators, so a row's bits depend on the
# shape alone (never on the chunk or the rows beside it). The epilogue applies H to each row's 128 sums, then svh.
_GEMM = r"""
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  BLOCK_ROWS
  constexpr int CW = 128 / SG, CT = CW / 16;
  threadgroup half smem[SMEM];
  threadgroup half* Bs = smem + sg * (2 * BK * CW);
  constexpr auto desc = matmul2d_descriptor(BM, CW, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tA((device half*)A_BASE, dextents<int32_t, 2>(K, rows_));
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB0(Bs, dextents<int32_t, 2>(BK, CW));
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tB1(Bs + BK * CW, dextents<int32_t, 2>(BK, CW));
  auto a0 = tA.slice(0, 0);
  auto C = op.template get_destination_cooperative_tensor<decltype(a0), decltype(tB0), float>();
  for (int i = 0; i < C.get_capacity(); i++) if (C.is_valid_element(i)) C[i] = 0.0f;
  const device uint* tcol = (const device uint*)T + T_BASE + size_t(nb * 8 + CT * sg) * TW;
  const int r0 = 2 * (int(lane) % 4), c0 = int(lane) / 4;
  const uint wf = FIRST_WORD;
  #define DECODE_STEP(dst, kk) \
    for (int q = 0; q < (BK / 16) * CT; q++) { \
      const device uint* tp = tcol + (size_t((kk) / 16 + q / CT) * NT + (q % CT)) * TW; \
      uint w[3]; \
      for (int u = 0; u < LW; u++) w[u] = tp[(wf + u) % uint(TW)]; \
      uint st[8]; \
      LANE_STATES_INLINE \
      threadgroup half* d = dst + ((q % CT) * 16 + c0) * BK + (q / CT) * 16 + r0; \
      *(threadgroup half2*)(d) = half2(exl3_value(st[0], CB), exl3_value(st[1], CB)); \
      *(threadgroup half2*)(d + 8) = half2(exl3_value(st[2], CB), exl3_value(st[3], CB)); \
      *(threadgroup half2*)(d + 8 * BK) = half2(exl3_value(st[4], CB), exl3_value(st[5], CB)); \
      *(threadgroup half2*)(d + 8 * BK + 8) = half2(exl3_value(st[6], CB), exl3_value(st[7], CB)); \
    }
  threadgroup half* B0 = Bs;
  threadgroup half* B1 = Bs + BK * CW;
  DECODE_STEP(B0, 0)
  for (int k0 = 0; k0 < K; k0 += BK) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    auto ak = tA.slice(k0, 0);
    if ((k0 / BK) % 2 == 0) {
      if (k0 + BK < K) { DECODE_STEP(B1, k0 + BK) }
      op.run(ak, tB0, C);
    } else {
      if (k0 + BK < K) { DECODE_STEP(B0, k0 + BK) }
      op.run(ak, tB1, C);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float* Cs = (threadgroup float*)smem;
  for (int s0 = 0; s0 < BM; s0 += ES) {
    for (int i = 0; i < C.get_capacity(); i++) {
      if (!C.is_valid_element(i)) continue;
      auto idx = C.get_multidimensional_index(i);
      if (idx[1] >= s0 && idx[1] < s0 + ES) Cs[(idx[1] - s0) * 128 + CW * sg + idx[0]] = C[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int r = sg; r < ES && s0 + r < rows_; r += SG) {
      float v[4];
      for (int j = 0; j < 4; j++) v[j] = Cs[r * 128 + 4 * lane + j];
      exl3_fwht128(v, lane);
      for (int j = 0; j < 4; j++) {
        const int n = nb * 128 + 4 * int(lane) + j;
        const int row = s0 + r;
        STORE
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
"""

_DENSE_ROWS = """const int nb = int(threadgroup_position_in_grid.x);
  const int proj = PROJ_OF(nb);
  const int rb = int(threadgroup_position_in_grid.y) * BM;
  const int rows_ = min(BM, int(XH_shape[1]) - rb);
  const int npr_ = int(XH_shape[1]);"""
_DENSE_STORE = "Y[size_t(rb + row) * N + n] = OT(v[j] * HSCALE * float(SVH[n]) BIAS_ADD);"

# A grouped launch: threadgroup row y takes block y of the table (expert, first pair, pairs), or nothing.
_GROUP_ROWS = """const int nb = int(threadgroup_position_in_grid.x) % NBP, PROJ = int(threadgroup_position_in_grid.x) / NBP;
  const int blk = int(threadgroup_position_in_grid.y);
  if (blk >= TABLE[0]) return;
  const int e = TABLE[1 + 3 * blk], first = TABLE[2 + 3 * blk], rows_ = TABLE[3 + 3 * blk];
  const int npair_ = int(XH_shape[XH_ndim - 2]);"""
_GU_STORE = ("GU[(size_t(first + row) * 2 + PROJ) * N + n] = "
             "v[j] * HSCALE * float(SVH[(size_t(e) * 2 + PROJ) * N + n]);")
_DOWN_STORE = "YP[size_t(first + row) * N + n] = v[j] * HSCALE * float(SVH[size_t(e) * N + n]) * PW[first + row];"

# router logits [M, E + 1] fp32 for bf16 rows: one simdgroup a 64-row x 16-expert tile, the whole K in one op
_LOGITS = r"""
  const int ct = int(threadgroup_position_in_grid.x), rb = int(threadgroup_position_in_grid.y) * 64;
  const int M = X_shape[0];
  constexpr auto desc = matmul2d_descriptor(64, 16, D, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * D, dextents<int32_t, 2>(D, M - rb));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tB((device half*)RW + (int64_t)ct * 16 * D, dextents<int32_t, 2>(D, 16));
  auto C = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
  op.run(tA, tB, C);
  for (int i = 0; i < C.get_capacity(); i++) {
    if (!C.is_valid_element(i)) continue;
    auto idx = C.get_multidimensional_index(i);
    const int e = ct * 16 + idx[0], r = rb + idx[1];
    if (r < M && e <= E) LG[size_t(r) * (E + 1) + e] = C[i];
  }
"""

# each row's route (one simdgroup a row, the decode route's arithmetic): slot experts and weights
_ROUTE_ROWS = r"""
  const uint lane = thread_index_in_simdgroup;
  const int r = int(threadgroup_position_in_grid.x) * 8 + int(simdgroup_index_in_threadgroup);
  const int R = L_shape[0];
  constexpr int PER = E / 32, S = TOPK + 1;
  if (r >= R) return;
  const device float* l = L + size_t(r) * (E + 1);
  float v[PER];
  float m = -INFINITY;
  for (int j = 0; j < PER; j++) { v[j] = l[int(lane) * PER + j]; m = max(m, v[j]); }
  m = simd_max(m);
  float s = 0.0f;
  for (int j = 0; j < PER; j++) { v[j] = metal::precise::exp(v[j] - m); s += v[j]; }
  for (ushort d = 16; d >= 1; d >>= 1) s = s + simd_shuffle_xor(s, d);
  for (int j = 0; j < PER; j++) v[j] = v[j] / s;
  uint taken = 0u;
  float w[TOPK];
  int ids[TOPK];
  for (int k = 0; k < TOPK; k++) {
    float bv = -1.0f;
    int bi = E;
    for (int j = 0; j < PER; j++)
      if (!(taken & (1u << j)) && v[j] > bv) { bv = v[j]; bi = int(lane) * PER + j; }
    for (ushort d = 16; d >= 1; d >>= 1) {
      const float ov = simd_shuffle_xor(bv, d);
      const int oi = simd_shuffle_xor(bi, d);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (bi / PER == int(lane)) taken |= 1u << (bi % PER);
    w[k] = bv;
    ids[k] = bi;
  }
  if (NORM) {
    float t = w[0];
    for (int k = 1; k < TOPK; k++) t += w[k];
    for (int k = 0; k < TOPK; k++) w[k] = w[k] / t;
  }
  if (lane == 0) {
    for (int k = 0; k < TOPK; k++) { SE[r * S + k] = ids[k]; SW[r * S + k] = w[k]; }
    SE[r * S + TOPK] = E;
    SW[r * S + TOPK] = 1.0f / (1.0f + metal::precise::exp(-l[E]));
  }
"""

# the grouped launches' table: each expert's pairs in blocks of BM, in expert order (one thread)
_TABLE = r"""
  if (thread_position_in_grid.x != 0) return;
  int b = 0, at = 0;
  for (int e = 0; e <= E; e++) {
    const int c = COUNTS[e];
    for (int s = 0; s < c; s += BM) {
      TABLE[1 + 3 * b] = e; TABLE[2 + 3 * b] = at + s; TABLE[3 + 3 * b] = min(BM, c - s);
      b++;
    }
    at += c;
  }
  TABLE[0] = b;
"""

# pair pos (sorted by expert) of row r = PAIR[pos] / S: x[r] times its expert's gate or up scales, rotated
_GATHER = r"""
  const uint lane = thread_index_in_simdgroup;
  const int unit = int(threadgroup_position_in_grid.x) * 8 + int(simdgroup_index_in_threadgroup);
  constexpr int KB = K / 128;
  const int P = PAIR_shape[0];
  if (unit >= P * 2 * KB) return;
  const int pos = unit / (2 * KB), p = (unit / KB) % 2, kb = unit % KB;
  const int pair = PAIR[pos], r = pair / S, e = SE[pair];
  float v[4];
  for (int j = 0; j < 4; j++) {
    const int k = kb * 128 + 4 * int(lane) + j;
    v[j] = float(X[size_t(r) * K + k]) * float(SUH[(size_t(e) * 2 + p) * K + k]);
  }
  exl3_fwht128(v, lane);
  for (int j = 0; j < 4; j++) XH[(size_t(p) * P + pos) * K + kb * 128 + 4 * int(lane) + j] = half(v[j] * HSCALE);
"""

# SiLU(gate) * up of pair pos times its expert's down scales, rotated
_ACT = r"""
  const uint lane = thread_index_in_simdgroup;
  const int unit = int(threadgroup_position_in_grid.x) * 8 + int(simdgroup_index_in_threadgroup);
  constexpr int KB = I / 128;
  const int P = PAIR_shape[0];
  if (unit >= P * KB) return;
  const int pos = unit / KB, kb = unit % KB;
  const int e = SE[PAIR[pos]];
  float v[4];
  for (int j = 0; j < 4; j++) {
    const int k = kb * 128 + 4 * int(lane) + j;
    const float g = GU[size_t(pos) * 2 * I + k];
    v[j] = g / (1.0f + metal::exp(-g)) * GU[size_t(pos) * 2 * I + I + k] * float(SUH[size_t(e) * I + k]);
  }
  exl3_fwht128(v, lane);
  for (int j = 0; j < 4; j++) XH[size_t(pos) * I + kb * 128 + 4 * int(lane) + j] = half(v[j] * HSCALE);
"""

# a row's MoE output: its slots' outputs added in slot order (the shared expert's last)
_COMBINE = r"""
  const int i = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  if (i >= D) return;
  float y = YP[size_t(INV[r * S]) * D + i];
  for (int j = 1; j < S; j++) y += YP[size_t(INV[r * S + j]) * D + i];
  Y[size_t(r) * D + i] = y;
"""


def _gemm(kind: str, k: int, n: int, k2: int, cb: int, store_consts: dict[str, int], inputs: list[str],
          outputs: list[str], header: str = "", bm: int = BM, bk: int = BK, sg: int = SG) -> Any:
    rows_src = _DENSE_ROWS if kind == "dense" else _GROUP_ROWS
    store = {"dense": _DENSE_STORE, "gu": _GU_STORE, "down": _DOWN_STORE}[kind]
    body = _GEMM.replace("BLOCK_ROWS", rows_src).replace("STORE", store)
    body = body.replace("LANE_STATES_INLINE", tiles.lane_states(k2).replace("\n", " \\\n"))
    body = body.replace("FIRST_WORD", tiles.first_word(k2))
    smem = max(sg * 2 * bk * (128 // sg), min(ES, bm) * 128 * 2)
    consts = {"K": k, "N": n, "BM": bm, "BK": bk, "SG": sg, "ES": min(ES, bm), "TW": tiles.tile_words(k2),
              "LW": tiles.lane_words(k2), "CB": cb, "SMEM": smem, **store_consts}
    return tiles.kernel("exl3_prompt_" + kind, body, inputs, outputs, consts, _MPP + header)


def tensor_units() -> bool:
    """Whether the prompt kernels' tensor-unit path runs here (M5 and later); elsewhere prompts take the row kernels."""

    from tensorfold.kernels import device

    return device.tensor_units()


def _dense_bm(n: int) -> int:
    return BM if n // 128 >= 48 else 32          # narrow layers: more threadgroups


def matmul(stack: rows.Stack, x: mx.array, out_dtype: Any = mx.bfloat16) -> mx.array:
    """y [M, N] = x [M, K] @ W for every projection of the stack in one launch, any M; chunk-invariant."""

    if not tensor_units():
        return rows.project(stack, tiles.rotate(x, stack.suh), out_dtype=out_dtype)
    if stack.layout != "stored":
        raise ValueError("prompt matmuls read the stored tile order")
    M = int(x.shape[0])
    xh = tiles.rotate(x, stack.suh, lanes=False)
    bias = stack.bias is not None
    bm = _dense_bm(stack.n)
    header = (f"\n#define PROJ_OF(nb) {rows.proj_of(stack.blocks)}\n#define A_BASE (XH + ((int64_t)proj * npr_ + rb) * K)\n"
              f"#define T_BASE 0\n#define OT {rows.DT[out_dtype]}\n"
              + ("#define BIAS_ADD + float(BIAS[n])\n" if bias else "#define BIAS_ADD\n"))
    run = _gemm("dense", stack.k, stack.n, stack.k2, stack.cb, {"NT": stack.n // 16},
                ["XH", "T", "SVH"] + (["BIAS"] if bias else []), ["Y"], header, bm=bm)
    arrays = [xh, stack.words, stack.svh] + ([stack.bias] if bias else [])
    return run(inputs=arrays, grid=(stack.n // 128 * 32 * SG, -(-M // bm), 1), threadgroup=(32 * SG, 1, 1),
               output_shapes=[(M, stack.n)], output_dtypes=[out_dtype])[0]


def router_logits(x: mx.array, ex: experts.Experts) -> mx.array:
    """logits [M, E + 1] fp32 of bf16 rows x [M, D] on the tensor units (the shared gate's last); chunk-invariant."""

    M, D = int(x.shape[0]), int(x.shape[1])
    rw = _padded_router(ex)
    run = tiles.kernel("exl3_prompt_logits", _LOGITS, ["X", "RW"], ["LG"], {"D": D, "E": ex.count}, _MPP)
    tiles_ = int(rw.shape[0]) // 16
    return run(inputs=[x, rw], grid=(tiles_ * 32, -(-M // 64), 1), threadgroup=(32, 1, 1),
               output_shapes=[(M, ex.count + 1)], output_dtypes=[mx.float32])[0]


_padded: dict[int, mx.array] = {}


def _padded_router(ex: experts.Experts) -> mx.array:
    hit = _padded.get(id(ex.router))
    if hit is None:
        n = int(ex.router.shape[0])
        rows_ = -(-n // 16) * 16
        hit = mx.concatenate([ex.router, mx.zeros((rows_ - n, int(ex.router.shape[1])), dtype=mx.float16)])
        mx.eval(hit)
        _padded[id(ex.router)] = hit
    return hit


def moe(ex: experts.Experts, x: mx.array) -> mx.array:
    """The MoE block's output [M, D] fp32 for normed rows x [M, D] bf16: router, routed experts, shared expert."""

    M, D, I = int(x.shape[0]), ex.dims, ex.width
    S = ex.top_k + 1
    if not tensor_units():                     # the decode kernels, a window of rows at a time
        from tensorfold.kernels.exl3 import glue

        outs = []
        for r0 in range(0, M, experts.FUSED_ROWS):
            xc = x[r0:r0 + experts.FUSED_ROWS]
            gu, slots, weights, units = experts.route_gate_up(xc, glue.router_logits(xc, ex.router), ex)
            outs.append(experts.down(gu, slots, weights, units, ex).sum(axis=1))
        return outs[0] if len(outs) == 1 else mx.concatenate(outs)
    logits = router_logits(x, ex)
    run = tiles.kernel("exl3_prompt_route", _ROUTE_ROWS, ["L"], ["SE", "SW"],
                       {"E": ex.count, "TOPK": ex.top_k, "NORM": int(ex.norm_topk)})
    se, sw = run(inputs=[logits], grid=(-(-M // 8) * 256, 1, 1), threadgroup=(256, 1, 1),
                 output_shapes=[(M, S), (M, S)], output_dtypes=[mx.int32, mx.float32])
    flat = se.reshape(-1)
    pair = mx.argsort(flat).astype(mx.int32)                 # pairs grouped by expert (order within one: any)
    inv = mx.argsort(pair).astype(mx.int32)
    counts = mx.zeros((ex.count + 1,), dtype=mx.int32).at[flat].add(1)
    P = M * S
    most = -(-P // GBM) + ex.count + 1
    table = tiles.kernel("exl3_prompt_table", _TABLE, ["COUNTS"], ["TABLE"], {"E": ex.count, "BM": GBM})(
        inputs=[counts], grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(1 + 3 * most,)],
        output_dtypes=[mx.int32])[0]
    pw = sw.reshape(-1)[pair]
    xh = tiles.kernel("exl3_prompt_gather", _GATHER, ["X", "PAIR", "SE", "SUH"], ["XH"], {"K": D, "S": S})(
        inputs=[x, pair, flat, ex.suh_gu], grid=(-(-(P * 2 * (D // 128)) // 8) * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(2, P, D)], output_dtypes=[mx.float16])[0]
    run = _gemm("gu", D, I, ex.k2_gu, ex.cb, {"NT": I // 16, "NBP": I // 128}, ["XH", "TABLE", "T", "SVH"], ["GU"],
                "\n#define A_BASE (XH + (int64_t)(PROJ * npair_ + first) * K)\n"
                "#define T_BASE (size_t(e) * 2 + PROJ) * (K / 16) * NT * TW\n", bm=GBM, bk=GU_TILES[0], sg=GU_TILES[1])
    gu = run(inputs=[xh, table, ex.gate_up, ex.svh_gu], grid=(2 * (I // 128) * 32 * GU_TILES[1], most, 1),
             threadgroup=(32 * GU_TILES[1], 1, 1), output_shapes=[(P, 2, I)], output_dtypes=[mx.float32])[0]
    xd = tiles.kernel("exl3_prompt_act", _ACT, ["GU", "PAIR", "SE", "SUH"], ["XH"], {"I": I})(
        inputs=[gu, pair, flat, ex.suh_d], grid=(-(-(P * (I // 128)) // 8) * 256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(P, I)], output_dtypes=[mx.float16])[0]
    run = _gemm("down", I, D, ex.k2_d, ex.cb, {"NT": D // 16, "NBP": D // 128}, ["XH", "TABLE", "T", "SVH", "PW"], ["YP"],
                "\n#define A_BASE (XH + (int64_t)first * K)\n#define T_BASE size_t(e) * (K / 16) * NT * TW\n", bm=GBM,
                bk=DOWN_TILES[0], sg=DOWN_TILES[1])
    yp = run(inputs=[xd, table, ex.down, ex.svh_d, pw], grid=(D // 128 * 32 * DOWN_TILES[1], most, 1),
             threadgroup=(32 * DOWN_TILES[1], 1, 1),
             output_shapes=[(P, D)], output_dtypes=[mx.float32])[0]
    return tiles.kernel("exl3_prompt_combine", _COMBINE, ["YP", "INV"], ["Y"], {"D": D, "S": S})(
        inputs=[yp, inv], grid=(D, M, 1), threadgroup=(256, 1, 1), output_shapes=[(M, D)],
        output_dtypes=[mx.float32])[0]


def sources() -> list[str]:
    return [_GEMM, _DENSE_ROWS, _DENSE_STORE, _GROUP_ROWS, _GU_STORE, _DOWN_STORE, _LOGITS, _ROUTE_ROWS, _TABLE,
            _GATHER, _ACT, _COMBINE]


__all__ = ["BM", "GBM", "matmul", "moe", "router_logits", "sources", "tensor_units"]
