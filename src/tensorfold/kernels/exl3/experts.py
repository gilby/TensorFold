"""Routed EXL3 experts for decode rows: route once, gate/up and down one launch each over the window's experts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from tensorfold.kernels.exl3 import rows, tiles

SLOTS_EXTRA = 1          # the shared expert rides as the last slot of every row (expert index E, weight its gate)

# A row's route, one simdgroup: softmax over E routed logits (max, exp, a fixed-order sum), the top K by repeated
# argmax (ties to the lower expert), weights renormalised over the K in slot order when NORM, the shared expert's
# sigmoid gate in slot K; PUT(slot, expert, weight) stores them and USED(expert) marks the window's experts.
_ROW_ROUTE = r"""
    {
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
        for (int k = 0; k < TOPK; k++) { PUT(k, ids[k], w[k]); USED(ids[k]); }
        PUT(TOPK, E, 1.0f / (1.0f + metal::precise::exp(-l[E])));
      }
    }
"""

# The window's experts in ascending order, the shared one last: OUT(0) their count, OUT(1 + i) the experts.
_UNITS = r"""
    {
      uint c = 0u;
      for (int j = 0; j < PER; j++) c += used[int(lane) * PER + j];
      uint at = simd_prefix_exclusive_sum(c);
      const uint total = simd_sum(c);
      for (int j = 0; j < PER; j++)
        if (used[int(lane) * PER + j]) UOUT(1 + int(at++), int(lane) * PER + j);
      if (lane == 0) { UOUT(0, int(total) + 1); UOUT(1 + int(total), E); }
    }
"""

# One threadgroup: every row's route, then the units.
_ROUTE = r"""
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int R = L_shape[0];
  constexpr int PER = E / 32, S = TOPK + 1;
  threadgroup uint used[E];
  for (int i = int(thread_position_in_threadgroup.x); i < E; i += 256) used[i] = 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #define PUT(k, id, wt) { SE[r * S + (k)] = (id); SW[r * S + (k)] = (wt); }
  #define USED(id) used[(id)] = 1u
  for (int r = sg; r < R; r += 8)
ROW_ROUTE
  #undef PUT
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #define UOUT(i, v) UNITS[(i)] = (v)
  if (sg == 0)
UNITS_BLOCK
  #undef UOUT
  #undef USED
"""

# The rows of a unit (expert e) and the slot each holds it in, ascending rows, into rl / sl; the count into cnt.
_UNIT_ROWS = r"""
  threadgroup int rl[RMAX], sl[RMAX], cnt[1];
  const int unit = int(threadgroup_position_in_grid.x) / UNIT_TGS;
  if (unit >= UNITS[0]) return;
  const int e = UNITS[1 + unit];
  const int R = SE_shape[0];
  if (thread_position_in_threadgroup.x == 0) {
    int c = 0;
    for (int r = 0; r < R; r++)
      for (int j = 0; j < TOPK + 1; j++)
        if (SE[r * (TOPK + 1) + j] == e) { rl[c] = r; sl[c] = j; c++; break; }
    cnt[0] = c;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int count = cnt[0];
"""

# The same from this threadgroup's own route of every row (RMAX rows at most); threadgroup 0 publishes the route.
_ROUTED_ROWS = r"""
  threadgroup int rl[RMAX], sl[RMAX], cnt[1];
  constexpr int PER = E / 32, S = TOPK + 1;
  threadgroup int se_t[RMAX * S], ulist[E + 2];
  threadgroup float sw_t[RMAX * S];
  threadgroup uint used[E];
  const int R = L_shape[0];
  for (int i = int(thread_position_in_threadgroup.x); i < E; i += 32 * SGS) used[i] = 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #define PUT(k, id, wt) { se_t[r * S + (k)] = (id); sw_t[r * S + (k)] = (wt); }
  #define USED(id) used[(id)] = 1u
  for (int r = sg; r < R; r += SGS)
ROW_ROUTE
  #undef PUT
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #define UOUT(i, v) ulist[(i)] = (v)
  if (sg == 0)
UNITS_BLOCK
  #undef UOUT
  #undef USED
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (threadgroup_position_in_grid.x == 0) {
    for (int i = int(thread_position_in_threadgroup.x); i < R * S; i += 32 * SGS) { SE[i] = se_t[i]; SW[i] = sw_t[i]; }
    for (int i = int(thread_position_in_threadgroup.x); i <= ulist[0]; i += 32 * SGS) UNITS[i] = ulist[i];
  }
  const int unit = int(threadgroup_position_in_grid.x) / UNIT_TGS;
  if (unit >= ulist[0]) return;
  const int e = ulist[1 + unit];
  if (thread_position_in_threadgroup.x == 0) {
    int c = 0;
    for (int r = 0; r < R; r++)
      for (int j = 0; j < S; j++)
        if (se_t[r * S + j] == e) { rl[c] = r; sl[c] = j; c++; break; }
    cnt[0] = c;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int count = cnt[0];
"""

# gate (p 0) or up (p 1), 128-column block nb of the unit's expert, for each of its rows: x rotated by the expert's
# suh into threadgroup memory (lane order), the shared decode loop, then H into GU (down's reader applies svh).
_GATE_UP = r"""
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
UNIT_ROWS
  const int p = (int(threadgroup_position_in_grid.x) / NBI) % 2, nb = int(threadgroup_position_in_grid.x) % NBI;
  constexpr int KB = K / 128, NT = N / 16, KBS = KB / KS, BW = WPT * 8;
  threadgroup half xs[RB][K];
  threadgroup float red[KS][RB][128];
  const device half* suh = SUH + (size_t(e) * 2 + p) * K;
  const device uint* wbase = (const device uint*)T + (size_t(e) * 2 + p) * (K / 16) * NT * TW;
  uint nxt[BW];
  if (sg < 8 * KS) {                          // the first unit's first block in flight during the rotation
    const device uint* tbase = wbase + size_t(nb * 8 + sg % 8) * TW + size_t((sg / 8) * KBS) * 8 * KSTRIDE;
    LOAD_BLOCK(nxt, 0)
  }
  for (int r0 = 0; r0 < count; r0 += RB) {
    const int rn = min(RB, count - r0);
    for (int q = sg; q < rn * KB; q += SGS) {
      const int i = q / KB, kb = q % KB;
      float v[4];
      for (int j = 0; j < 4; j++) {
        const int k = kb * 128 + 4 * int(lane) + j;
        v[j] = float(X[size_t(rl[r0 + i]) * K + k]) * float(suh[k]);
      }
      exl3_fwht128(v, lane);
      for (int j = 0; j < 4; j++) xs[i][kb * 128 + exl3_slot(4 * int(lane) + j)] = half(v[j] * HSCALE);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int u = sg; u < 8 * KS; u += SGS) {
      const int nt = nb * 8 + (u % 8), ks = u / 8;
      const int kb0 = ks * KBS;
      const device uint* tbase = wbase + size_t(nt) * TW + size_t(kb0) * 8 * KSTRIDE;
      #define XPTR(r) ((const threadgroup uint4*)(&xs[(r)][kb0 * 128]) + (lane % 4))
CORE_LOOP
      #undef XPTR
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
        GU[((size_t(rl[r0 + r]) * (TOPK + 1) + sl[r0 + r]) * 2 + p) * N + n] = v[j] * HSCALE;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
"""

# down's 128-column block nb for the unit's rows: gate and up times their svh, SiLU(gate) * up times suh, rotated
# into threadgroup memory; the shared decode loop; then H, svh and the row's routing weight into YE[row][slot].
_DOWN = r"""
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
UNIT_ROWS
  const int nb = int(threadgroup_position_in_grid.x) % NBO;
  constexpr int KB = K / 128, NT = N / 16, KBS = KB / KS, BW = WPT * 8;
  threadgroup half xs[RB][K];
  threadgroup float red[KS][RB][128];
  const device half* suh = SUH + size_t(e) * K;
  const device half* svg = SVG + size_t(e) * 2 * K;
  const device uint* wbase = (const device uint*)T + size_t(e) * (K / 16) * NT * TW;
  uint nxt[BW];
  if (sg < 8 * KS) {                          // the first unit's first block in flight during the rotation
    const device uint* tbase = wbase + size_t(nb * 8 + sg % 8) * TW + size_t((sg / 8) * KBS) * 8 * KSTRIDE;
    LOAD_BLOCK(nxt, 0)
  }
  for (int r0 = 0; r0 < count; r0 += RB) {
    const int rn = min(RB, count - r0);
    for (int q = sg; q < rn * KB; q += SGS) {
      const int i = q / KB, kb = q % KB;
      const size_t at = size_t(rl[r0 + i]) * (TOPK + 1) + sl[r0 + i];
      float v[4];
      for (int j = 0; j < 4; j++) {
        const int k = kb * 128 + 4 * int(lane) + j;
        const float g = GU[at * 2 * K + k] * float(svg[k]), up = GU[(at * 2 + 1) * K + k] * float(svg[K + k]);
        v[j] = g / (1.0f + metal::exp(-g)) * up * float(suh[k]);
      }
      exl3_fwht128(v, lane);
      for (int j = 0; j < 4; j++) xs[i][kb * 128 + exl3_slot(4 * int(lane) + j)] = half(v[j] * HSCALE);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int u = sg; u < 8 * KS; u += SGS) {
      const int nt = nb * 8 + (u % 8), ks = u / 8;
      const int kb0 = ks * KBS;
      const device uint* tbase = wbase + size_t(nt) * TW + size_t(kb0) * 8 * KSTRIDE;
      #define XPTR(r) ((const threadgroup uint4*)(&xs[(r)][kb0 * 128]) + (lane % 4))
CORE_LOOP
      #undef XPTR
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
      const int row = rl[r0 + r], slot = sl[r0 + r];
      const float wt = SW[row * (TOPK + 1) + slot];
      for (int j = 0; j < 4; j++) {
        const int n = nb * 128 + 4 * int(lane) + j;
        YE[(size_t(row) * (TOPK + 1) + slot) * N + n] = v[j] * HSCALE * float(SVH[size_t(e) * N + n]) * wt;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
"""


@dataclass
class Experts:
    """One layer's routed experts and its shared expert (index E), EXL3 tiles in the checkpoint's stored order."""

    gate_up: mx.array        # uint32 [E + 1, 2, D/16, I/16, TW]
    suh_gu: mx.array         # fp16 [E + 1, 2, D]
    svh_gu: mx.array         # fp16 [E + 1, 2, I]
    down: mx.array           # uint32 [E + 1, I/16, D/16, TW]
    suh_d: mx.array          # fp16 [E + 1, I]
    svh_d: mx.array          # fp16 [E + 1, D]
    router: mx.array         # fp16 [E + 1, D]: the routed experts' gate rows, then the shared expert's gate
    dims: int                # D
    width: int               # I
    count: int               # E
    top_k: int
    k2_gu: int
    k2_d: int
    cb: int
    norm_topk: bool = True

    def nbytes(self) -> int:
        return int(self.gate_up.nbytes + self.down.nbytes)


KS_GU = KS_D = 1         # K slices an expert threadgroup (small threadgroups share a core: their phases overlap)


def _rb(R: int) -> int:
    return 1 if R == 1 else 4


FUSED_ROWS = 32          # windows up to this many rows route inside the gate/up launch (its threadgroup memory)


def route(logits: mx.array, ex: Experts, dep: mx.array | None = None) -> tuple[mx.array, mx.array, mx.array]:
    """(slot experts [R, K + 1] int32, slot weights [R, K + 1] fp32, units [E + 2] int32) from logits [R, E + 1]."""

    R = int(logits.shape[0])
    body = _ROUTE.replace("ROW_ROUTE", _ROW_ROUTE).replace("UNITS_BLOCK", _UNITS)
    run = tiles.kernel("exl3_route", body, ["L"] + (["DEP"] if dep is not None else []), ["SE", "SW", "UNITS"],
                       {"E": ex.count, "TOPK": ex.top_k, "NORM": int(ex.norm_topk)})
    S = ex.top_k + SLOTS_EXTRA
    return tuple(run(inputs=[logits] + ([dep] if dep is not None else []), grid=(256, 1, 1), threadgroup=(256, 1, 1),
                     output_shapes=[(R, S), (R, S), (ex.count + 2,)], output_dtypes=[mx.int32, mx.float32, mx.int32]))


def units_for(R: int, ex: Experts) -> int:
    """The most units a window of R rows can route to: its routed experts, then the shared one."""

    return min(ex.count, ex.top_k * R) + 1


_plans: dict[Any, Any] = {}


def _expert_kernel(kind: str, ex: Experts, R: int, routed: bool = False) -> tuple[Any, int, int]:
    rb = _rb(R)
    rmax = max(8, 1 << (R - 1).bit_length())
    key = (kind, ex.dims, ex.width, ex.top_k, ex.k2_gu, ex.k2_d, ex.cb, ex.count, rb, rmax, routed)
    hit = _plans.get(key)
    if hit is not None:
        return hit
    gu = kind == "gate_up"
    k, n, k2 = (ex.dims, ex.width, ex.k2_gu) if gu else (ex.width, ex.dims, ex.k2_d)
    load, words, wpt = rows._loads(k2)
    ks = KS_GU if gu else KS_D
    while ks > 1 and (k // 128) % ks:
        ks //= 2
    tw = tiles.tile_words(k2)
    unit_rows = _ROUTED_ROWS.replace("ROW_ROUTE", _ROW_ROUTE).replace("UNITS_BLOCK", _UNITS) if routed else _UNIT_ROWS
    core = rows._CORE.replace("FIRST_LOAD", "if (!(r0 == 0 && u == sg)) { LOAD_BLOCK(nxt, 0) }")
    body = (_GATE_UP if gu else _DOWN).replace("UNIT_ROWS", unit_rows).replace("CORE_LOOP", core)
    body = body.replace("LANE_WORDS", words).replace("LANE_STATES", tiles.lane_states(k2))
    unit_tgs = 2 * (n // 128) if gu else n // 128
    consts = {"K": k, "N": n, "K2": k2, "CB": ex.cb, "KS": ks, "RB": rb, "TW": tw, "WPT": wpt, "KSTRIDE": (n // 16) * tw,
              "TOPK": ex.top_k, "UNIT_TGS": unit_tgs, "RMAX": rmax, "SGS": min(32, 8 * ks), ("NBI" if gu else "NBO"): n // 128}
    if routed:
        consts.update(E=ex.count, NORM=int(ex.norm_topk))
        inputs, outputs = ["X", "L", "T", "SUH", "SVH"], ["GU", "SE", "SW", "UNITS"]
    elif gu:
        inputs, outputs = ["X", "SE", "UNITS", "T", "SUH", "SVH"], ["GU"]
    else:
        inputs, outputs = ["GU", "SE", "SW", "UNITS", "T", "SUH", "SVH", "SVG"], ["YE"]
    run = tiles.kernel("exl3_" + kind, body, inputs, outputs, consts, "\n" + load)
    hit = _plans[key] = (run, unit_tgs, 32 * consts["SGS"])
    return hit


def gate_up(x: mx.array, slots: mx.array, units: mx.array, ex: Experts) -> mx.array:
    """GU [R, K + 1, 2, I] fp32: each row's gate and up sums (after H, before svh) of its slots' experts."""

    R = int(x.shape[0])
    run, unit_tgs, size = _expert_kernel("gate_up", ex, R)
    grid = units_for(R, ex) * unit_tgs
    return run(inputs=[x, slots, units, ex.gate_up, ex.suh_gu, ex.svh_gu], grid=(grid * size, 1, 1),
               threadgroup=(size, 1, 1), output_shapes=[(R, ex.top_k + 1, 2, ex.width)],
               output_dtypes=[mx.float32])[0]


def route_gate_up(x: mx.array, logits: mx.array, ex: Experts) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """(GU, slots, weights, units) from the logits: windows of up to FUSED_ROWS route inside the gate/up launch."""

    R = int(x.shape[0])
    if R > FUSED_ROWS:
        slots, weights, units = route(logits, ex)
        return gate_up(x, slots, units, ex), slots, weights, units
    run, unit_tgs, size = _expert_kernel("gate_up", ex, R, routed=True)
    grid = units_for(R, ex) * unit_tgs
    S = ex.top_k + SLOTS_EXTRA
    gu, slots, weights, units = run(
        inputs=[x, logits, ex.gate_up, ex.suh_gu, ex.svh_gu], grid=(grid * size, 1, 1), threadgroup=(size, 1, 1),
        output_shapes=[(R, S, 2, ex.width), (R, S), (R, S), (ex.count + 2,)],
        output_dtypes=[mx.float32, mx.int32, mx.float32, mx.int32])
    return gu, slots, weights, units


def down(gu: mx.array, slots: mx.array, weights: mx.array, units: mx.array, ex: Experts) -> mx.array:
    """YE [R, K + 1, D] fp32: each slot's down output times its routing weight; the slots' sum in order is the MoE
    output."""

    R = int(gu.shape[0])
    run, unit_tgs, size = _expert_kernel("down", ex, R)
    grid = units_for(R, ex) * unit_tgs
    return run(inputs=[gu, slots, weights, units, ex.down, ex.suh_d, ex.svh_d, ex.svh_gu], grid=(grid * size, 1, 1),
               threadgroup=(size, 1, 1), output_shapes=[(R, ex.top_k + 1, ex.dims)],
               output_dtypes=[mx.float32])[0]


def sources() -> list[str]:
    return [_ROW_ROUTE, _UNITS, _ROUTE, _UNIT_ROWS, _ROUTED_ROWS, _GATE_UP, _DOWN]


__all__ = ["Experts", "FUSED_ROWS", "down", "gate_up", "route", "route_gate_up", "sources", "units_for"]
