"""Row glue for EXL3 decode: residual + RMSNorm reading the last block's partial outputs, the next inputs rotated."""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from tensorfold.kernels.exl3 import tiles

# One threadgroup a row: simdgroup s holds
# 128-block s of the D inputs, lane L its values 4 L .. 4 L + 3. The residual added is the last block's output:
# PEND 1 the experts' slots summed in slot order, PEND 2 a split projection's splits summed in split order then
# times svh. h = bf16(hidden + residual), rounded once; x = bf16(w * (h * rsqrt(mean(h^2) + eps))), the sum of
# squares each lane's four in order, a fixed tree, the blocks in order. Each of P scales then rotates x into XH.
_NORM = r"""
  const uint lane = thread_index_in_simdgroup;
  const int s = int(simdgroup_index_in_threadgroup);
  const int r = int(threadgroup_position_in_grid.y);
  const int R = H_shape[0];
  constexpr int NB = D / 128;
  threadgroup float red[NB];
  float h[4];
  for (int j = 0; j < 4; j++) {
    const int k = s * 128 + 4 * int(lane) + j;
#if EXL3_PEND == 1
    float p = YE[size_t(r) * SLOTS * D + k];
    for (int q = 1; q < SLOTS; q++) p += YE[(size_t(r) * SLOTS + q) * D + k];
    h[j] = float(bfloat(float(H[size_t(r) * D + k]) + p));
#elif EXL3_PEND == 2
    float p = ZP[size_t(r) * D + k];
    for (int g = 1; g < GS; g++) p += ZP[(size_t(g) * R + r) * D + k];
    h[j] = float(bfloat(float(H[size_t(r) * D + k]) + p * float(PSVH[k])));
#else
    h[j] = float(H[size_t(r) * D + k]);
#endif
  }
  float ss = 0.0f;
  for (int j = 0; j < 4; j++) ss = fma(h[j], h[j], ss);
  for (ushort d = 16; d >= 1; d >>= 1) ss = ss + simd_shuffle_xor(ss, d);
  if (lane == 0) red[s] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = red[0];
  for (int b = 1; b < NB; b++) total += red[b];
  const float inv = metal::rsqrt(total / float(D) + EPS[0]);
  float x[4];
  for (int j = 0; j < 4; j++) x[j] = float(bfloat(float(NW[s * 128 + 4 * int(lane) + j]) * (h[j] * inv)));
  if (int(threadgroup_position_in_grid.x) == 0) {
    for (int j = 0; j < 4; j++) {
      const int k = s * 128 + 4 * int(lane) + j;
#if EXL3_PEND
      HO[size_t(r) * D + k] = bfloat(h[j]);
#endif
      XO[size_t(r) * D + k] = bfloat(x[j]);
    }
#if EXL3_P
    for (int p = 0; p < P; p++) {
      float v[4];
      for (int j = 0; j < 4; j++) v[j] = x[j] * float(SUH[p * D + s * 128 + 4 * int(lane) + j]);
      exl3_fwht128(v, lane);
      for (int j = 0; j < 4; j++) XH[(size_t(p) * R + r) * D + s * 128 + exl3_slot(4 * int(lane) + j)] = half(v[j] * HSCALE);
    }
#endif
  }
"""

# TensorFold's row gdn_post (SiLU(z) * RMSNorm(y) * w, one simdgroup a (row, v head)) whose bf16 output, one
# head being one 128-block of out_proj's inputs, goes straight on to out_proj's scales and rotation.
# The post-attention norm and the router in one launch: threadgroup g takes router rows 16 g .. 16 g + 15 for every
# row of the window, RC rows at a time: each row's norm as _NORM's (threadgroup 0 stores h and x), then simdgroup
# s reads router row 16 g + s once for the RC rows (lanes over K in 8-element runs, a fixed tree).
_NORM_ROUTER = r"""
  const uint lane = thread_index_in_simdgroup;
  const int s = int(simdgroup_index_in_threadgroup);
  const int R = H_shape[0];
  constexpr int NB = D / 128;
  threadgroup float red[RC][NB];
  threadgroup bfloat xs[RC][D];
  const int e = int(threadgroup_position_in_grid.x) * NB + s;
  for (int r0 = 0; r0 < R; r0 += RC) {
    const int rn = min(RC, R - r0);
    float h[RC][4];
    for (int i = 0; i < rn; i++) {
      const int r = r0 + i;
      for (int j = 0; j < 4; j++) {
        const int k = s * 128 + 4 * int(lane) + j;
        float p = ZP[size_t(r) * D + k];
        for (int g = 1; g < GS; g++) p += ZP[(size_t(g) * R + r) * D + k];
        h[i][j] = float(bfloat(float(H[size_t(r) * D + k]) + p * float(PSVH[k])));
      }
      float ss = 0.0f;
      for (int j = 0; j < 4; j++) ss = fma(h[i][j], h[i][j], ss);
      for (ushort d = 16; d >= 1; d >>= 1) ss = ss + simd_shuffle_xor(ss, d);
      if (lane == 0) red[i][s] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < rn; i++) {
      const int r = r0 + i;
      float total = red[i][0];
      for (int b = 1; b < NB; b++) total += red[i][b];
      const float inv = metal::rsqrt(total / float(D) + EPS[0]);
      for (int j = 0; j < 4; j++) {
        const int k = s * 128 + 4 * int(lane) + j;
        const bfloat x = bfloat(float(NW[k]) * (h[i][j] * inv));
        xs[i][k] = x;
        if (int(threadgroup_position_in_grid.x) == 0) { HO[size_t(r) * D + k] = bfloat(h[i][j]); XO[size_t(r) * D + k] = x; }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e <= E) {
      const device uint4* wr = (const device uint4*)(RW + size_t(e) * D);
      float a[RC];
      for (int i = 0; i < RC; i++) a[i] = 0.0f;
      for (int c = int(lane); c < D / 8; c += 32) {
        const uint4 wv = wr[c];
        float wf[8];
        for (int q = 0; q < 8; q++) wf[q] = float(as_type<half2>(wv[q / 2])[q % 2]);
        for (int i = 0; i < RC; i++)
          if (i < rn)
            for (int q = 0; q < 8; q++) a[i] = fma(float(xs[i][8 * c + q]), wf[q], a[i]);
      }
      for (int i = 0; i < RC; i++) {
        float v = a[i];
        for (ushort d = 16; d >= 1; d >>= 1) v = v + simd_shuffle_xor(v, d);
        if (lane == 0 && i < rn) LG[size_t(r0 + i) * (E + 1) + e] = v;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
"""

_GDN_POST_ROT = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint hv = threadgroup_position_in_grid.y;
  const uint m = threadgroup_position_in_grid.z;
  const int W = int(threadgroups_per_grid.z);
  constexpr int PER = DV / 32;
  float yv[PER];
  float ss = 0.0f;
  for (int j = 0; j < PER; j++) {
    yv[j] = float(Y[(m * NV + hv) * DV + lane * PER + j]);
    ss += yv[j] * yv[j];
  }
  ss = simd_sum(ss);
  const float inv = metal::rsqrt(ss / float(DV) + eps[0]);
  float o[4];
  for (int j = 0; j < PER; j++) {
    const int d = int(lane) * PER + j;
    const float x = float(bfloat(float(NW[d]) * (yv[j] * inv)));
    const float zf = float(Z[m * ZS + ZO + hv * DV + d]);
    o[j] = float(bfloat(zf / (1.0f + metal::exp(-zf)) * x)) * float(SUH[hv * DV + d]);
  }
  exl3_fwht128(o, lane);
  for (int j = 0; j < 4; j++) XH[size_t(m) * NV * DV + hv * DV + exl3_slot(4 * int(lane) + j)] = half(o[j] * HSCALE);
"""

_consts: dict[Any, mx.array] = {}


def _eps(eps: float) -> mx.array:
    hit = _consts.get(eps)
    if hit is None:
        hit = _consts[eps] = mx.array([float(eps)], dtype=mx.float32)
    return hit


def _norm(hidden: mx.array, weight: mx.array, eps: float, *, ye: mx.array | None = None, zp: tuple | None = None,
          suh: mx.array | None = None) -> list[mx.array]:
    R, D = int(hidden.shape[0]), int(hidden.shape[1])
    pend = 1 if ye is not None else (2 if zp is not None else 0)
    P = int(suh.shape[0]) if suh is not None else 0
    body = _NORM
    consts = {"D": D, "PEND": pend, "P": P, "SLOTS": int(ye.shape[1]) if ye is not None else 1,
              "GS": int(zp[0].shape[0]) if zp is not None else 1}
    inputs, arrays = ["H", "NW", "EPS"], [hidden, weight, _eps(eps)]
    if ye is not None:
        inputs, arrays = inputs + ["YE"], arrays + [ye]
    if zp is not None:
        inputs, arrays = inputs + ["ZP", "PSVH"], arrays + list(zp)
    if P:
        inputs, arrays = inputs + ["SUH"], arrays + [suh]
    outputs = (["HO"] if pend else []) + ["XO"] + (["XH"] if P else [])
    shapes = ([(R, D)] if pend else []) + [(R, D)] + ([(P, R, D)] if P else [])
    dtypes = ([mx.bfloat16] if pend else []) + [mx.bfloat16] + ([mx.float16] if P else [])
    run = tiles.kernel("exl3_norm", body, inputs, outputs, consts, f"\n#define EXL3_PEND {pend}\n#define EXL3_P {P}\n")
    out = run(inputs=arrays, grid=((D // 128) * 32, R, 1), threadgroup=((D // 128) * 32, 1, 1),
              output_shapes=shapes, output_dtypes=dtypes)
    if not pend:
        out = [hidden, *out]
    return out


def norm_rotate(hidden: mx.array, weight: mx.array, eps: float, *, ye: mx.array | None = None,
                zp: tuple[mx.array, mx.array] | None = None, suh: mx.array | None = None) -> list[mx.array]:
    """[h, x, (XH)]: h = hidden + the experts' slots ``ye`` [R, S, D] or a split projection ``zp`` (z, svh), x = its
    RMSNorm times weight, XH = x rotated by each row of ``suh`` [P, D] (lane order)."""

    return _norm(hidden, weight, eps, ye=ye, zp=zp, suh=suh)


def norm_router(hidden: mx.array, weight: mx.array, eps: float, router: mx.array, *, ye: mx.array | None = None,
                zp: tuple[mx.array, mx.array] | None = None) -> list[mx.array]:
    """[h, x, logits [R, E + 1] fp32]: the post-attention norm and the router's logits (the shared gate's last)."""

    if zp is None:
        h, x = _norm(hidden, weight, eps, ye=ye)[:2]
        return [h, x, router_logits(x, router)]
    R, D = int(hidden.shape[0]), int(hidden.shape[1])
    n = int(router.shape[0])
    run = tiles.kernel("exl3_norm_router", _NORM_ROUTER, ["H", "NW", "EPS", "ZP", "PSVH", "RW"], ["HO", "XO", "LG"],
                       {"D": D, "E": n - 1, "GS": int(zp[0].shape[0]), "RC": 4})
    return list(run(inputs=[hidden, weight, _eps(eps), *zp, router], grid=(-(-n // (D // 128)) * D // 4, 1, 1),
                    threadgroup=(D // 4, 1, 1), output_shapes=[(R, D), (R, D), (R, n)],
                    output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32]))


def router_logits(x: mx.array, router: mx.array) -> mx.array:
    """x [R, D] bf16 @ router.T (fp16 [E + 1, D]) in fp32: a simdgroup a router row, read once for every row."""

    from tensorfold.kernels.exl3 import rows

    R, D = int(x.shape[0]), int(x.shape[1])
    n = int(router.shape[0])
    body = ("  const uint lane = thread_index_in_simdgroup;\n  const int R = XP_shape[0];\n"
            "  const int c = int(threadgroup_position_in_grid.x) * 32 + int(simdgroup_index_in_threadgroup);\n"
            "  if (c >= NP) return;\n" + rows._PLAIN.replace("PLAIN_OUT(r0 + r, c, v)", "LG[size_t(r0 + r) * NP + c] = v"))
    run = tiles.kernel("exl3_router", body, ["XP", "WP"], ["LG"], {"K": D, "NP": n})
    return run(inputs=[x, router], grid=(-(-n // 32) * 1024, 1, 1), threadgroup=(1024, 1, 1),
               output_shapes=[(R, n)], output_dtypes=[mx.float32])[0]


def gdn_post_rotate(rec: mx.array, y: mx.array, weight: mx.array, eps: float, suh: mx.array, *, zo: int) -> mx.array:
    """out_proj's XH [1, W, NV * DV] from the recurrence ``rec`` [1, W, NV, DV] and z read in place from ``y``."""

    _, W, nv, dv = (int(s) for s in rec.shape)
    if dv != 128:
        raise ValueError(f"gdn_post_rotate: value heads of 128 (one Hadamard block) expected, got {dv}")
    zs = int(y.shape[-1])
    run = tiles.kernel("exl3_gdn_post", _GDN_POST_ROT, ["Y", "Z", "NW", "eps", "SUH"], ["XH"],
                       {"NV": nv, "DV": dv, "ZS": zs, "ZO": zo})
    return run(inputs=[rec, y.reshape(W, zs), weight, _eps(eps), suh], grid=(32, nv, W), threadgroup=(32, 1, 1),
               output_shapes=[(1, W, nv * dv)], output_dtypes=[mx.float16])[0]


def sources() -> list[str]:
    return [_NORM, _NORM_ROUTER, _GDN_POST_ROT]


__all__ = ["gdn_post_rotate", "norm_rotate", "norm_router", "router_logits", "sources"]
