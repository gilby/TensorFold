// Volta (sm_70) tree attention chunks on tensor cores (mma.m8n8k4, fp16 in, fp32 sums): attention.py's _shared and
// _tail as two kernels around one per-warp routine.
//
// A warp owns 8 query rows ((window row, head) pairs) and walks a 512-key chunk in 32-key sub-tiles from the chunk
// start: scores Q.K^T, an online softmax per row, then P.V. Every output element of an mma depends only on its own
// query row, so a row's chunk partial is the same whichever rows share its warp. ``shared_kernel`` runs the chunks
// wholly below a stream's committed keys with all its window rows in one block (each K/V sub-tile read once for all
// of them); ``tail_kernel`` runs the rest one window row a block, keys below the committed count from the cache and
// keys on the row's own path from the window's new keys. A committed key and the same key still on a path give the
// same bits, so drafted rows equal serial ones. Partials (unnormalized o, running max m, sum l) keep the layout
// attention.py's _merge reads.
//
// Packed int8 caches (kvpack rows: a head-row's D codes, then an fp16 scale a 32-value group) are read the same way,
// each loaded value dequantized as attention.py's _packed does, ``(code + 0.5) * s / 128`` in fp32 rounded to bf16,
// then to fp16 like a bf16 cache's value; the window's own rows arrive packed too, so a key on a path and the same key
// committed still give the same bits.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

constexpr int CH = 512, KT = 32;                 // keys a chunk, keys a sub-tile
constexpr int MAX_NODES = 128;                   // a window's rows (attention.py's MAX_NODES): a path's longest tail
constexpr int TAIL_CHUNKS = 1 + (MAX_NODES + CH - 1) / CH;   // chunks a row's tail spans past its committed keys
constexpr int QPAD = 8, KPAD = 4, VPAD = 4, PPAD = 8;        // smem row padding (bank spread), in halves
constexpr int MAXW = 12;                         // warps (8 query rows each) of a shared block
constexpr int TW = 8;                            // warps of a tail block: all load, warp 0 computes
constexpr int LOADERS = 256;                     // threads that load a sub-tile (every block has at least this many)

__device__ __forceinline__ void mma884(float (&d)[8], unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma884rr(float (&d)[8], unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned h2(unsigned bf2) {         // two bf16 -> two fp16 (round to nearest)
    const __half2 h = __floats2half2_rn(__uint_as_float(bf2 << 16), __uint_as_float(bf2 & 0xFFFF0000u));
    return *reinterpret_cast<const unsigned*>(&h);
}

__device__ __forceinline__ uint4 h8(uint4 u) { return make_uint4(h2(u.x), h2(u.y), h2(u.z), h2(u.w)); }

__device__ __forceinline__ unsigned short dq(unsigned word, int byte, float s) {   // one packed code -> fp16 bits
    const float x = ((float)(signed char)((word >> (8 * byte)) & 0xFFu) + 0.5f) * s * 0.0078125f;
    return __half_as_ushort(__float2half_rn(__bfloat162float(__float2bfloat16_rn(x))));
}

__device__ __forceinline__ unsigned dq2(unsigned word, int pair, float s) {
    return (unsigned)dq(word, 2 * pair, s) | ((unsigned)dq(word, 2 * pair + 1, s) << 16);
}

// Packed pieces carry 8 codes (x, y) and their group's fp16 scale bits (z): fp16 values in the model's rotated basis.
__device__ __forceinline__ uint4 q8(uint4 u) {
    const float s = __half2float(__ushort_as_half((unsigned short)(u.z & 0xFFFFu)));
    return make_uint4(dq2(u.x, 0, s), dq2(u.x, 1, s), dq2(u.y, 0, s), dq2(u.y, 1, s));
}

// v4 packed pieces: (code + 0.5) * (s / 128) rounded once to fp16 (the code from a biased byte by the 1024-exponent
// trick, exact; then one fma: c * s' + s'/2 is the exact product rounded once)
__device__ __forceinline__ unsigned dq4pair(unsigned word, unsigned sel, __half2 sc, __half2 half_sc) {
    const unsigned h = __byte_perm(word ^ 0x80808080u, 0x64646464u, sel);   // 1024 + code + 128 a half
    const __half2 c = __hsub2(*reinterpret_cast<const __half2*>(&h), __float2half2_rn(1152.f));
    const __half2 v = __hfma2(c, sc, half_sc);
    return *reinterpret_cast<const unsigned*>(&v);
}

__device__ __forceinline__ uint4 q8v4(uint4 u) {
    const __half s = __hmul(__ushort_as_half((unsigned short)(u.z & 0xFFFFu)), __float2half(0.0078125f));
    const __half2 sc = __half2half2(s), half_sc = __half2half2(__hmul(s, __float2half(0.5f)));
    return make_uint4(dq4pair(u.x, 0x4140, sc, half_sc), dq4pair(u.x, 0x4342, sc, half_sc),
                      dq4pair(u.y, 0x4140, sc, half_sc), dq4pair(u.y, 0x4342, sc, half_sc));
}

// A piece of 8 values at column d of a head-row: bf16 values (QB false), or codes and their group's scale.
template <int D, bool QB>
__device__ __forceinline__ uint4 piece(const unsigned char* row, int d) {
    if constexpr (QB) {
        const uint2 c = *reinterpret_cast<const uint2*>(row + d);
        const unsigned short sc = *reinterpret_cast<const unsigned short*>(row + D + (d >> 5) * 2);
        return make_uint4(c.x, c.y, sc, 0);
    } else {
        return *reinterpret_cast<const uint4*>(row + 2 * d);
    }
}

template <bool QB, bool RV = false>
__device__ __forceinline__ uint4 to_half(uint4 u) {
    if constexpr (QB) { if constexpr (RV) return q8v4(u); else return q8(u); }
    else return h8(u);
}

template <int D, bool QB>
struct Row { static constexpr int BYTES = QB ? D + D / 16 : 2 * D; };   // bytes a cached (or window) head-row

// RV (v4): values row-major in smem like keys (VS = D + 32: a key's row 16 banks on from the last), else transposed
template <int D, bool RV = false>
struct Smem {
    static constexpr int QS = D + QPAD, KS = D + KPAD, VS = RV ? D + 32 : KT + VPAD, PS = KT + PPAD;
    static size_t bytes(int warps) {
        return 2 * ((size_t)warps * 8 * QS + KT * KS + (RV ? KT : D) * VS + (size_t)warps * 8 * PS) + KT * 4;
    }
};

// One 32-key sub-tile in two halves, so the next one is in flight while this one computes: ``fetch`` into registers
// (NP 16-byte pieces of keys and of values a thread), ``put`` into smem (keys row-major fp16, values transposed,
// validity per key). ``src(i, kp, vp)`` gives key i's cache or path rows, or false for a key the row never sees.
template <int D, bool QB, bool RV = false>
struct KVTile {
    static constexpr int PIECES = KT * D / 8, NP = (PIECES + LOADERS - 1) / LOADERS;   // 8 bf16 values a piece
    uint4 k[NP], v[NP];
    bool ok[NP];
    template <typename Src>
    __device__ __forceinline__ void fetch(int k0, Src src) {
#pragma unroll
        for (int j = 0; j < NP; ++j) {
            const int e = threadIdx.x + j * LOADERS;
            k[j] = v[j] = make_uint4(0, 0, 0, 0);
            ok[j] = false;
            if (threadIdx.x >= LOADERS || e >= PIECES) continue;
            const int t = e / (D / 8), d = (e % (D / 8)) * 8;              // keys: a row's pieces side by side
            const int tv = e % KT, dv = (e / KT) * 8;                         // values: a column's keys side by side
            const unsigned char* kp;
            const unsigned char* vp;
            const unsigned char* kq;
            const unsigned char* vq;
            ok[j] = src(k0 + t, kp, vp);
            if (ok[j]) k[j] = piece<D, QB>(kp, d);
            if constexpr (RV) {                                               // values coalesced as keys
                if (ok[j]) v[j] = piece<D, QB>(vp, d);
            } else {
                if (src(k0 + tv, kq, vq)) v[j] = piece<D, QB>(vq, dv);
            }
        }
    }
    __device__ __forceinline__ void put(__half* Ks, __half* Vt, int* okk) const {
        using S = Smem<D, RV>;
#pragma unroll
        for (int j = 0; j < NP; ++j) {
            const int e = threadIdx.x + j * LOADERS;
            if (threadIdx.x >= LOADERS || e >= PIECES) continue;
            const int t = e / (D / 8), d = (e % (D / 8)) * 8;
            const int tv = e % KT, dv = (e / KT) * 8;
            const uint4 kh = ok[j] ? to_half<QB, RV>(k[j]) : make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint2*>(Ks + t * S::KS + d) = make_uint2(kh.x, kh.y);       // KS keeps 8-byte rows
            *reinterpret_cast<uint2*>(Ks + t * S::KS + d + 4) = make_uint2(kh.z, kh.w);
            if constexpr (RV) {
                *reinterpret_cast<uint4*>(Vt + t * S::VS + d) = ok[j] ? to_half<QB, RV>(v[j]) : make_uint4(0, 0, 0, 0);
                if (d == 0) okk[t] = ok[j];
                continue;
            }
            const uint4 vh = to_half<QB>(v[j]);                                     // an unseen key: zeros either way
            const unsigned vw[4] = {vh.x, vh.y, vh.z, vh.w};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                Vt[(dv + 2 * q) * S::VS + tv] = __ushort_as_half((unsigned short)(vw[q] & 0xFFFF));
                Vt[(dv + 2 * q + 1) * S::VS + tv] = __ushort_as_half((unsigned short)(vw[q] >> 16));
            }
            if (d == 0) okk[t] = ok[j];
        }
    }
};

// The per-warp routine: 8 query rows (Q in smem at q, row stride QS) against the sub-tile in smem.
template <int D, bool RV = false>
struct Warp {
    static constexpr int NT = D / 32;          // 8-column output tiles a quadpair
    float m[2], l[2], o[NT][8];
    __device__ __forceinline__ void init() {
        m[0] = m[1] = -INFINITY;
        l[0] = l[1] = 0.f;
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int i = 0; i < 8; ++i) o[t][i] = 0.f;
    }
    // ``lim`` (prompts): the last key index each of this thread's two rows may see, ``k0`` the sub-tile's first key;
    // a row past every key of a sub-tile takes it as a no-op (its max, sum and output keep their bits)
    __device__ __forceinline__ void step(const __half* q, const __half* Ks, const __half* Vt, const int* ok,
                                         __half* P, float scale, int k0 = 0, const int* lim = nullptr) {
        using S = Smem<D, RV>;
        const int lane = threadIdx.x & 31, qp = (lane >> 2) & 3, idx = (lane & 3) + 4 * (lane >= 16);
        float c[4][8];                                          // four chains over k-steps mod 4, summed (0+1)+(2+3)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) c[j][i] = 0.f;
        const __half* qrow = q + idx * S::QS;
        const __half* krow = Ks + (8 * qp + idx) * S::KS;
#pragma unroll 4
        for (int kk = 0; kk < D / 4; kk += 4) {
            const uint4 a01 = *reinterpret_cast<const uint4*>(qrow + 4 * kk);
            const uint4 a23 = *reinterpret_cast<const uint4*>(qrow + 4 * kk + 8);
            const uint2 b0 = *reinterpret_cast<const uint2*>(krow + 4 * kk);
            const uint2 b1 = *reinterpret_cast<const uint2*>(krow + 4 * kk + 4);
            const uint2 b2 = *reinterpret_cast<const uint2*>(krow + 4 * kk + 8);
            const uint2 b3 = *reinterpret_cast<const uint2*>(krow + 4 * kk + 12);
            mma884(c[0], a01.x, a01.y, b0.x, b0.y);
            mma884(c[1], a01.z, a01.w, b1.x, b1.y);
            mma884(c[2], a23.x, a23.y, b2.x, b2.y);
            mma884(c[3], a23.z, a23.w, b3.x, b3.y);
        }
        float s[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) s[i] = (c[0][i] + c[1][i]) + (c[2][i] + c[3][i]);
        // s[i]: row (lane&1) + 2((i>>1)&1) + 4(lane>=16), key 8qp + (i&1) + 2((lane>>1)&1) + 4(i>>2)
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int key = 8 * qp + (i & 1) + 2 * ((lane >> 1) & 1) + 4 * (i >> 2);
            const bool seen = ok[key] && (lim == nullptr || k0 + key <= lim[(i >> 1) & 1]);
            s[i] = seen ? s[i] * scale : -INFINITY;
            mx[(i >> 1) & 1] = fmaxf(mx[(i >> 1) & 1], s[i]);
        }
        float alpha[2], sum[2] = {0.f, 0.f};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 4));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 8));
            const float nm = fmaxf(m[r], mx[r]);
            alpha[r] = mx[r] == -INFINITY ? 1.f : (m[r] == -INFINITY ? 0.f : expf(m[r] - nm));
            m[r] = nm;
        }
        const int row0 = (lane & 1) + 4 * (lane >= 16);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = (i >> 1) & 1;
            const float p = s[i] == -INFINITY ? 0.f : expf(s[i] - m[r]);
            sum[r] += p;
            const int key = 8 * qp + (i & 1) + 2 * ((lane >> 1) & 1) + 4 * (i >> 2);
            P[(row0 + 2 * r) * S::PS + key] = __float2half_rn(p);
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            sum[r] += __shfl_xor_sync(0xffffffffu, sum[r], 2);
            sum[r] += __shfl_xor_sync(0xffffffffu, sum[r], 4);
            sum[r] += __shfl_xor_sync(0xffffffffu, sum[r], 8);
            l[r] = l[r] * alpha[r] + sum[r];
        }
#pragma unroll
        for (int t = 0; t < NT; ++t)
#pragma unroll
            for (int i = 0; i < 8; ++i) o[t][i] *= alpha[(i >> 1) & 1];
        __syncwarp();
#pragma unroll
        for (int kk = 0; kk < KT / 4; ++kk) {
            const uint2 a = *reinterpret_cast<const uint2*>(P + idx * S::PS + 4 * kk);
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                if constexpr (RV) {               // row-major values: key 4kk + lane%4, four dims of this quadpair's eight
                    const uint2 b = *reinterpret_cast<const uint2*>(
                        Vt + (4 * kk + (lane & 3)) * S::VS + 32 * t + 8 * qp + 4 * (lane >= 16));
                    mma884rr(o[t], a.x, a.y, b.x, b.y);
                } else {
                    const uint2 b = *reinterpret_cast<const uint2*>(Vt + (8 * (4 * t + qp) + idx) * S::VS + 4 * kk);
                    mma884(o[t], a.x, a.y, b.x, b.y);
                }
            }
        }
        __syncwarp();
    }
    // Prompts: o / l of each of the warp's eight rows as fp32 (``OUT`` row stride D), or nothing for padding rows.
    template <typename RowOf>
    __device__ __forceinline__ void finish(float* OUT, RowOf row_of) {
        const int lane = threadIdx.x & 31, qp = (lane >> 2) & 3;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = (lane & 1) + 2 * ((i >> 1) & 1) + 4 * (lane >= 16);
            long at;
            if (!row_of(r, at)) continue;
            const float inv = 1.f / l[(i >> 1) & 1];
#pragma unroll
            for (int t = 0; t < NT; ++t)
                OUT[at * D + 8 * (4 * t + qp) + (i & 1) + 2 * ((lane >> 1) & 1) + 4 * (i >> 2)] = o[t][i] * inv;
        }
    }
    // ``row_of(r, node, head)``: query row r of the warp's eight, or node < 0 for padding
    template <typename RowOf>
    __device__ __forceinline__ void store(float* PO, float* PM, float* PL, int chunk, int W, int H, RowOf row_of) {
        const int lane = threadIdx.x & 31, qp = (lane >> 2) & 3;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = (lane & 1) + 2 * ((i >> 1) & 1) + 4 * (lane >= 16);
            int node, head;
            row_of(r, node, head);
            if (node < 0) continue;
            const long bi = ((long)chunk * W + node) * H + head;
#pragma unroll
            for (int t = 0; t < NT; ++t)
                PO[bi * D + 8 * (4 * t + qp) + (i & 1) + 2 * ((lane >> 1) & 1) + 4 * (i >> 2)] = o[t][i];
            if (qp == 0 && ((lane >> 1) & 1) == 0 && (i & 5) == 0) { PM[bi] = m[(i >> 1) & 1]; PL[bi] = l[(i >> 1) & 1]; }
        }
    }
};

// Chunks wholly below a stream's committed keys: grid (chunk, key head, stream * tiles), ``warps`` query groups.
template <int D, bool QB, bool RV>
__global__ void __launch_bounds__(MAXW * 32) shared_kernel(
        const unsigned short* __restrict__ Q, const unsigned char* __restrict__ base, const long* __restrict__ OFF,
        const int* __restrict__ STREAM, float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL,
        int W, int H, int HK, int G, int tiles, int warps, float scale) {
    using S = Smem<D, RV>;
    constexpr int RB = Row<D, QB>::BYTES;                    // bytes a cached head-row
    extern __shared__ __align__(16) unsigned char smem[];
    const int warp = threadIdx.x >> 5;
    const int chunk = blockIdx.x, hk = blockIdx.y, s = blockIdx.z / tiles, tile = blockIdx.z % tiles;
    const int first = STREAM[s * 4], rows = STREAM[s * 4 + 1], p = STREAM[s * 4 + 2];
    if (chunk >= p / CH) return;
    const int q0 = tile * warps * 8, nq = min(warps * 8, rows * G - q0);     // (row, head) pairs of this block
    if (nq <= 0) return;
    __half* Qs = reinterpret_cast<__half*>(smem);
    __half* Ks = Qs + warps * 8 * S::QS;
    __half* Vt = Ks + KT * S::KS;
    __half* Ps = Vt + (RV ? KT : D) * S::VS;
    int* ok = reinterpret_cast<int*>(Ps + warps * 8 * S::PS);
    for (int e = threadIdx.x; e < warps * 8 * (D / 8); e += blockDim.x) {
        const int j = e / (D / 8), d = (e % (D / 8)) * 8;
        uint4 u = make_uint4(0, 0, 0, 0);
        if (j < nq) {
            const int pair = q0 + j, node = first + pair / G, head = hk * G + pair % G;
            u = h8(*reinterpret_cast<const uint4*>(Q + ((long)node * H + head) * D + d));
        }
        *reinterpret_cast<uint4*>(Qs + j * S::QS + d) = u;
    }
    const long koff = OFF[s * 2] * 2, voff = OFF[s * 2 + 1] * 2;
    Warp<D, RV> acc;
    acc.init();
    auto src = [&](int i, const unsigned char*& kp, const unsigned char*& vp) {
        kp = base + koff + ((long)i * HK + hk) * RB;
        vp = base + voff + ((long)i * HK + hk) * RB;
        return true;
    };
    KVTile<D, QB, RV> tl;
    tl.fetch(chunk * CH, src);
    for (int k0 = chunk * CH; k0 < chunk * CH + CH; k0 += KT) {
        __syncthreads();
        tl.put(Ks, Vt, ok);
        __syncthreads();
        if (k0 + KT < chunk * CH + CH) tl.fetch(k0 + KT, src);
        if (warp * 8 < nq) acc.step(Qs + warp * 8 * S::QS, Ks, Vt, ok, Ps + warp * 8 * S::PS, scale);
    }
    if (warp * 8 < nq)
        acc.store(PO, PM, PL, chunk, W, H, [&](int r, int& node, int& head) {
            const int j = warp * 8 + r;
            if (j >= nq) { node = -1; return; }
            node = first + (q0 + j) / G;
            head = hk * G + (q0 + j) % G;
        });
}

// The rest of each row's chunks (from the one holding its stream's committed count): grid (row, key head, tail).
template <int D, bool QB, bool RV>
__global__ void __launch_bounds__(TW * 32) tail_kernel(
        const unsigned short* __restrict__ Q, const unsigned char* __restrict__ KN, const unsigned char* __restrict__ VN,
        const unsigned char* __restrict__ base, const long* __restrict__ OFF, const int* __restrict__ STREAM,
        const int* __restrict__ ROWS, const int* __restrict__ PATHS, const int* __restrict__ DEPTHS,
        float* __restrict__ PO, float* __restrict__ PM, float* __restrict__ PL, int W, int H, int HK, int G, int maxd,
        float scale) {
    using S = Smem<D, RV>;
    constexpr int RB = Row<D, QB>::BYTES;                    // bytes a cached (or window) head-row
    extern __shared__ __align__(16) unsigned char smem[];
    const int node = blockIdx.x, hk = blockIdx.y;
    const int s = ROWS[node];
    const int p = STREAM[s * 4 + 2], nch = STREAM[s * 4 + 3];
    const int chunk = p / CH + blockIdx.z;
    if (chunk >= nch) return;
    const int depth = DEPTHS[node];
    const long koff = OFF[s * 2] * 2, voff = OFF[s * 2 + 1] * 2;
    __half* Qs = reinterpret_cast<__half*>(smem);
    __half* Ks = Qs + 8 * S::QS;
    __half* Vt = Ks + KT * S::KS;
    __half* Ps = Vt + (RV ? KT : D) * S::VS;
    int* ok = reinterpret_cast<int*>(Ps + 8 * S::PS);
    for (int e = threadIdx.x; e < 8 * (D / 8); e += blockDim.x) {
        const int j = e / (D / 8), d = (e % (D / 8)) * 8;
        const uint4 u = j < G ? h8(*reinterpret_cast<const uint4*>(Q + ((long)node * H + hk * G + j) * D + d))
                              : make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(Qs + j * S::QS + d) = u;
    }
    Warp<D, RV> acc;
    acc.init();
    const int kend = min(chunk * CH + CH, p + depth);
    auto src = [&](int i, const unsigned char*& kp, const unsigned char*& vp) {
        if (i < p) {
            kp = base + koff + ((long)i * HK + hk) * RB;
            vp = base + voff + ((long)i * HK + hk) * RB;
            return true;
        }
        if (i - p < depth) {
            const int pn = PATHS[node * maxd + (i - p)];
            kp = KN + ((long)pn * HK + hk) * RB;
            vp = VN + ((long)pn * HK + hk) * RB;
            return true;
        }
        return false;
    };
    KVTile<D, QB, RV> tl;
    tl.fetch(chunk * CH, src);
    for (int k0 = chunk * CH; k0 < kend; k0 += KT) {
        __syncthreads();
        tl.put(Ks, Vt, ok);
        __syncthreads();
        if (k0 + KT < kend) tl.fetch(k0 + KT, src);
        if (threadIdx.x < 32) acc.step(Qs, Ks, Vt, ok, Ps, scale);
    }
    if (threadIdx.x < 32)
        acc.store(PO, PM, PL, chunk, W, H, [&](int r, int& n, int& head) {
            if (r >= G) { n = -1; return; }
            n = node;
            head = hk * G + r;
        });
}

// Prompts, causal: grid (pair tiles, key head). A block takes ``warps`` x 8 (row, head) pairs of one key head and walks
// every 32-key sub-tile from key 0 to its last row's position, in order, an online softmax a row in registers: no
// scores or partials in DRAM. A row's bits follow from its position alone (a sub-tile past it is a no-op), so any
// chunking or tiling of the prompt rows gives the same bits. Caches bf16, or packed int8 rows read as decode reads them.
template <int D, bool QB, bool RV>
__global__ void __launch_bounds__(MAXW * 32) prompt_kernel(
        const unsigned short* __restrict__ Q, const unsigned char* __restrict__ KC, const unsigned char* __restrict__ VC,
        float* __restrict__ OUT, int W, int H, int HK, int G, int p0, int warps, float scale) {
    using S = Smem<D, RV>;
    constexpr int RB = Row<D, QB>::BYTES;
    extern __shared__ __align__(16) unsigned char smem[];
    const int warp = threadIdx.x >> 5, hk = blockIdx.y;
    const int q0 = blockIdx.x * warps * 8, nq = min(warps * 8, W * G - q0);
    if (nq <= 0) return;
    __half* Qs = reinterpret_cast<__half*>(smem);
    __half* Ks = Qs + warps * 8 * S::QS;
    __half* Vt = Ks + KT * S::KS;
    __half* Ps = Vt + (RV ? KT : D) * S::VS;
    int* ok = reinterpret_cast<int*>(Ps + warps * 8 * S::PS);
    for (int e = threadIdx.x; e < warps * 8 * (D / 8); e += blockDim.x) {
        const int j = e / (D / 8), d = (e % (D / 8)) * 8;
        uint4 u = make_uint4(0, 0, 0, 0);
        if (j < nq) {
            const int pair = q0 + j, node = pair / G, head = hk * G + pair % G;
            u = h8(*reinterpret_cast<const uint4*>(Q + ((long)node * H + head) * D + d));
        }
        *reinterpret_cast<uint4*>(Qs + j * S::QS + d) = u;
    }
    const int lane = threadIdx.x & 31;
    int lim[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {                            // this thread's two rows (Warp's row layout)
        const int j = warp * 8 + (lane & 1) + 2 * r + 4 * (lane >= 16);
        lim[r] = j < nq ? p0 + (q0 + j) / G : -1;
    }
    const int kend = p0 + (q0 + nq - 1) / G + 1;             // keys through the block's last row
    Warp<D, RV> acc;
    acc.init();
    auto src = [&](int i, const unsigned char*& kp, const unsigned char*& vp) {
        kp = KC + ((long)i * HK + hk) * RB;
        vp = VC + ((long)i * HK + hk) * RB;
        return i < kend;
    };
    KVTile<D, QB, RV> tl;
    tl.fetch(0, src);
    for (int k0 = 0; k0 < kend; k0 += KT) {
        __syncthreads();
        tl.put(Ks, Vt, ok);
        __syncthreads();
        if (k0 + KT < kend) tl.fetch(k0 + KT, src);
        if (warp * 8 < nq) acc.step(Qs + warp * 8 * S::QS, Ks, Vt, ok, Ps + warp * 8 * S::PS, scale, k0, lim);
    }
    if (warp * 8 < nq)
        acc.finish(OUT, [&](int r, long& at) {
            const int j = warp * 8 + r;
            if (j >= nq) return false;
            const int pair = q0 + j;
            at = (long)(pair / G) * H + hk * G + pair % G;
            return true;
        });
}

}  // namespace

// ``q`` (W, H, D) bf16; ``kn``, ``vn`` the window's own keys and values (W, HK, D) bf16, or packed int8 rows
// (W, HK, D + D / 16) when the caches hold them too; ``offs`` (S, 2) each stream's
// key and value cache as bf16 element offsets from ``base`` (attention.offsets); ``streams``, ``rows``, ``paths``,
// ``depths`` the plan; ``po``, ``pm``, ``pl`` the partials.
void chunks(torch::Tensor q, torch::Tensor kn, torch::Tensor vn, torch::Tensor base, torch::Tensor offs,
            torch::Tensor streams, torch::Tensor rows, torch::Tensor paths, torch::Tensor depths, torch::Tensor po,
            torch::Tensor pm, torch::Tensor pl, int64_t n_chunks, double scale, bool rv) {
    const int W = q.size(0), H = q.size(1), D = q.size(2), HK = kn.size(1), G = H / HK, NS = streams.size(0);
    TORCH_CHECK(G <= 8 && (D == 128 || D == 256), "attention_volta: up to 8 query heads a key head, head dim 128 or 256");
    TORCH_CHECK(offs.scalar_type() == at::kLong && paths.scalar_type() == at::kInt, "attention_volta: index types");
    const bool packed = kn.scalar_type() == at::kChar;
    TORCH_CHECK((kn.scalar_type() == at::kBFloat16 && kn.size(2) == D) || (packed && kn.size(2) == D + D / 16),
                "attention_volta: bf16 keys of the head dim, or packed int8 rows");
    TORCH_CHECK(vn.scalar_type() == kn.scalar_type() && vn.sizes() == kn.sizes(), "attention_volta: values as keys");
    TORCH_CHECK(paths.size(1) <= MAX_NODES, "attention_volta: a path of at most ", MAX_NODES, " rows");
    if (W == 0 || n_chunks == 0) return;
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto u = [](const torch::Tensor& t) { return reinterpret_cast<const unsigned short*>(t.data_ptr<at::BFloat16>()); };
    auto b8 = [](const torch::Tensor& t) { return reinterpret_cast<const unsigned char*>(t.data_ptr()); };
    const int pairs = W * G;                                   // a stream has at most all of them
    const int warps = std::min(MAXW, (pairs + 7) / 8);
    const int tiles = (pairs + warps * 8 - 1) / (warps * 8);
#define LAUNCH(DV, QV, RVV)                                                                                             \
    {                                                                                                            \
        static bool armed = false;                             /* the opt-in past 48 KB, once an instantiation */ \
        if (!armed) {                                                                                            \
            C10_CUDA_CHECK(cudaFuncSetAttribute(shared_kernel<DV, QV, RVV>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                                (int)Smem<DV, RVV>::bytes(MAXW)));                                    \
            C10_CUDA_CHECK(cudaFuncSetAttribute(tail_kernel<DV, QV, RVV>, cudaFuncAttributeMaxDynamicSharedMemorySize,  \
                                                (int)Smem<DV, RVV>::bytes(1)));                                       \
            armed = true;                                                                                        \
        }                                                                                                        \
        if (n_chunks > 1) {                                                                                      \
            shared_kernel<DV, QV, RVV><<<dim3(n_chunks, HK, NS * tiles), std::max(LOADERS, warps * 32),               \
                                    Smem<DV, RVV>::bytes(warps), stream>>>(                                           \
                u(q), b8(base), offs.data_ptr<long>(), streams.data_ptr<int>(), po.data_ptr<float>(),            \
                pm.data_ptr<float>(), pl.data_ptr<float>(), W, H, HK, G, tiles, warps, (float)scale);            \
            C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                      \
        }                                                                                                        \
        tail_kernel<DV, QV, RVV><<<dim3(W, HK, TAIL_CHUNKS), TW * 32, Smem<DV, RVV>::bytes(1), stream>>>(                   \
            u(q), b8(kn), b8(vn), b8(base), offs.data_ptr<long>(), streams.data_ptr<int>(), rows.data_ptr<int>(),\
            paths.data_ptr<int>(), depths.data_ptr<int>(), po.data_ptr<float>(), pm.data_ptr<float>(),           \
            pl.data_ptr<float>(), W, H, HK, G, (int)paths.size(1), (float)scale);                                \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                          \
    }
    if (rv) {
        if (packed) {
            if (D == 256) LAUNCH(256, true, true) else LAUNCH(128, true, true)
        } else {
            if (D == 256) LAUNCH(256, false, true) else LAUNCH(128, false, true)
        }
    } else if (packed) {
        if (D == 256) LAUNCH(256, true, false) else LAUNCH(128, true, false)
    } else {
        if (D == 256) LAUNCH(256, false, false) else LAUNCH(128, false, false)
    }
#undef LAUNCH
}

// ``q`` (W, H, D) bf16 at positions [p0, p0 + W); ``kc``, ``vc`` the caches (n >= p0 + W, HK, D) bf16 or packed int8
// rows; ``out`` (W, H, D) fp32. Pair tiles of ``warps`` x 8 (a function of the shape only).
void prompt(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor out, int64_t p0, double scale, bool rv) {
    const int W = q.size(0), H = q.size(1), D = q.size(2), HK = kc.size(1), G = H / HK;
    const bool packed = kc.scalar_type() == at::kChar;
    TORCH_CHECK(G <= 8 && (D == 128 || D == 256) && H % HK == 0, "attention_volta prompt: heads and head dim");
    TORCH_CHECK((kc.scalar_type() == at::kBFloat16 && kc.size(2) == D) || (packed && kc.size(2) == D + D / 16),
                "attention_volta prompt: bf16 caches of the head dim, or packed int8 rows");
    TORCH_CHECK(vc.scalar_type() == kc.scalar_type() && vc.sizes() == kc.sizes() && kc.size(0) >= p0 + W,
                "attention_volta prompt: caches through the chunk's keys");
    TORCH_CHECK(q.is_contiguous() && kc.is_contiguous() && vc.is_contiguous() && out.is_contiguous() &&
                out.scalar_type() == at::kFloat, "attention_volta prompt: contiguous tensors, fp32 out");
    if (W == 0) return;
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto u = [](const torch::Tensor& t) { return reinterpret_cast<const unsigned short*>(t.data_ptr<at::BFloat16>()); };
    auto b8 = [](const torch::Tensor& t) { return reinterpret_cast<const unsigned char*>(t.data_ptr()); };
    const int warps = MAXW, tiles = (W * G + warps * 8 - 1) / (warps * 8);
#define PROMPT(DV, QV, RVV)                                                                                             \
    {                                                                                                              \
        static bool armed = false;                                                                                 \
        if (!armed) {                                                                                              \
            C10_CUDA_CHECK(cudaFuncSetAttribute(prompt_kernel<DV, QV, RVV>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                                (int)Smem<DV, RVV>::bytes(MAXW)));                                      \
            armed = true;                                                                                          \
        }                                                                                                          \
        prompt_kernel<DV, QV, RVV><<<dim3(tiles, HK), MAXW * 32, Smem<DV, RVV>::bytes(warps), stream>>>(                    \
            u(q), b8(kc), b8(vc), out.data_ptr<float>(), W, H, HK, G, (int)p0, warps, (float)scale);              \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                            \
    }
    if (rv) {
        if (packed) {
            if (D == 256) PROMPT(256, true, true) else PROMPT(128, true, true)
        } else {
            if (D == 256) PROMPT(256, false, true) else PROMPT(128, false, true)
        }
    } else if (packed) {
        if (D == 256) PROMPT(256, true, false) else PROMPT(128, true, false)
    } else {
        if (D == 256) PROMPT(256, false, false) else PROMPT(128, false, false)
    }
#undef PROMPT
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prompt", &prompt, "Volta causal prompt attention: one pass over the keys a pair tile, nothing to DRAM");
    m.def("chunks", &chunks, "Volta tree attention on tensor cores: every chunk partial of every window row");
}
