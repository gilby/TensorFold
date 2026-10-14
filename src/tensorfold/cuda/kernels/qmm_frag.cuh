// Shared by the 4, 5, 6 and 8-bit lane matmuls: cp.async staging, ldmatrix, the bf16 mma, B-fragment decoding.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace qmm_frag {

// A 64-column tile's group block: the 4-bit nibble plane as before, then above 4 bits each value's top E bits.
template <int GS, int BITS>
struct Planes {
    static_assert(BITS == 4 || BITS == 5 || BITS == 6 || BITS == 8, "lane formats: 4, 5, 6 or 8 bits");
    static constexpr int E = BITS == 8 ? 4 : BITS - 4;   // top bits a value keeps in the high plane
    static constexpr int V = GS / 32;                     // 32-input pieces a group
    static constexpr int LO = 8 * GS;                     // nibble-plane words of a tile's group block
    static constexpr int HI = 2 * GS * E;                 // high-plane words
    static constexpr int WORDS = LO + HI;
    static constexpr int BYTES = WORDS * 4;               // 64 * GS * BITS / 8
    static constexpr int C = E ? 4 / E : 1;               // (n8 tile, 32-input piece) combos a high word holds
    static constexpr uint32_t MASK = E == 1 ? 0x00010001u : E == 2 ? 0x00030003u : 0x000F000Fu;
};

// A lane's nibble word for n8 tile jg and piece v, and its high word shifted to that combo's bits.
template <int GS, int BITS>
__device__ __forceinline__ void lane_words(const uint32_t* pw, int jg, int lane, int v, uint32_t& lo, uint32_t& hi) {
    using P = Planes<GS, BITS>;
    const uint32_t* t = pw + (jg >> 3) * P::WORDS;
    const int j = jg & 7;
    lo = t[(j * 32 + lane) * P::V + v];
    if constexpr (P::E > 0) {
        const int c = j * P::V + v;
        hi = t[P::LO + (c / P::C) * 32 + lane] >> ((c % P::C) * 4 * P::E);
    } else {
        hi = 0;
    }
}

// Lane matmul tile shapes: BM rows by BN columns a block, WM x WN warps, each warp (BM / WM) x (BN / WN).
template <int GS, int BM, int BN, int WM, int WN, int STAGES, int BITS = 4>
struct LaneTile {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;               // m16 tiles a warp
    static constexpr int NT = BN / WN / 8;                // n8 tiles a warp
    static constexpr int ROW = GS * 2;                    // bytes of one input row a group
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = BM * ROW;                    // stage bytes: inputs,
    static constexpr int W = BN / 64 * Planes<GS, BITS>::BYTES;   // weights,
    static constexpr int S = BN * 2;                      // scales, biases (bf16),
    static constexpr int XS = BM * 4;                     // and input sums (fp32)
    static constexpr int STAGE = X + W + 2 * S + XS;
    static constexpr int PARTIALS = MT * NT * 4 * THREADS * 4;   // a K slice's partial, parked for the cluster sum
    static constexpr int SMEM = STAGES * STAGE > PARTIALS ? STAGES * STAGE : PARTIALS;
};

__device__ __forceinline__ uint32_t smem(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem(dst)), "l"(src));
}

// 16 bytes, or zeros without reading when ``read`` is false (a row past M: computed, never stored)
__device__ __forceinline__ void cp16z(void* dst, const void* src, bool read) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem(dst)), "l"(src), "r"(read ? 16 : 0));
}

__device__ __forceinline__ void cp8(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem(dst)), "l"(src));
}

__device__ __forceinline__ void cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem(dst)), "l"(src));
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem(p)));
}

__device__ __forceinline__ void ldmatrix2(uint32_t (&r)[2], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(smem(p)));
}

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The same mma from a zero accumulator (a group's first k16 step): no registers to clear.
__device__ __forceinline__ void mma0(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    const float z = 0.0f;
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%10, %10, %10, %10};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(z));
}

// Nibbles at bits [s, s + 4) and [16 + s, 20 + s) as a bf16 pair: exponent bits make 128 + q, then minus 128.
__device__ __forceinline__ uint32_t pair(uint32_t w, int s) {
    const uint32_t t = ((w >> s) & 0x000F000Fu) | 0x43004300u;
    uint32_t r;
#if __CUDA_ARCH__ >= 900
    asm("sub.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(t), "r"(0x43004300u));
#else
    // no sub.bf16x2 before sm_90: t * 1 - 128 rounds the exact q once too (an exact value, the same bits)
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x3F803F80u), "r"(0xC300C300u));
#endif
    return r;
}

// t - c on exact bf16 pairs: sub.bf16x2 from sm_90, an fma with -c before it (the same bits).
__device__ __forceinline__ uint32_t bsub2(uint32_t t, uint32_t c) {
    uint32_t r;
#if __CUDA_ARCH__ >= 900
    asm("sub.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(t), "r"(c));
#else
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x3F803F80u), "r"(c ^ 0x80008000u));
#endif
    return r;
}

// Values at nibble slots p and p + 4 as an exact bf16 pair (4 bits: pair()'s bits).
template <int BITS>
__device__ __forceinline__ uint32_t lane_pair(uint32_t lo, uint32_t hi, int p) {
    using P = Planes<32, BITS>;
    if constexpr (BITS == 8) {                   // 16 hi + lo from two exact nibble pairs, one exact fma
        const uint32_t l = bsub2(((lo >> (4 * p)) & 0x000F000Fu) | 0x43004300u, 0x43004300u);
        const uint32_t h = bsub2(((hi >> (4 * p)) & 0x000F000Fu) | 0x43004300u, 0x43004300u);
        uint32_t r;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(h), "r"(0x41804180u), "r"(l));
        return r;
    } else {
        uint32_t q = (lo >> (4 * p)) & 0x000F000Fu;
        if constexpr (P::E > 0) q |= ((hi >> (P::E * p)) & P::MASK) << 4;
        return bsub2(q | 0x43004300u, 0x43004300u);
    }
}

// Programmatic dependent launch (sm_90+; no-ops before, and when the launch did not ask for it): wait for the
// previous kernel's writes, and let the next kernel's blocks start once every block here has said so.
__device__ __forceinline__ void grid_wait() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
#endif
}

__device__ __forceinline__ void grid_launch() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
#endif
}

// Block b's (first row, first column), row tiles fastest in bands of ``group`` so blocks in flight share L2.
__device__ __forceinline__ int2 tile_of(int b, int M, int N, int BM, int BN, int group) {
    const int rows_t = (M + BM - 1) / BM, cols_t = (N + BN - 1) / BN, band = group * cols_t;
    const int first = b / band * group, in_band = b % band, height = min(group, rows_t - first);
    return make_int2((first + in_band % height) * BM, in_band / height * BN);
}

// A 16-byte chunk c of row r sits at chunk c ^ (r % CHUNKS) so ldmatrix's eight rows hit eight bank groups.
template <int CHUNKS>
__device__ __forceinline__ int swz(int r, int c) { return c ^ (r % CHUNKS); }

} // namespace qmm_frag
