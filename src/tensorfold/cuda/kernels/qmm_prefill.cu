// The prompt matmul's entry: 4-bit kernels here, other widths in qmm_prefill5.cu, qmm_prefill6.cu, qmm_prefill8b.cu.

#include "qmm_prefill.cuh"

void qmm_prefill5(QMM_PREFILL_ARGS);
void qmm_prefill6(QMM_PREFILL_ARGS);
void qmm_prefill8b(QMM_PREFILL_ARGS);

// ``tile`` (0: 128x128, 3 stages; 1: 64x128; 2: 128x64; 3: 64x64; 4: 128x128, 4 stages; 5-11: 64x64 warp tiles) never
// changes a row's bits.
void qmm_prefill_cuda(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scales, const at::Tensor& biases,
                      at::Tensor& out, int N, int gs, bool f32, int tile, int bits) {
    switch (bits) {
        case 4: break;
        case 5: qmm_prefill5(x, w, scales, biases, out, N, gs, f32); return;
        case 6: qmm_prefill6(x, w, scales, biases, out, N, gs, f32); return;
        case 8: qmm_prefill8b(x, w, scales, biases, out, N, gs, f32); return;
        default: TORCH_CHECK(false, "lane formats: 4, 5, 6 or 8 bits, not ", bits);
    }
    if (gs == 64) { if (f32) dispatch<64, true>(tile, x, w, scales, biases, out, N); else dispatch<64, false>(tile, x, w, scales, biases, out, N); }
    else { if (f32) dispatch<32, true>(tile, x, w, scales, biases, out, N); else dispatch<32, false>(tile, x, w, scales, biases, out, N); }
}
