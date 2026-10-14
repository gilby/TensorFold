// The lane matmul's entry: 4-bit kernels here, 5, 6 and 8-bit ones in qmm5.cu, qmm6.cu, qmm8.cu.

#include "qmm_lane.cuh"

void qmm_lane5(QMM_LANE_ARGS);
void qmm_lane6(QMM_LANE_ARGS);
void qmm_lane8(QMM_LANE_ARGS);

// Clusters hold up to 8 K slices (the portable size) from sm_90; more, unreduced slices or older GPUs use the buffer.
bool qmm_clusters(int SK, bool reduce) {
    return SK > 1 && SK <= 8 && reduce && at::cuda::getCurrentDeviceProperties()->major >= 9;
}

void qmm_cuda(const at::Tensor& x, const at::Tensor& xs, const at::Tensor& w, const at::Tensor& scales,
              const at::Tensor& biases, at::Tensor& out, const at::Tensor& part, int N, int SK, int gs, int bm,
              bool f32, bool reduce, int bits) {
    const int M = x.size(0);
    const bool cluster = qmm_clusters(SK, reduce);
    switch (bits) {
        case 4: qmm_bits<4>(QMM_LANE_PASS); break;
        case 5: qmm_lane5(QMM_LANE_PASS); break;
        case 6: qmm_lane6(QMM_LANE_PASS); break;
        case 8: qmm_lane8(QMM_LANE_PASS); break;
        default: TORCH_CHECK(false, "lane formats: 4, 5, 6 or 8 bits, not ", bits);
    }
    if (SK > 1 && reduce && !cluster) {
        const long long total = static_cast<long long>(M) * N;
        const int threads = 256;
        const int blocks = static_cast<int>((total + threads - 1) / threads);
        auto stream = at::cuda::getCurrentCUDAStream();
        if (f32) reduce_kernel<true><<<blocks, threads, 0, stream>>>(part.data_ptr<float>(), out.data_ptr(), total, SK);
        else reduce_kernel<false><<<blocks, threads, 0, stream>>>(part.data_ptr<float>(), out.data_ptr(), total, SK);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
