// The grouped lane matmul's entry: 4-bit kernels here, other widths in qmm_group5.cu, qmm_group6.cu, qmm_group8.cu.

#include "qmm_group.cuh"

void qmm_group5(QMM_GROUP_ARGS);
void qmm_group6(QMM_GROUP_ARGS);
void qmm_group8(QMM_GROUP_ARGS);

void qmm_group_cuda(const at::Tensor& x, const at::Tensor& xs, const std::vector<at::Tensor>& ws,
                    const std::vector<at::Tensor>& scales, const std::vector<at::Tensor>& biases,
                    std::vector<at::Tensor>& outs, const std::vector<int64_t>& ns, const std::vector<int64_t>& sks,
                    bool f32, int tile, int pdl, int bits) {
    Parts parts = {};
    parts.count = static_cast<int>(ws.size());
    int C = 1;
    for (int i = 0; i < parts.count; ++i) {
        Part& P = parts.p[i];
        P.w = reinterpret_cast<const uint32_t*>(ws[i].data_ptr());
        P.scales = reinterpret_cast<const __nv_bfloat16*>(scales[i].data_ptr());
        P.biases = reinterpret_cast<const __nv_bfloat16*>(biases[i].data_ptr());
        P.out = outs[i].data_ptr();
        P.n = static_cast<int>(ns[i]);
        P.npad = static_cast<int>(scales[i].stride(0));
        P.sk = static_cast<int>(sks[i]);
        C = std::max(C, P.sk);
    }
    const auto* props = at::cuda::getCurrentDeviceProperties();
    const bool gb10 = props->major == 12 && props->minor == 1;
    const bool early = pdl < 0 ? gb10 : pdl > 0;           // overlapping launches pay on GB10, cost the capped PRO 6000
    const int M = x.size(0);
    switch (bits) {
        case 4: group_bits<4>(QMM_GROUP_PASS); break;
        case 5: qmm_group5(QMM_GROUP_PASS); break;
        case 6: qmm_group6(QMM_GROUP_PASS); break;
        case 8: qmm_group8(QMM_GROUP_PASS); break;
        default: TORCH_CHECK(false, "lane formats: 4, 5, 6 or 8 bits, not ", bits);
    }
}
