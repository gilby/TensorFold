// The prompt matmul's 8-bit kernels, built beside the other widths.

#include "qmm_prefill.cuh"

void qmm_prefill8b(QMM_PREFILL_ARGS) { prefill_bits<8>(QMM_PREFILL_PASS); }
