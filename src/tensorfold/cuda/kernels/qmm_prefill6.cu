// The prompt matmul's 6-bit kernels, built beside the other widths.

#include "qmm_prefill.cuh"

void qmm_prefill6(QMM_PREFILL_ARGS) { prefill_bits<6>(QMM_PREFILL_PASS); }
