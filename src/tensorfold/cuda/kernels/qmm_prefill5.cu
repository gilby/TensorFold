// The prompt matmul's 5-bit kernels, built beside the other widths.

#include "qmm_prefill.cuh"

void qmm_prefill5(QMM_PREFILL_ARGS) { prefill_bits<5>(QMM_PREFILL_PASS); }
