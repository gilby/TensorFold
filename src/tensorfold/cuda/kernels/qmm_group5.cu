// The grouped lane matmul's 5-bit kernels, built beside the other widths.

#include "qmm_group.cuh"

void qmm_group5(QMM_GROUP_ARGS) { group_bits<5>(QMM_GROUP_PASS); }
