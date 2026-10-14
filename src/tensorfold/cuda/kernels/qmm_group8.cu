// The grouped lane matmul's 8-bit kernels, built beside the other widths.

#include "qmm_group.cuh"

void qmm_group8(QMM_GROUP_ARGS) { group_bits<8>(QMM_GROUP_PASS); }
