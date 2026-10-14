// The grouped lane matmul's 6-bit kernels, built beside the other widths.

#include "qmm_group.cuh"

void qmm_group6(QMM_GROUP_ARGS) { group_bits<6>(QMM_GROUP_PASS); }
