// The lane matmul's 8-bit kernels, built beside the other widths.

#include "qmm_lane.cuh"

void qmm_lane8(QMM_LANE_ARGS) { qmm_bits<8>(QMM_LANE_PASS); }
