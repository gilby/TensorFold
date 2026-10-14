// The lane matmul's 5-bit kernels, built beside the other widths.

#include "qmm_lane.cuh"

void qmm_lane5(QMM_LANE_ARGS) { qmm_bits<5>(QMM_LANE_PASS); }
