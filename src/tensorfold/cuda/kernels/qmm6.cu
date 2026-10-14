// The lane matmul's 6-bit kernels, built beside the other widths.

#include "qmm_lane.cuh"

void qmm_lane6(QMM_LANE_ARGS) { qmm_bits<6>(QMM_LANE_PASS); }
