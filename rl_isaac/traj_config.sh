#!/bin/bash
# ============================================================
# Shared trajectory config for all rl_isaac scripts.
# Edit these two values, then run any script:
#   ./rl_isaac/run_smoke_test.sh
#   ./rl_isaac/run_baseline.sh
#   ./rl_isaac/evaluate_policy.sh
# ============================================================

TRAJ_DIR="results/llm_iterations/do_a_backflip:_start_from_the_default_full_stance;_push_off_with_front_feet_first_to_almost_stand_up_and_then_take_off_from_back_legs_thrust;_land_first_on_the_front_feet_and_quickly_followed_by_rear_1776831174"
ITER_NUM=11

# ── Resolve file paths from TRAJ_DIR + ITER_NUM ──
STATE_TRAJ="$TRAJ_DIR/state_traj_iter_${ITER_NUM}.npy"
GRF_TRAJ="$TRAJ_DIR/grf_traj_iter_${ITER_NUM}.npy"
JOINT_VEL_TRAJ="$TRAJ_DIR/joint_vel_traj_iter_${ITER_NUM}.npy"
CONTACT_SEQ="$TRAJ_DIR/contact_sequence_iter_${ITER_NUM}.npy"
PLANNED_VIDEO="$TRAJ_DIR/planned_traj_iter_${ITER_NUM}.mp4"
