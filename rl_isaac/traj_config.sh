#!/bin/bash
# ============================================================
# Shared trajectory config for all rl_isaac scripts.
# Edit these two values, then run any script:
#   ./rl_isaac/run_smoke_test.sh
#   ./rl_isaac/run_baseline.sh
#   ./rl_isaac/evaluate_policy.sh
# ============================================================

# TRAJ_DIR="results/llm_iterations/jump_180"
# ITER_NUM=18
TRAJ_DIR="results/llm_iterations/do_a_backflip"
ITER_NUM=3

# ── Resolve file paths from TRAJ_DIR + ITER_NUM ──
STATE_TRAJ="$TRAJ_DIR/state_traj_iter_${ITER_NUM}.npy"
GRF_TRAJ="$TRAJ_DIR/grf_traj_iter_${ITER_NUM}.npy"
JOINT_VEL_TRAJ="$TRAJ_DIR/joint_vel_traj_iter_${ITER_NUM}.npy"
CONTACT_SEQ="$TRAJ_DIR/contact_sequence_iter_${ITER_NUM}.npy"
PLANNED_VIDEO="$TRAJ_DIR/planned_traj_iter_${ITER_NUM}.mp4"
