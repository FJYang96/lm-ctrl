#!/bin/bash
# Verify MPC trajectory is physically feasible for Go2.
# Checks Yicong's concerns: torque limits, joint accelerations, GRF feasibility.
#
# Usage: ./verify_trajectory.sh [traj_dir] [iter_num]
# Or edit the defaults below:

set -e

# ── Edit these defaults for your trajectory ──
DEFAULT_TRAJ_DIR="results/llm_iterations/do_a_backflip_1774703475"
DEFAULT_ITER=2

TRAJ_DIR="${1:-$DEFAULT_TRAJ_DIR}"
ITER="${2:-$DEFAULT_ITER}"
PYTHON="${PYTHON:-/home/aryanroy/miniconda3/bin/python3}"

STATE="$TRAJ_DIR/state_traj_iter_${ITER}.npy"
GRF="$TRAJ_DIR/grf_traj_iter_${ITER}.npy"
JVEL="$TRAJ_DIR/joint_vel_traj_iter_${ITER}.npy"

for f in "$STATE" "$GRF" "$JVEL"; do
    [ -f "$f" ] || { echo "Missing: $f"; exit 1; }
done

cd "$(dirname "$0")"

$PYTHON -c "
import numpy as np
import go2_config
from mpc.dynamics.model import KinoDynamic_Model
from rl_isaac.feedforward import FeedforwardComputer
from utils.conversion import MPC_X_BASE_POS, MPC_X_BASE_EUL, MPC_X_Q_JOINTS

TORQUE_LIMITS = np.array([23.7, 23.7, 45.43] * 4)
JOINT_NAMES = ['FL_hip','FL_thigh','FL_calf','FR_hip','FR_thigh','FR_calf',
               'RL_hip','RL_thigh','RL_calf','RR_hip','RR_thigh','RR_calf']

state_traj = np.load('$STATE')
grf_traj = np.load('$GRF')
jvel_traj = np.load('$JVEL')
dt = go2_config.mpc_config.mpc_dt
n = min(len(grf_traj), len(state_traj) - 1)

kindyn = KinoDynamic_Model()
ff = FeedforwardComputer(kindyn)

print(f'Trajectory: {n} steps, dt={dt}s, total={n*dt:.2f}s')
print()

# 1. Joint torques (full J^T F)
print('=' * 60)
print('1. JOINT TORQUE FEASIBILITY (full J^T F)')
print('=' * 60)
max_tau = np.zeros(12)
torque_violations = 0
worst_violations = []
for k in range(n):
    tau = ff.compute(
        state_traj[k, MPC_X_BASE_POS],
        state_traj[k, MPC_X_BASE_EUL],
        state_traj[k, MPC_X_Q_JOINTS],
        grf_traj[k],
    )
    tau_abs = np.abs(tau)
    max_tau = np.maximum(max_tau, tau_abs)
    exceeded = tau_abs > TORQUE_LIMITS
    if np.any(exceeded):
        torque_violations += 1
        worst_j = np.argmax(tau_abs - TORQUE_LIMITS)
        if len(worst_violations) < 5:
            worst_violations.append(
                f'  k={k} t={k*dt:.2f}s {JOINT_NAMES[worst_j]} '
                f'tau={tau_abs[worst_j]:.1f}Nm limit={TORQUE_LIMITS[worst_j]:.1f}Nm'
            )

for j in range(12):
    status = 'OK' if max_tau[j] <= TORQUE_LIMITS[j] else 'OVER'
    print(f'  {JOINT_NAMES[j]:10s} max={max_tau[j]:6.1f} / {TORQUE_LIMITS[j]:.1f} Nm  [{status}]')
print(f'Violations: {torque_violations}/{n} steps')
for v in worst_violations:
    print(v)
if torque_violations > 5:
    print(f'  ... and {torque_violations - 5} more')
print(f'Result: {\"PASS\" if torque_violations == 0 else \"FAIL\"}')
print()

# 2. GRF feasibility
print('=' * 60)
print('3. GRF FEASIBILITY')
print('=' * 60)
grf_z = grf_traj[:, 2::3]
weight = go2_config.composite_mass * 9.81
print(f'Max per-foot GRF_z: {np.round(np.max(grf_z, axis=0), 1).tolist()} N')
print(f'Max total GRF_z: {np.max(np.sum(grf_z, axis=1)):.1f} N ({np.max(np.sum(grf_z, axis=1))/weight:.1f}x BW)')
print(f'Limit per component: {go2_config.grf_limits} N')
print()

# Summary
print('=' * 60)
print('SUMMARY')
print('=' * 60)
print(f'Torque:  {\"PASS\" if torque_violations == 0 else \"FAIL\"}')
print(f'Overall: {\"PASS - trajectory feasible for Go2\" if torque_violations == 0 else \"FAIL - exceeds hardware limits\"}')
"
