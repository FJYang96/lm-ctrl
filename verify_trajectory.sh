#!/bin/bash
# Verify MPC trajectory is physically feasible for Go2.
# Checks Yicong's concerns: torque limits, joint accelerations, GRF feasibility.
#
# Usage: ./verify_trajectory.sh [traj_dir] [iter_num]
# Or edit the defaults below:

set -e

# ── Edit these defaults for your trajectory ──
DEFAULT_TRAJ_DIR="results/llm_iterations/backflip" 
DEFAULT_ITER=20

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
import casadi as cs
import numpy as np
from liecasadi import SO3
import go2_config
from mpc.dynamics.model import KinoDynamic_Model
from rl_isaac.feedforward import FeedforwardComputer
from utils.conversion import (
    MPC_X_BASE_POS, MPC_X_BASE_VEL, MPC_X_BASE_EUL,
    MPC_X_BASE_ANG, MPC_X_Q_JOINTS,
)

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

# ── Build full inverse dynamics CasADi function ──
bp = cs.SX.sym('bp', 3)
brpy = cs.SX.sym('brpy', 3)
jp = cs.SX.sym('jp', 12)
bv = cs.SX.sym('bv', 6)
jv = cs.SX.sym('jv', 12)
ba = cs.SX.sym('ba', 6)
ja = cs.SX.sym('ja', 12)
grf_s = cs.SX.sym('grf', 12)

r, p, y = brpy[0], brpy[1], brpy[2]
H = cs.SX.eye(4)
H[0:3, 0:3] = SO3.from_euler(cs.vertcat(r, p, y)).as_matrix()
H[0:3, 3] = bp

M = kindyn.mass_mass_fun(H, jp)
h = kindyn.bias_force_fun(H, jp, bv, jv)
JtF = cs.SX.zeros(18)
for i, foot in enumerate(['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']):
    J = kindyn.kindyn.jacobian_fun(foot)(H, jp)[0:3, :]
    JtF += J.T @ grf_s[i*3:(i+1)*3]

a_base = cs.inv(M[0:6, 0:6]) @ (-h[0:6] + JtF[0:6] - M[0:6, 6:18] @ ja)
tau = M[6:18, 0:6] @ a_base + M[6:18, 6:18] @ ja + h[6:18] - JtF[6:18]

full_id_fn = cs.Function('full_id', [bp, brpy, jp, bv, jv, ba, ja, grf_s], [tau])

# ── Precompute accelerations (backward differences, matching MPC) ──
base_vel_all = np.hstack([state_traj[:, MPC_X_BASE_VEL], state_traj[:, MPC_X_BASE_ANG]])
# MPC uses: q_ddot_j[k] = (U[k] - U[k-1]) / dt  (backward difference)
base_acc = np.zeros((n, 6))
joint_acc = np.zeros((n, 12))
for _k in range(1, n):
    base_acc[_k] = (base_vel_all[_k] - base_vel_all[_k - 1]) / dt
    joint_acc[_k] = (jvel_traj[_k] - jvel_traj[_k - 1]) / dt

# ── 1. J^T·F torques (GRF contribution only) ──
print('=' * 60)
print('1. GRF-IMPLIED TORQUES (J^T F only)')
print('=' * 60)
max_jtf = np.zeros(12)
for k in range(n):
    tau_jtf = np.abs(ff.compute(
        state_traj[k, MPC_X_BASE_POS], state_traj[k, MPC_X_BASE_EUL],
        state_traj[k, MPC_X_Q_JOINTS], grf_traj[k],
    ))
    max_jtf = np.maximum(max_jtf, tau_jtf)
for j in range(12):
    status = 'OK' if max_jtf[j] <= TORQUE_LIMITS[j] else 'OVER'
    print(f'  {JOINT_NAMES[j]:10s} max={max_jtf[j]:6.1f} / {TORQUE_LIMITS[j]:.1f} Nm  [{status}]')
print()

# ── 2. Full inverse dynamics torques (actual motor torque) ──
print('=' * 60)
print('2. FULL INVERSE DYNAMICS TORQUES (actual motor torque)')
print('=' * 60)
max_tau = np.zeros(12)
torque_violations = 0
worst_violations = []
for k in range(n):
    bv_k = np.concatenate([state_traj[k, MPC_X_BASE_VEL], state_traj[k, MPC_X_BASE_ANG]])
    ba_k = base_acc[k] if k < len(base_acc) else np.zeros(6)
    ja_k = joint_acc[k] if k < len(joint_acc) else np.zeros(12)
    tau_full = np.abs(np.array(full_id_fn(
        state_traj[k, MPC_X_BASE_POS], state_traj[k, MPC_X_BASE_EUL],
        state_traj[k, MPC_X_Q_JOINTS], bv_k, jvel_traj[k], ba_k, ja_k, grf_traj[k],
    )).flatten())
    max_tau = np.maximum(max_tau, tau_full)
    exceeded = tau_full > TORQUE_LIMITS
    if np.any(exceeded):
        torque_violations += 1
        worst_j = np.argmax(tau_full - TORQUE_LIMITS)
        if len(worst_violations) < 5:
            worst_violations.append(
                f'  k={k} t={k*dt:.2f}s {JOINT_NAMES[worst_j]} '
                f'tau={tau_full[worst_j]:.1f}Nm limit={TORQUE_LIMITS[worst_j]:.1f}Nm'
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

# ── 3. GRF feasibility ──
print('=' * 60)
print('3. GRF FEASIBILITY')
print('=' * 60)
grf_z = grf_traj[:, 2::3]
weight = go2_config.composite_mass * 9.81
print(f'Max per-foot GRF_z: {np.round(np.max(grf_z, axis=0), 1).tolist()} N')
print(f'Max total GRF_z: {np.max(np.sum(grf_z, axis=1)):.1f} N ({np.max(np.sum(grf_z, axis=1))/weight:.1f}x BW)')
print(f'Limit per component: {go2_config.grf_limits} N')
print()

# ── Summary ──
print('=' * 60)
print('SUMMARY')
print('=' * 60)
jtf_pass = np.all(max_jtf <= TORQUE_LIMITS)
id_pass = torque_violations == 0
print(f'J^T*F torques:  {\"PASS\" if jtf_pass else \"FAIL\"}')
print(f'Full ID torques: {\"PASS\" if id_pass else \"FAIL\"} ({torque_violations} steps over limit)')
print(f'Overall: {\"PASS - trajectory feasible for Go2\" if id_pass else \"FAIL - full ID torques exceed actuator limits\"}')
"
