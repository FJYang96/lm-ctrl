import time
import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from gym_quadruped import quadruped_env
from tqdm import tqdm

import config
from examples.model import KinoDynamic_Model

# ============================================================
# Config
# ============================================================
robot_name = "go2"
terrain_type = "flat"
sim_dt = 0.01
sim_duration = 2.0
if_render = False

print(f"Testing robot {robot_name} on terrain {terrain_type}")

# ============================================================
# Index conventions
# ============================================================
QP_BASE_POS = slice(0, 3)
QP_BASE_QUAT = slice(3, 7)
QP_JOINTS = slice(7, 19)

QV_BASE_LIN = slice(0, 3)
QV_BASE_ANG = slice(3, 6)
QV_JOINTS   = slice(6, 18)

MP_X_BASE_POS  = slice(0, 3)
MP_X_BASE_VEL  = slice(3, 6)
MP_X_BASE_EUL  = slice(6, 9)
MP_X_BASE_ANG  = slice(9, 12)
MP_X_Q         = slice(12, 24)

MP_U_QD        = slice(0, 12)
MP_U_CONTACT_F = slice(12, 24)

# ============================================================
# Utilities
# ============================================================
def quat_wxyz_to_rotmat(q):
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),   1-2*(x*x+y*y)],
    ])
    return R

def quat_wxyz_to_euler_xyz(q):
    R = quat_wxyz_to_rotmat(q)
    sy = -R[2,0]
    sy = np.clip(sy, -1.0, 1.0)
    roll  = np.arctan2(R[2,1], R[2,2])
    pitch = np.arcsin(sy)
    yaw   = np.arctan2(R[1,0], R[0,0])
    return np.array([roll, pitch, yaw])

# ============================================================
# Simple Unitree Go2 FK (approx)
# ============================================================
HIP_OFFSET_X = 0.183   # forward/back from COM
HIP_OFFSET_Y = 0.0838  # lateral offset
HIP_LINK = 0.2
UPPER_LINK = 0.2
LOWER_LINK = 0.2

LEG_ORDER = ["FL", "FR", "RL", "RR"]
LEG_OFFSETS = {
    "FL": np.array([ HIP_OFFSET_X,  HIP_OFFSET_Y, 0.0]),
    "FR": np.array([ HIP_OFFSET_X, -HIP_OFFSET_Y, 0.0]),
    "RL": np.array([-HIP_OFFSET_X,  HIP_OFFSET_Y, 0.0]),
    "RR": np.array([-HIP_OFFSET_X, -HIP_OFFSET_Y, 0.0]),
}

def foot_fk_local(joints):
    """Return foot pos in base frame for given [abd, hip, knee]"""
    abd, hip, knee = joints
    # Lateral offset from abduction
    y = HIP_OFFSET_Y * np.sign(abd if abd!=0 else 1)
    # Sagittal plane
    x = HIP_LINK * np.cos(hip)
    z = -HIP_LINK * np.sin(hip)
    x += UPPER_LINK * np.cos(hip + knee)
    z += -UPPER_LINK * np.sin(hip + knee)
    x += LOWER_LINK * np.cos(hip + knee)
    z += -LOWER_LINK * np.sin(hip + knee)
    return np.array([0.0, y, 0.0]) + np.array([x,0,z])

def feet_positions_world_from_qpos(env, qpos):
    base_pos = qpos[QP_BASE_POS]
    base_quat = qpos[QP_BASE_QUAT]
    R = quat_wxyz_to_rotmat(base_quat)

    feet = []
    for i, leg in enumerate(LEG_ORDER):
        joints = qpos[QP_JOINTS][3*i:3*(i+1)]
        p_local = LEG_OFFSETS[leg] + foot_fk_local(joints)
        p_world = base_pos + R @ p_local
        feet.append(p_world)
    return np.array(feet)

# ============================================================
# Projection & Integrator
# ============================================================
def project_ground_contact(base_pos, feet_w, xdot, ground_z=0.0):
    if np.any(feet_w[:,2] < ground_z):
        xdot = np.array(xdot, dtype=float)
        xdot[MP_X_BASE_VEL.start + 2] = 0.0
    return xdot

def rk4_step(fd_fun, x, u, p, dt):
    k1 = np.array(cs.DM(fd_fun(x[:,None], u[:,None], p[:,None])).toarray())[:,0]
    k2 = np.array(cs.DM(fd_fun((x + 0.5*dt*k1)[:,None], u[:,None], p[:,None])).toarray())[:,0]
    k3 = np.array(cs.DM(fd_fun((x + 0.5*dt*k2)[:,None], u[:,None], p[:,None])).toarray())[:,0]
    k4 = np.array(cs.DM(fd_fun((x + dt*k3)[:,None], u[:,None], p[:,None])).toarray())[:,0]
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ============================================================
# Env setup
# ============================================================
state_obs_names = quadruped_env.QuadrupedEnv._DEFAULT_OBS + (
    "base_ori_euler_xyz",
    "contact_state",
    "contact_forces",
)

env = quadruped_env.QuadrupedEnv(
    robot=robot_name,
    scene=terrain_type,
    ref_base_lin_vel=(0.5, 1.0),
    ground_friction_coeff=config.mu_friction,
    base_vel_command_type="forward",
    state_obs_names=state_obs_names,
    sim_dt=sim_dt,
)

initial_qpos = np.zeros(19)
initial_qpos[QP_BASE_POS] = [0.0, 0.0, 0.23]
initial_qpos[QP_BASE_QUAT] = [1.0, 0.0, 0.0, 0.0]
initial_qpos[QP_JOINTS] = [0.0, 1.0, -2.1]*4
initial_qvel = np.zeros(18)
state = env.reset(qpos=initial_qpos, qvel=initial_qvel)

# ============================================================
# Sim loop
# ============================================================
print("-"*20, "Simulating", "-"*20)
state_traj = [state]
actions = []
for _ in tqdm(range(int(sim_duration / sim_dt))):
    action = np.ones(12) * 0.2
    s, _, _, _, _ = env.step(action=action)
    state_traj.append(s)
    actions.append(action)
env.close()
actions = np.array(actions)

# ============================================================
# Build trajectories
# ============================================================
qpos_traj = np.array([s["qpos"] for s in state_traj])
qvel_traj = np.array([s["qvel"] for s in state_traj])
base_eul_traj = np.array([s["base_ori_euler_xyz"] for s in state_traj])
contact_state_traj = np.array([s["contact_state"] for s in state_traj])
contact_forces_traj = np.array([s["contact_forces"] for s in state_traj])

mpc_state_traj = np.concatenate(
    [qpos_traj[:, QP_BASE_POS],
     qvel_traj[:, QV_BASE_LIN],
     base_eul_traj,
     qvel_traj[:, QV_BASE_ANG],
     qpos_traj[:, QP_JOINTS],
     np.zeros((qpos_traj.shape[0], 6))],
    axis=1
)

mpc_input_traj = np.concatenate(
    [qvel_traj[:, QV_JOINTS],
     contact_forces_traj],
    axis=1
)

# ============================================================
# Forward dynamics model
# ============================================================
kinodynamic_model = KinoDynamic_Model(config)
kinodynamic_model.export_robot_model()

param = np.concatenate([
    np.ones(4),
    np.array([config.mu_friction]),
    np.zeros(4),
    np.zeros(3),
    np.array([0.0]),
    np.zeros(6),
    config.inertia.flatten(),
    np.array([config.mass])
])

T = mpc_state_traj.shape[0]
model_states_traj = np.zeros((T, 30))
model_states_traj[0] = mpc_state_traj[0]
fd = kinodynamic_model.forward_dynamics

print("-"*20, "FD with contact projection (RK4)", "-"*20)
for i in range(1, T):
    param[0:4] = contact_state_traj[i].astype(float)
    base_pos = model_states_traj[i-1, MP_X_BASE_POS]
    base_eul = model_states_traj[i-1, MP_X_BASE_EUL]
    param[9:12] = base_pos
    param[12] = base_eul[2]

    x = model_states_traj[i-1].copy()
    u = mpc_input_traj[i-1].copy()
    p = param.copy()

    x_next = rk4_step(fd, x, u, p, sim_dt)

    # FK-based projection
    qpos_next = np.zeros(19)
    qpos_next[QP_BASE_POS] = x_next[MP_X_BASE_POS]
    qpos_next[QP_BASE_QUAT] = [1.0, 0.0, 0.0, 0.0]  # quick approx
    qpos_next[QP_JOINTS] = x_next[MP_X_Q]
    feet_w = feet_positions_world_from_qpos(env, qpos_next)

    if np.any(feet_w[:,2] < 0.0):
        xdot = np.array(cs.DM(fd(x_next[:,None], u[:,None], p[:,None])).toarray())[:,0]
        xdot_proj = project_ground_contact(x_next[MP_X_BASE_POS], feet_w, xdot)
        x_next = x + sim_dt*xdot_proj

    model_states_traj[i] = x_next

# ============================================================
# Plot
# ============================================================
time_axis = np.arange(T)*sim_dt
fig, axs = plt.subplots(3, 4, figsize=(12, 8))
axs = axs.ravel()
labels = ["x","y","z","vx","vy","vz","roll","pitch","yaw","wx","wy","wz"]
for i in range(12):
    axs[i].plot(time_axis, model_states_traj[:, i], label="fd")
    axs[i].plot(time_axis, mpc_state_traj[:, i], label="sim")
    axs[i].set_title(labels[i])
    axs[i].grid(True, alpha=0.3)
    axs[i].legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/fd_debug_fixed.png", dpi=150)
print("Saved results/fd_debug_fixed.png")
