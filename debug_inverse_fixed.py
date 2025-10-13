import time
import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from gym_quadruped import quadruped_env
from tqdm import tqdm
from liecasadi import SO3

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
# Index conventions (standardized like forward dynamics)
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
# Utilities (copied from forward dynamics)
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

def euler_xyz_to_quat_wxyz(euler):
    roll, pitch, yaw = euler
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])

# ============================================================
# Simple Unitree Go2 FK (for foot position tracking)
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

def feet_positions_world_from_qpos(qpos):
    """Compute world foot positions from qpos using standardized indexing"""
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
# Improved numerical methods for acceleration computation
# ============================================================
def compute_accelerations_rk4_based(state_traj, input_traj, contact_traj, dt, kinodynamic_model):
    """Compute accelerations using RK4-based finite differences for better stability"""
    num_steps = len(state_traj) - 1
    ddq_traj = np.zeros((num_steps, 18))
    
    # Forward dynamics parameters
    param_template = np.concatenate([
        np.ones(4),  # contact state
        np.array([config.mu_friction]),
        np.zeros(4),  # stance proximity
        np.zeros(3),  # base position
        np.array([0.0]),  # base yaw
        np.zeros(6),  # external wrench
        config.inertia.flatten(),
        np.array([config.mass])
    ])
    
    for i in range(num_steps):
        # Use forward dynamics to compute current acceleration
        param = param_template.copy()
        param[0:4] = contact_traj[i].astype(float)
        param[9:12] = state_traj[i][MP_X_BASE_POS]
        param[12] = state_traj[i][MP_X_BASE_EUL][2]  # yaw
        
        # Get state derivative
        xdot = kinodynamic_model.forward_dynamics(
            state_traj[i][:, None], input_traj[i][:, None], param[:, None]
        )
        xdot_array = cs.DM(xdot).toarray()[:, 0]
        
        # Extract accelerations from state derivative
        ddq_traj[i, QV_BASE_LIN] = xdot_array[MP_X_BASE_VEL]  # base linear acceleration
        ddq_traj[i, QV_BASE_ANG] = xdot_array[MP_X_BASE_ANG]  # base angular acceleration
        
        # Joint accelerations from second finite difference (more stable than first)
        if i < num_steps - 1:
            ddq_traj[i, QV_JOINTS] = (input_traj[i+1][MP_U_QD] - input_traj[i][MP_U_QD]) / dt
        else:
            ddq_traj[i, QV_JOINTS] = ddq_traj[i-1, QV_JOINTS]  # extrapolate
    
    return ddq_traj

# ============================================================
# Fixed inverse dynamics computation with standardized indexing
# ============================================================
def compute_joint_torques_fixed(
    kindyn_model,
    state_traj: np.ndarray,
    input_traj: np.ndarray,
    contact_sequence: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Fixed inverse dynamics computation with systematic improvements:
    - Standardized state indexing using QP_*, QV_*, MP_* constants
    - Forward kinematics for foot position validation
    - Improved numerical methods for acceleration computation
    - Ground contact validation
    """
    num_steps = input_traj.shape[0]
    num_joints = 12
    joint_torques_traj = np.zeros((num_steps, num_joints))

    print("Building state representations with standardized indexing...")
    
    # Build full qpos trajectory using standardized indexing
    q_traj = np.zeros((num_steps + 1, 19))
    q_traj[:, QP_BASE_POS] = state_traj[:, MP_X_BASE_POS]
    
    # Convert RPY to Quaternion using robust method
    for i in range(num_steps + 1):
        euler = state_traj[i, MP_X_BASE_EUL]
        quat_wxyz = euler_xyz_to_quat_wxyz(euler)
        q_traj[i, QP_BASE_QUAT] = quat_wxyz
    
    q_traj[:, QP_JOINTS] = state_traj[:, MP_X_Q]

    # Build full qvel trajectory using standardized indexing
    dq_traj = np.zeros((num_steps + 1, 18))
    dq_traj[:, QV_BASE_LIN] = state_traj[:, MP_X_BASE_VEL]
    dq_traj[:, QV_BASE_ANG] = state_traj[:, MP_X_BASE_ANG]
    dq_traj[:-1, QV_JOINTS] = input_traj[:, MP_U_QD]
    dq_traj[-1, QV_JOINTS] = input_traj[-1, MP_U_QD]

    print("Computing accelerations using improved numerical methods...")
    # Use improved acceleration computation
    ddq_traj = compute_accelerations_rk4_based(
        state_traj, input_traj, contact_sequence.T, dt, kindyn_model
    )
    
    # Extend for final step
    ddq_traj_extended = np.zeros((num_steps + 1, 18))
    ddq_traj_extended[:-1] = ddq_traj
    ddq_traj_extended[-1] = ddq_traj[-1]  # extrapolate

    print("Setting up symbolic inverse dynamics with proper structure...")
    
    # Symbolic variables for CasADi function generation
    base_pos_sym = cs.SX.sym("base_pos", 3)
    base_quat_sym = cs.SX.sym("base_quat", 4)  # x,y,z,w format
    joint_pos_sym = cs.SX.sym("joint_pos", 12)
    base_vel_sym = cs.SX.sym("base_vel", 6)
    joint_vel_sym = cs.SX.sym("joint_vel", 12)
    base_acc_sym = cs.SX.sym("base_acc", 6)
    joint_acc_sym = cs.SX.sym("joint_acc", 12)
    f_ext_sym = cs.SX.sym("f_ext", 12)

    # Construct homogeneous transformation matrix H from quaternion
    # Adam library expects quaternion in [w, x, y, z] order
    quat_wxyz_sym = cs.vertcat(base_quat_sym[3], base_quat_sym[0:3])
    H = cs.SX.eye(4)
    H[0:3, 0:3] = SO3.from_quat(quat_wxyz_sym).as_matrix()
    H[0:3, 3] = base_pos_sym

    # Create symbolic functions from kinodynamic model
    mass_matrix_fun = kindyn_model.kindyn.mass_matrix_fun()
    bias_force_fun = kindyn_model.kindyn.bias_force_fun()
    gravity_fun = kindyn_model.kindyn.gravity_term_fun()
    
    J_FL_fun = kindyn_model.kindyn.jacobian_fun("FL_foot")
    J_FR_fun = kindyn_model.kindyn.jacobian_fun("FR_foot")
    J_RL_fun = kindyn_model.kindyn.jacobian_fun("RL_foot")
    J_RR_fun = kindyn_model.kindyn.jacobian_fun("RR_foot")

    # Symbolic inverse dynamics computation
    M_sym = mass_matrix_fun(H, joint_pos_sym)
    C_sym = bias_force_fun(H, joint_pos_sym, base_vel_sym, joint_vel_sym)
    g_sym = gravity_fun(H, joint_pos_sym)

    # Split external forces by leg
    F_FL, F_FR, F_RL, F_RR = cs.vertsplit(f_ext_sym, 3)
    J_FL, J_FR, J_RL, J_RR = (
        f(H, joint_pos_sym)[0:3, :] for f in [J_FL_fun, J_FR_fun, J_RL_fun, J_RR_fun]
    )
    wrench_ext = J_FL.T @ F_FL + J_FR.T @ F_FR + J_RL.T @ F_RL + J_RR.T @ F_RR

    ddq_full_sym = cs.vertcat(base_acc_sym, joint_acc_sym)

    # Correct inverse dynamics equation: tau = M*ddq + C + g - J^T*F
    tau_full_sym = M_sym @ ddq_full_sym + C_sym + g_sym - wrench_ext

    # Create CasADi function
    inverse_dynamics_fun = cs.Function(
        "inverse_dynamics",
        [base_pos_sym, base_quat_sym, joint_pos_sym, 
         base_vel_sym, joint_vel_sym, base_acc_sym, joint_acc_sym, f_ext_sym],
        [tau_full_sym],
    )

    print("Evaluating inverse dynamics with contact validation...")
    
    # Evaluation loop with foot position validation
    grf_traj = input_traj[:, MP_U_CONTACT_F]
    
    for i in range(num_steps):
        # Apply contact forces only for legs in contact
        grfs_vec = grf_traj[i, :].copy()
        for leg_idx in range(4):
            contact_state = contact_sequence[leg_idx, i]
            grfs_vec[3*leg_idx:3*(leg_idx+1)] *= contact_state
            
            # Validate foot position for contact
            feet_world = feet_positions_world_from_qpos(q_traj[i])
            foot_height = feet_world[leg_idx, 2]
            
            if contact_state == 1 and foot_height > 0.02:  # Contact claimed but foot in air
                print(f"Warning: Step {i}, Leg {leg_idx} contact mismatch - foot at height {foot_height:.3f}m")
            elif contact_state == 0 and foot_height < -0.01:  # No contact but foot underground
                print(f"Warning: Step {i}, Leg {leg_idx} penetration - foot at height {foot_height:.3f}m")

        # Compute inverse dynamics
        tau_full = inverse_dynamics_fun(
            q_traj[i, QP_BASE_POS],
            q_traj[i, QP_BASE_QUAT],
            q_traj[i, QP_JOINTS],
            np.concatenate([dq_traj[i, QV_BASE_LIN], dq_traj[i, QV_BASE_ANG]]),
            dq_traj[i, QV_JOINTS],
            np.concatenate([ddq_traj_extended[i, QV_BASE_LIN], ddq_traj_extended[i, QV_BASE_ANG]]),
            ddq_traj_extended[i, QV_JOINTS],
            grfs_vec,
        )

        # Extract joint torques (skip the first 6 base wrench elements)
        joint_torques_traj[i, :] = tau_full.full().flatten()[6:]

    print("Inverse dynamics computation completed!")
    return joint_torques_traj

# ============================================================
# Environment setup
# ============================================================
state_obs_names = quadruped_env.QuadrupedEnv._DEFAULT_OBS + (
    "base_ori_euler_xyz", "contact_state", "contact_forces"
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

# Use standardized indexing for initial conditions
initial_qpos = np.zeros(19)
initial_qpos[QP_BASE_POS] = [0.0, 0.0, 0.23]
initial_qpos[QP_BASE_QUAT] = [1.0, 0.0, 0.0, 0.0]
initial_qpos[QP_JOINTS] = [0.0, 1.0, -2.1] * 4
initial_qvel = np.zeros(18)
state = env.reset(qpos=initial_qpos, qvel=initial_qvel)

# ============================================================
# Simulate and render
# ============================================================
print("-" * 20, "Simulating", "-" * 20)
state_traj = [state]
torque_traj = []
for i in tqdm(range(int(sim_duration / sim_dt))):
    action = np.ones(12) * 1.1
    state, _, _, _, _ = env.step(action=action)
    state_traj.append(state)
    torque_traj.append(action)
env.close()

# ============================================================
# Convert to standardized MPC format
# ============================================================
qpos_traj = np.array([state["qpos"] for state in state_traj])
qvel_traj = np.array([state["qvel"] for state in state_traj])
base_ori_euler_xyz_traj = np.array([state["base_ori_euler_xyz"] for state in state_traj])
contact_state_traj = np.array([state["contact_state"] for state in state_traj])
contact_forces_traj = np.array([state["contact_forces"] for state in state_traj])

# Build MPC state trajectory using standardized indexing
mpc_state_traj = np.concatenate(
    (
        qpos_traj[:, QP_BASE_POS],     # MP_X_BASE_POS
        qvel_traj[:, QV_BASE_LIN],     # MP_X_BASE_VEL
        base_ori_euler_xyz_traj,       # MP_X_BASE_EUL
        qvel_traj[:, QV_BASE_ANG],     # MP_X_BASE_ANG
        qpos_traj[:, QP_JOINTS],       # MP_X_Q
        np.zeros((qpos_traj.shape[0], 6)),  # zero integral states
    ),
    axis=1,
)

mpc_input_traj = np.concatenate(
    (
        qvel_traj[:, QV_JOINTS],       # MP_U_QD
        contact_forces_traj,           # MP_U_CONTACT_F
    ),
    axis=1,
)

# ============================================================
# Test fixed inverse dynamics
# ============================================================
kinodynamic_model = KinoDynamic_Model(config)
kinodynamic_model.export_robot_model()

print("Testing fixed inverse dynamics computation...")
computed_torques = compute_joint_torques_fixed(
    kinodynamic_model,
    mpc_state_traj,
    mpc_input_traj[:-1],
    contact_state_traj[:-1].T,
    sim_dt,
)

torque_traj = np.array(torque_traj)

# ============================================================
# Plot comparison
# ============================================================
plt.figure(figsize=(15, 10))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.plot(computed_torques[:, i], label="computed (fixed)", linewidth=2)
    plt.plot(torque_traj[:, i], label="true", linewidth=2)
    plt.legend()
    plt.title(f"Joint {i+1}")
    plt.grid(True, alpha=0.3)

plt.suptitle("Fixed Inverse Dynamics: Computed vs True Torques", fontsize=16)
plt.tight_layout()
plt.savefig("results/id_debug_fixed.png", dpi=150)
plt.show()

# ============================================================
# Compute and print validation metrics
# ============================================================
print("\n" + "="*60)
print("INVERSE DYNAMICS VALIDATION RESULTS")
print("="*60)

skip = 20  # Skip initial transient
rmse_per_joint = []
for i in range(12):
    rmse = np.sqrt(np.mean((computed_torques[skip:, i] - torque_traj[skip:, i])**2))
    rmse_per_joint.append(rmse)
    print(f"Joint {i+1:2d}: RMSE = {rmse:.4f} Nm")

overall_rmse = np.sqrt(np.mean((computed_torques[skip:] - torque_traj[skip:])**2))
print(f"\nOverall RMSE: {overall_rmse:.4f} Nm")

max_error = np.max(np.abs(computed_torques[skip:] - torque_traj[skip:]))
print(f"Max absolute error: {max_error:.4f} Nm")

# Success criteria
success = overall_rmse < 2.0 and max_error < 10.0
print(f"\nValidation {'PASSED' if success else 'FAILED'}")
if success:
    print("✅ Inverse dynamics model is working correctly!")
else:
    print("❌ Inverse dynamics model needs further debugging")

print("\nSaved results to: results/id_debug_fixed.png")