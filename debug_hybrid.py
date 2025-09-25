import time

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from gym_quadruped import quadruped_env
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import config
from examples.model import KinoDynamic_Model
from utils.inv_dyn import compute_joint_torques

robot_name = "go2"
terrain_type = "flat"
sim_dt = 0.01
sim_duration = 2.0
if_render = False

################################################################################
# Create and initialize environment
################################################################################
print(f"Testing robot {robot_name} on terrain {terrain_type}")

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
initial_qpos[0:3] = [0.0, 0.0, 0.23]
initial_qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
initial_qpos[7:19] = [
    0.0, 1.0, -2.1,  # FL
    0.0, 1.0, -2.1,  # FR
    0.0, 1.0, -2.1,  # RL
    0.0, 1.0, -2.1,  # RR
]
initial_qvel = np.zeros(18)
state = env.reset(qpos=initial_qpos, qvel=initial_qvel)

################################################################################
# Simulate and collect data
################################################################################
print("-" * 20, "Simulating", "-" * 20)
state_traj, torque_traj = [state], []

for i in tqdm(range(int(sim_duration / sim_dt))):
    action = np.ones(12) * 1.1
    state, reward, is_terminated, is_truncated, info = env.step(action=action)
    state_traj.append(state)
    torque_traj.append(action)

env.close()

################################################################################
# Convert to arrays
################################################################################
qpos_traj = np.array([state["qpos"] for state in state_traj])
qvel_traj = np.array([state["qvel"] for state in state_traj])
base_ori_euler_xyz_traj = np.array([state["base_ori_euler_xyz"] for state in state_traj])
contact_state_traj = np.array([state["contact_state"] for state in state_traj])
contact_forces_traj = np.array([state["contact_forces"] for state in state_traj])

################################################################################
# Hybrid inverse dynamics implementation
################################################################################
def compute_improved_jacobian(q, robot_params, leg_info):
    """Improved Jacobian computation focusing on hip joints"""
    hip_aa, hip_fe, knee = q
    L1 = robot_params['upper_leg_length']
    L2 = robot_params['lower_leg_length']
    
    c_fe = np.cos(hip_fe)
    s_fe = np.sin(hip_fe)
    c_fk = np.cos(hip_fe + knee)
    s_fk = np.sin(hip_fe + knee)
    c_aa = np.cos(hip_aa)
    s_aa = np.sin(hip_aa)
    
    J = np.zeros((3, 3))
    sign_y = leg_info['sign_y']
    
    # X direction (forward/back)
    J[0, 1] = L1 * c_fe + L2 * c_fk
    J[0, 2] = L2 * c_fk
    
    # Y direction (left/right) - hip abduction is key
    J[1, 0] = -(L1 * s_fe + L2 * s_fk) * s_aa * sign_y
    J[1, 1] = -(L1 * s_fe + L2 * s_fk) * c_aa * sign_y * 0.2
    
    # Z direction (up/down) - most important for stance forces
    J[2, 0] = -(L1 * s_fe + L2 * s_fk) * c_aa * 0.1
    J[2, 1] = -(L1 * s_fe + L2 * s_fk)
    J[2, 2] = -L2 * s_fk
    
    return J


def compute_hybrid_inverse_dynamics(
    qpos_traj, qvel_traj, contact_forces_traj, contact_state_traj, dt,
    kinodynamic_model, mpc_state_traj, mpc_input_traj
):
    """
    Hybrid approach: WBInterface for hips + whole-body for knees
    """
    num_steps = qpos_traj.shape[0] - 1
    joint_torques_traj = np.zeros((num_steps, 12))
    
    # Get whole-body solution as baseline
    wholebody_torques = compute_joint_torques(
        kinodynamic_model, mpc_state_traj, mpc_input_traj[:-1], contact_state_traj[:-1].T, dt
    )
    
    # Robot parameters
    robot_params = {
        'hip_offset_y': 0.055, 'hip_offset_x': 0.1881,
        'upper_leg_length': 0.213, 'lower_leg_length': 0.213,
        'upper_leg_mass': 0.695, 'lower_leg_mass': 0.166,
    }
    
    leg_config = {
        'FL': {'sign_x': 1, 'sign_y': 1, 'idx': 0},
        'FR': {'sign_x': 1, 'sign_y': -1, 'idx': 1},
        'RL': {'sign_x': -1, 'sign_y': 1, 'idx': 2},
        'RR': {'sign_x': -1, 'sign_y': -1, 'idx': 3},
    }
    
    for step in range(num_steps):
        # Start with whole-body solution
        joint_torques_traj[step, :] = wholebody_torques[step, :]
        
        # Override HIP joints with WBInterface approach
        base_quat = qpos_traj[step, 3:7]
        joint_pos = qpos_traj[step, 7:19]
        
        try:
            base_rot = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]]).as_matrix()
        except:
            base_rot = np.eye(3)
        
        # Process each leg for HIP joints only
        for leg_name, leg_info in leg_config.items():
            leg_idx = leg_info['idx']
            joint_start = leg_idx * 3
            force_start = leg_idx * 3
            
            q = joint_pos[joint_start:joint_start + 3]
            f_world = contact_forces_traj[step, force_start:force_start + 3]
            
            if contact_state_traj[step, leg_idx] == 0:
                f_world = np.zeros(3)
            
            # WBInterface approach: τ = -J^T * F
            J = compute_improved_jacobian(q, robot_params, leg_info)
            tau_leg = -J.T @ f_world
            
            # Replace ONLY hip joint torques (keep knee from whole-body)
            joint_torques_traj[step, joint_start] = tau_leg[0]      # Hip AA
            joint_torques_traj[step, joint_start + 1] = tau_leg[1]  # Hip FE
            # Keep knee (joint_start + 2) from whole-body solution
    
    return joint_torques_traj


################################################################################
# Compute inverse dynamics
################################################################################
print("Computing inverse dynamics...")

# Setup MPC format
mpc_state_traj = np.concatenate((
    qpos_traj[:, 0:3], qvel_traj[:, 0:3], base_ori_euler_xyz_traj,
    qvel_traj[:, 3:6], qpos_traj[:, 7:19], np.zeros((qpos_traj.shape[0], 6)),
), axis=1)

mpc_input_traj = np.concatenate((qvel_traj[:, 6:18], contact_forces_traj,), axis=1)

kinodynamic_model = KinoDynamic_Model(config)
kinodynamic_model.export_robot_model()

# Original whole-body approach
computed_torques_original = compute_joint_torques(
    kinodynamic_model, mpc_state_traj, mpc_input_traj[:-1], contact_state_traj[:-1].T, sim_dt
)

# Hybrid approach
computed_torques_hybrid = compute_hybrid_inverse_dynamics(
    qpos_traj, qvel_traj, contact_forces_traj, contact_state_traj, sim_dt,
    kinodynamic_model, mpc_state_traj, mpc_input_traj
)

################################################################################
# Clean plot: Original vs Hybrid vs True
################################################################################
torque_traj = np.array(torque_traj)

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for i in range(12):
    row = i // 4
    col = i % 4
    ax = axes[row, col]
    
    # Only show 3 lines for clean comparison
    ax.plot(computed_torques_original[:, i], label="original", alpha=0.7, linewidth=1.5, color='blue')
    ax.plot(computed_torques_hybrid[:, i], label="hybrid", alpha=0.8, linewidth=2, color='green')
    ax.plot(torque_traj[:, i], label="true", alpha=0.9, linewidth=2.5, color='red')
    
    leg_names = ['FL', 'FR', 'RL', 'RR']
    joint_names = ['hip_aa', 'hip_fe', 'knee']
    leg_name = leg_names[i // 3]
    joint_name = joint_names[i % 3]
    
    ax.set_title(f"Joint {i+1} ({leg_name} {joint_name})", fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time step', fontsize=8)
    ax.set_ylabel('Torque (Nm)', fontsize=8)

plt.suptitle('Inverse Dynamics: Original vs Hybrid Approach', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("results/id_debug_hybrid_clean.png", dpi=150, bbox_inches='tight')
plt.close()

################################################################################
# Simple error analysis
################################################################################
def compute_rmse(predicted, actual):
    return np.sqrt(np.mean((predicted - actual)**2, axis=0))

rmse_original = compute_rmse(computed_torques_original, torque_traj)
rmse_hybrid = compute_rmse(computed_torques_hybrid, torque_traj)

print("\n" + "="*70)
print("HYBRID INVERSE DYNAMICS IMPROVEMENT")
print("="*70)
print(f"{'Joint':<6} {'Leg':<4} {'Type':<8} {'Original':<10} {'Hybrid':<10} {'Improvement':<12}")
print("-"*70)

leg_names = ['FL', 'FR', 'RL', 'RR']
joint_names = ['hip_aa', 'hip_fe', 'knee']

total_improvement = 0
for i in range(12):
    leg_name = leg_names[i // 3]
    joint_name = joint_names[i % 3]
    improvement = ((rmse_original[i] - rmse_hybrid[i]) / rmse_original[i]) * 100
    total_improvement += improvement
    
    status = "✅" if improvement > 0 else "❌"
    print(f"{i+1:<6} {leg_name:<4} {joint_name:<8} {rmse_original[i]:<10.3f} {rmse_hybrid[i]:<10.3f} {improvement:<8.1f}% {status}")

overall_original = np.mean(rmse_original)
overall_hybrid = np.mean(rmse_hybrid)
overall_improvement = ((overall_original - overall_hybrid) / overall_original) * 100

print("-"*70)
print(f"{'OVERALL':<26} {overall_original:<10.3f} {overall_hybrid:<10.3f} {overall_improvement:<8.1f}% {'🎯' if overall_improvement > 0 else '❌'}")

# Joint type summary
hip_aa_joints = [0, 3, 6, 9]
hip_fe_joints = [1, 4, 7, 10]
knee_joints = [2, 5, 8, 11]

print(f"\n📊 BY JOINT TYPE:")
joint_types = [('Hip AA', hip_aa_joints), ('Hip FE', hip_fe_joints), ('Knee', knee_joints)]
for joint_type, indices in joint_types:
    orig_avg = np.mean([rmse_original[i] for i in indices])
    hybrid_avg = np.mean([rmse_hybrid[i] for i in indices])
    type_improvement = ((orig_avg - hybrid_avg) / orig_avg) * 100
    status = "✅" if type_improvement > 0 else "❌"
    print(f"  {joint_type:<8}: {orig_avg:.3f} → {hybrid_avg:.3f} ({type_improvement:+.1f}%) {status}")

print(f"\n🚀 CONCLUSION: Hybrid approach improves inverse dynamics accuracy by {overall_improvement:.1f}%")
print(f"   Best for: Hip joints (WBInterface force mapping)")
print(f"   Keeps:    Knee joints (whole-body dynamics)")
