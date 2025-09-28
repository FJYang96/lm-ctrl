import numpy as np
import matplotlib.pyplot as plt
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

################################################################################
# Create environment
################################################################################
print(f"Testing robot {robot_name} on terrain {terrain_type}")

env = quadruped_env.QuadrupedEnv(
    robot=robot_name,
    scene=terrain_type,
    ref_base_lin_vel=(0.0, 0.0),
    ground_friction_coeff=config.mu_friction,
    base_vel_command_type="forward",
    state_obs_names=quadruped_env.QuadrupedEnv._DEFAULT_OBS + (
        "base_ori_euler_xyz", "contact_state", "contact_forces"
    ),
    sim_dt=sim_dt,
)

# Initial pose
initial_qpos = np.zeros(19)
initial_qpos[0:3] = [0.0, 0.0, 0.23]
initial_qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
initial_qpos[7:19] = [0.0, 0.8, -1.6] * 4
state = env.reset(qpos=initial_qpos, qvel=np.zeros(18))

################################################################################
# Generate realistic motion with PD control
################################################################################
print("Generating realistic motion...")

Kp = np.array([20, 40, 40] * 4)
Kd = np.array([2, 4, 4] * 4)

state_traj, torque_traj = [state], []
num_steps = int(sim_duration / sim_dt)
time_vec = np.linspace(0, sim_duration, num_steps)

# Reference trajectory
ref_positions = np.zeros((num_steps, 12))
for i in range(12):
    if i % 3 == 0:  # Hip abduction
        amplitude = 0.05
    elif i % 3 == 1:  # Hip flexion
        amplitude = 0.2
    else:  # Knee
        amplitude = 0.3
    ref_positions[:, i] = initial_qpos[7 + i] + amplitude * np.sin(2*np.pi*0.5*time_vec + i*np.pi/6)

for i in tqdm(range(num_steps)):
    current_pos = state["qpos"][7:19]
    current_vel = state["qvel"][6:18]
    
    # PD control
    action = Kp * (ref_positions[i] - current_pos) + Kd * (-current_vel)
    action = np.clip(action, -20, 20)
    
    state, _, _, _, _ = env.step(action=action)
    state_traj.append(state)
    torque_traj.append(action)

env.close()

################################################################################
# Convert to arrays
################################################################################
qpos_traj = np.array([s["qpos"] for s in state_traj])
qvel_traj = np.array([s["qvel"] for s in state_traj])
base_ori_euler_xyz_traj = np.array([s["base_ori_euler_xyz"] for s in state_traj])
contact_state_traj = np.array([s["contact_state"] for s in state_traj])
contact_forces_traj = np.array([s["contact_forces"] for s in state_traj])

################################################################################
# WBInterface-style inverse dynamics (based on actual WBInterface code)
################################################################################
def compute_wbinterface_inverse_dynamics(
    qpos_traj, qvel_traj, contact_forces_traj, contact_state_traj, 
    feet_jac_traj, legs_qfrc_passive_traj, dt
):
    """
    Implements inverse dynamics following WBInterface logic:
    - For STANCE legs: τ = -J^T * F
    - For SWING legs: Different control (simplified here)
    - Includes friction compensation like WBInterface
    
    Based on WBInterface.compute_stance_and_swing_torque()
    """
    num_steps = qpos_traj.shape[0] - 1
    joint_torques_traj = np.zeros((num_steps, 12))
    
    legs_order = ['FL', 'FR', 'RL', 'RR']
    
    for step in range(num_steps):
        # Current contact state
        current_contact = contact_state_traj[step]
        
        # Process each leg following WBInterface logic
        for leg_id, leg_name in enumerate(legs_order):
            joint_start = leg_id * 3
            force_start = leg_id * 3
            
            # Get forces for this leg
            nmpc_GRF = contact_forces_traj[step, force_start:force_start + 3]
            
            if current_contact[leg_id] == 1:  # STANCE phase (like WBInterface)
                # WBInterface: tau.FL = -np.matmul(feet_jac.FL.T, nmpc_GRFs.FL)
                # We use simplified Jacobian here
                J = compute_foot_jacobian_for_leg(
                    qpos_traj[step, 7 + joint_start:7 + joint_start + 3],
                    leg_id
                )
                tau_leg = -J.T @ nmpc_GRF
                
            else:  # SWING phase
                # WBInterface uses complex swing controller
                # We'll use simplified version with gravity compensation
                joint_vel = qvel_traj[step, 6 + joint_start:6 + joint_start + 3]
                tau_leg = compute_swing_torque_simplified(
                    qpos_traj[step, 7 + joint_start:7 + joint_start + 3],
                    joint_vel,
                    qpos_traj[step, 3:7]  # Base orientation for gravity
                )
            
            # Store torques
            joint_torques_traj[step, joint_start:joint_start + 3] = tau_leg
            
            # Friction compensation (like WBInterface)
            if feet_jac_traj is not None and legs_qfrc_passive_traj is not None:
                # WBInterface: tau.FL -= legs_qfrc_passive.FL
                passive_forces = legs_qfrc_passive_traj[step, joint_start:joint_start + 3]
                joint_torques_traj[step, joint_start:joint_start + 3] -= passive_forces * 0.5
    
    return joint_torques_traj

def compute_foot_jacobian_for_leg(q_leg, leg_id):
    """
    Compute foot Jacobian for a specific leg
    Following proper kinematics (no arbitrary scaling)
    """
    hip_aa, hip_fe, knee = q_leg
    
    L1 = 0.213  # Upper leg
    L2 = 0.213  # Lower leg
    
    # Leg configuration
    sign_y = 1 if leg_id in [0, 2] else -1  # FL, RL vs FR, RR
    hip_offset_y = 0.055 * sign_y
    
    c_aa, s_aa = np.cos(hip_aa), np.sin(hip_aa)
    c_fe, s_fe = np.cos(hip_fe), np.sin(hip_fe)
    c_fk = np.cos(hip_fe + knee)
    s_fk = np.sin(hip_fe + knee)
    
    J = np.zeros((3, 3))
    
    # Proper kinematics (no arbitrary factors!)
    J[0, 0] = 0
    J[0, 1] = -L1 * s_fe - L2 * s_fk
    J[0, 2] = -L2 * s_fk
    
    J[1, 0] = hip_offset_y * c_aa - (L1 * c_fe + L2 * c_fk) * s_aa
    J[1, 1] = (L1 * s_fe + L2 * s_fk) * c_aa
    J[1, 2] = L2 * s_fk * c_aa
    
    J[2, 0] = hip_offset_y * s_aa + (L1 * c_fe + L2 * c_fk) * c_aa
    J[2, 1] = -(L1 * c_fe + L2 * c_fk) * s_aa
    J[2, 2] = -L2 * c_fk * s_aa
    
    return J

def compute_swing_torque_simplified(q_leg, qd_leg, base_quat):
    """
    Simplified swing torque computation
    (WBInterface uses full Cartesian space control)
    """
    # Get gravity direction in body frame
    try:
        base_rot = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]]).as_matrix()
    except:
        base_rot = np.eye(3)
    
    g_world = np.array([0, 0, -9.81])
    g_body = base_rot.T @ g_world
    
    # Simple gravity compensation
    m_upper, m_lower = 0.695, 0.166
    L1, L2 = 0.213, 0.213
    
    hip_aa, hip_fe, knee = q_leg
    
    tau = np.zeros(3)
    tau[0] = -0.5 * g_body[1] * np.sin(hip_fe) * np.cos(hip_aa)
    tau[1] = -(m_upper * L1/2 * np.cos(hip_fe) + m_lower * L1 * np.cos(hip_fe)) * g_body[2]
    tau[2] = -m_lower * L2/2 * np.cos(hip_fe + knee) * g_body[2]
    
    # Add damping
    tau -= 0.1 * qd_leg
    
    return tau

################################################################################
# Compute inverse dynamics using both methods
################################################################################
print("Computing inverse dynamics...")

# 1. Standard whole-body method
mpc_state_traj = np.concatenate((
    qpos_traj[:, 0:3], qvel_traj[:, 0:3], base_ori_euler_xyz_traj,
    qvel_traj[:, 3:6], qpos_traj[:, 7:19], np.zeros((qpos_traj.shape[0], 6))
), axis=1)

mpc_input_traj = np.concatenate((qvel_traj[:, 6:18], contact_forces_traj), axis=1)

kinodynamic_model = KinoDynamic_Model(config)
kinodynamic_model.export_robot_model()

predicted_wholebody = compute_joint_torques(
    kinodynamic_model, mpc_state_traj, mpc_input_traj[:-1], 
    contact_state_traj[:-1].T, sim_dt
)

# 2. WBInterface-style method
# Note: We're passing None for Jacobians and passive forces for simplicity
# In real implementation, you'd compute these from the simulator
predicted_wbinterface = compute_wbinterface_inverse_dynamics(
    qpos_traj, qvel_traj, contact_forces_traj, contact_state_traj,
    None, None, sim_dt
)

true_torques = np.array(torque_traj)

################################################################################
# Plot comparison: True vs Both Predictions
################################################################################
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
skip = 20  # Skip initial transient

leg_names = ['FL', 'FR', 'RL', 'RR']
joint_names = ['Hip AA', 'Hip FE', 'Knee']

for i in range(12):
    ax = axes[i // 4, i % 4]
    
    # Plot lines
    ax.plot(true_torques[skip:, i], label='True (PD Commanded)', 
            color='black', linewidth=2.5, alpha=0.9)
    ax.plot(predicted_wholebody[skip:, i], label='Whole-body ID', 
            color='blue', linewidth=1.5, alpha=0.7)
    ax.plot(predicted_wbinterface[skip:, i], label='WBInterface-style', 
            color='red', linewidth=1.5, alpha=0.7, linestyle='--')
    
    # Mark stance/swing phases
    for t in range(skip, len(contact_state_traj)-1):
        if contact_state_traj[t, i//3] == 0:  # Swing
            ax.axvspan(t-skip, t-skip+1, alpha=0.05, color='orange')
    
    # Formatting
    leg = leg_names[i // 3]
    joint = joint_names[i % 3]
    ax.set_title(f'{leg} {joint}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Torque (Nm)')
    ax.grid(True, alpha=0.3)
    
    # Add RMSE for both methods
    rmse_wb = np.sqrt(np.mean((predicted_wholebody[skip:, i] - true_torques[skip:, i])**2))
    rmse_wbi = np.sqrt(np.mean((predicted_wbinterface[skip:, i] - true_torques[skip:, i])**2))
    
    ax.text(0.02, 0.98, f'RMSE WB: {rmse_wb:.2f}\nRMSE WBI: {rmse_wbi:.2f}', 
            transform=ax.transAxes, fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if i == 0:
        ax.legend(loc='upper right', fontsize=8)

plt.suptitle('True vs Predicted: Whole-body vs WBInterface-style Inverse Dynamics', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/wbinterface_comparison.png', dpi=150)
plt.show()

################################################################################
# Print comparison statistics
################################################################################
print("\n" + "="*80)
print("INVERSE DYNAMICS COMPARISON: WHOLE-BODY vs WBINTERFACE-STYLE")
print("="*80)
print(f"{'Joint':<15} {'WB RMSE':<12} {'WBI RMSE':<12} {'Better':<10} {'Improvement':<12}")
print("-"*80)

for i in range(12):
    leg = leg_names[i // 3]
    joint = joint_names[i % 3]
    
    rmse_wb = np.sqrt(np.mean((predicted_wholebody[skip:, i] - true_torques[skip:, i])**2))
    rmse_wbi = np.sqrt(np.mean((predicted_wbinterface[skip:, i] - true_torques[skip:, i])**2))
    
    better = "WB" if rmse_wb < rmse_wbi else "WBI"
    improvement = ((rmse_wb - rmse_wbi) / rmse_wb) * 100 if rmse_wb > 0 else 0
    
    print(f"{leg} {joint:<10} {rmse_wb:<12.3f} {rmse_wbi:<12.3f} {better:<10} {improvement:+.1f}%")

# Overall statistics
overall_wb = np.sqrt(np.mean((predicted_wholebody[skip:] - true_torques[skip:])**2))
overall_wbi = np.sqrt(np.mean((predicted_wbinterface[skip:] - true_torques[skip:])**2))

print("-"*80)
print(f"{'OVERALL':<15} {overall_wb:<12.3f} {overall_wbi:<12.3f}", end="")
print(f" {'WB' if overall_wb < overall_wbi else 'WBI':<10}", end="")
print(f" {((overall_wb - overall_wbi) / overall_wb) * 100:+.1f}%")

print(f"\n✅ Winner: {'Whole-body' if overall_wb < overall_wbi else 'WBInterface-style'}")
print(f"   by {abs(overall_wb - overall_wbi):.3f} Nm average RMSE")

# Stance vs Swing analysis
print("\n📊 STANCE vs SWING Performance:")
stance_mask = contact_state_traj[skip:-1].repeat(3, axis=1).flatten() == 1
swing_mask = ~stance_mask

stance_err_wb = np.sqrt(np.mean((predicted_wholebody[skip:].flatten()[stance_mask] - 
                                 true_torques[skip:].flatten()[stance_mask])**2))
swing_err_wb = np.sqrt(np.mean((predicted_wholebody[skip:].flatten()[swing_mask] - 
                                true_torques[skip:].flatten()[swing_mask])**2))

stance_err_wbi = np.sqrt(np.mean((predicted_wbinterface[skip:].flatten()[stance_mask] - 
                                  true_torques[skip:].flatten()[stance_mask])**2))
swing_err_wbi = np.sqrt(np.mean((predicted_wbinterface[skip:].flatten()[swing_mask] - 
                                 true_torques[skip:].flatten()[swing_mask])**2))

print(f"  Stance - WB: {stance_err_wb:.3f}, WBI: {stance_err_wbi:.3f}")
print(f"  Swing  - WB: {swing_err_wb:.3f}, WBI: {swing_err_wbi:.3f}")