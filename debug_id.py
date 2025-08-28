import time

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from gym_quadruped import quadruped_env
from tqdm import tqdm

import config
from examples.model import KinoDynamic_Model
from utils.inv_dyn import compute_joint_torques

robot_name = "go2"  # 'b2', 'go1', 'go2', 'hyqreal', 'mini_cheetah', 'aliengo'
terrain_type = "flat"  # 'flat', 'perlin'
sim_dt = 0.01  # seconds
sim_duration = 2.0  # seconds
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
    ref_base_lin_vel=(0.5, 1.0),  # pass a float for a fixed value
    ground_friction_coeff=config.mu_friction,  # pass a float for a fixed value
    base_vel_command_type="forward",  # "forward", "random", "forward+rotate", "human"
    state_obs_names=state_obs_names,  # Desired quantities in the 'state', see ALL_OBS in quadruped_env.py
    sim_dt=sim_dt,
)

initial_qpos = np.zeros(19)
initial_qpos[0:3] = [0.0, 0.0, 0.23]  # base position
initial_qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # base orientation (quaternion)
initial_qpos[7:19] = [  # joint angles
    0.0,
    1.0,
    -2.1,  # FL: no abd, folded hip/knee
    0.0,
    1.0,
    -2.1,  # FR
    0.0,
    1.0,
    -2.1,  # RL
    0.0,
    1.0,
    -2.1,  # RR
]
initial_qvel = np.zeros(18)
state = env.reset(qpos=initial_qpos, qvel=initial_qvel)

################################################################################
# Print out the action and observation spaces
################################################################################
if False:
    print("-" * 20, "Observation space", "-" * 20)
    for obs_name, obs_val in state.items():
        print(f"{obs_name:20s}--\t{obs_val.shape}")

    print("-" * 20, "Action space", "-" * 25)
    print(env.action_space)

################################################################################
# Simulate and render
################################################################################
print("-" * 20, "Simulating and rendering", "-" * 13)
state_traj, torque_traj = [state], []
images = []
sim_times, render_times = [], []
action_index = 0
for i in tqdm(range(int(sim_duration / sim_dt))):
    import imageio

    # Step forward in the simulation
    start_time = time.time()
    # action = env.action_space.sample() * 0  # Sample random action
    action = np.ones(12) * 1.1
    state, reward, is_terminated, is_truncated, info = env.step(action=action)
    state_traj.append(state)
    torque_traj.append(action)
    sim_times.append(time.time() - start_time)

    # Render the environment into an image
    if if_render:
        render_start_time = time.time()
        image = env.render(mode="rgb_array", tint_robot=True)
        render_times.append(time.time() - render_start_time)
    else:
        image = None
        render_times.append(0)

    images.append(image)

if if_render:
    fps = 1 / sim_dt
    imageio.mimsave("results/env_test.mp4", images, fps=fps)
env.close()

print("-" * 20, "Performance", "-" * 20)
print(
    f"Sim time: {np.mean(sim_times):.3f} s, Render time: {np.mean(render_times):.3f} s"
)

################################################################################
# Convert the simulated state to MPC format
################################################################################
qpos_traj = np.array([state["qpos"] for state in state_traj])
qvel_traj = np.array([state["qvel"] for state in state_traj])
base_ori_euler_xyz_traj = np.array(
    [state["base_ori_euler_xyz"] for state in state_traj]
)
contact_state_traj = np.array([state["contact_state"] for state in state_traj])
contact_forces_traj = np.array([state["contact_forces"] for state in state_traj])

mpc_state_traj = np.concatenate(
    (
        qpos_traj[:, 0:3],
        qvel_traj[:, 0:3],
        base_ori_euler_xyz_traj,
        qvel_traj[:, 3:6],
        qpos_traj[:, 7:19],
        np.zeros((qpos_traj.shape[0], 6)),  # zero integral states
    ),
    axis=1,
)

mpc_input_traj = np.concatenate(
    (
        qvel_traj[:, 6:18],
        contact_forces_traj,
    ),
    axis=1,
)

################################################################################
# Test forward dynamics
################################################################################
kinodynamic_model = KinoDynamic_Model(config)
kinodynamic_model.export_robot_model()

param = np.concatenate(
    (
        np.ones(4),  # contact state
        np.array([config.mu_friction]),  # friction
        np.array([0, 0, 0, 0]),  # stance proximity
        np.array([0, 0, 0]),  # base position
        np.array([0]),  # base yaw
        np.array([0, 0, 0, 0, 0, 0]),  # external wrench
        config.inertia.flatten(),  # inertia
        np.array([config.mass]),  # mass
    ),
)

model_states_traj = np.zeros((mpc_state_traj.shape[0], 30))
model_states_traj[0] = mpc_state_traj[0]
for i in range(1, mpc_state_traj.shape[0]):
    param[0:4] = contact_state_traj[i]
    param[9:12] = mpc_state_traj[i][0:3]
    param[12] = mpc_state_traj[i][9]

    xdot = kinodynamic_model.forward_dynamics(
        model_states_traj[i][:, None], mpc_input_traj[i][:, None], param[:, None]
    )
    model_states_traj[i] = model_states_traj[i] + sim_dt * cs.DM(xdot).toarray()[:, 0]
model_states_traj = np.array(model_states_traj)

for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.plot(model_states_traj[:, i], label="fd")
    plt.plot(mpc_state_traj[:, i], label="sim")
    plt.legend()
plt.savefig("results/fd_debug.png")

################################################################################
# Test inverse dynamics
################################################################################
torque_traj = np.array(torque_traj)
computed_torques = compute_joint_torques(
    kinodynamic_model,
    mpc_state_traj,
    mpc_input_traj[:-1],
    contact_state_traj[:-1].T,
    sim_dt,
)

for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.plot(computed_torques[:, i], label="computed")
    plt.plot(torque_traj[:, i], label="true")
    plt.legend()
plt.savefig("results/id_debug.png")
