import imageio
import numpy as np
from gym_quadruped.quadruped_env import QuadrupedEnv
from tqdm import tqdm

import config
from examples.model import KinoDynamic_Model
from examples.mpc import HoppingMPC
from utils.inv_dyn import compute_joint_torques
from utils.visualization import render_planned_trajectory


def color_print(text):
    print("\033[1m\033[38;5;208m" + text + "\033[0m")


# ----------------------------------------------------------------------------------------------------------------
# STAGE 0: Create the model and the simulation environment
# ----------------------------------------------------------------------------------------------------------------
color_print("--- Stage 0: Creating the model and the simulation environment ---")

# create the model and MPC
kinodynamic_model = KinoDynamic_Model(config)
mpc = HoppingMPC(model=kinodynamic_model, config=config, build=False)

env = QuadrupedEnv(
    robot=config.robot,
    scene="flat",
    ground_friction_coeff=config.sim_params[
        "ground_friction_coeff"
    ],  # pass a float for a fixed value
    # state_obs_names=...,  # Desired quantities in the 'state', see ALL_OBS in quadruped_env.py
    sim_dt=config.sim_params["sim_dt"],
)

# ----------------------------------------------------------------------------------------------------------------
# STAGE 1: Trajectory Optimization using HoppingMPC
# ----------------------------------------------------------------------------------------------------------------
color_print("--- Stage 1: Solving Kinodynamic Trajectory Optimization ---")

# Set up the initial state and reference for the hopping motion
initial_state = {
    "position": config.initial_qpos[0:3],
    "linear_velocity": np.zeros(3),
    "orientation": np.zeros(3),
    "angular_velocity": np.zeros(3),
    "joint_FL": config.initial_qpos[7:10],
    "joint_FR": config.initial_qpos[10:13],
    "joint_RL": config.initial_qpos[13:16],
    "joint_RR": config.initial_qpos[16:19],
}

reference = {
    "ref_position": config.initial_qpos[0:3] + np.array([0.1, 0.0, 0.0]),
    "ref_linear_velocity": np.array([0.0, 0.0, 0.0]),
    "ref_orientation": np.zeros(3),
    "ref_angular_velocity": np.zeros(3),
    "ref_joints": config.initial_qpos[7:19],
}
ref = np.concatenate(
    (
        reference["ref_position"],
        reference["ref_linear_velocity"],
        reference["ref_orientation"],
        reference["ref_angular_velocity"],
        reference["ref_joints"],
        np.zeros(24),
    )
)

state_traj, grf_traj, joint_vel_traj, status = mpc.solve_trajectory(
    initial_state, ref, config.contact_sequence
)

if status != 0:
    print(f"Optimization failed with status: {status}")
    exit(1)

print("Optimization successful. Extracted trajectory of states and GRFs.")
np.save("results/state_traj.npy", state_traj)
np.save("results/joint_vel_traj.npy", joint_vel_traj)
np.save("results/grf_traj.npy", grf_traj)
np.save("results/contact_sequence.npy", config.contact_sequence)

# state_traj = np.load("results/state_traj.npy")
# joint_vel_traj = np.load("results/joint_vel_traj.npy")
# grf_traj = np.load("results/grf_traj.npy")
# contact_sequence = np.load("results/contact_sequence.npy")

print("Rendering planned trajectory...")
planned_traj_images = render_planned_trajectory(state_traj, joint_vel_traj, env)
imageio.mimsave("results/planned_traj.mp4", planned_traj_images, fps=1 / config.mpc_dt)

# ----------------------------------------------------------------------------------------------------------------
# STAGE 2: Inverse Dynamics to find Joint Torques
# ----------------------------------------------------------------------------------------------------------------
color_print("--- Stage 2: Computing Joint Torques via Inverse Dynamics ---")

joint_torques_traj = compute_joint_torques(
    kinodynamic_model, state_traj, grf_traj, config.contact_sequence, config.mpc_dt
)
np.save("results/joint_torques_traj.npy", joint_torques_traj)

# ----------------------------------------------------------------------------------------------------------------
# STAGE 3: Simulate the trajectory
# ----------------------------------------------------------------------------------------------------------------
color_print("--- Stage 3: Simulating the trajectory ---")
# simulate the trajectory
env.reset(qpos=config.initial_qpos, qvel=config.initial_qvel)
images = []
action_index = 0
for i in tqdm(range(int(config.duration / config.sim_dt))):
    # Step forward in the simulation
    action = joint_torques_traj[action_index, :]
    if (i + 1) % int(config.mpc_dt / config.sim_dt) == 0:
        action_index += 1
    state, reward, is_terminated, is_truncated, info = env.step(action=action)

    # Render the environment into an image
    image = env.render(mode="rgb_array", tint_robot=True)
    overplotted_image = np.uint8(0.7 * image + 0.3 * planned_traj_images[action_index])
    images.append(overplotted_image)

fps = 1 / config.sim_dt
imageio.mimsave("results/trajectory.mp4", images, fps=fps)
env.close()
