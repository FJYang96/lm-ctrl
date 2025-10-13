import argparse

import imageio
import numpy as np
from gym_quadruped.quadruped_env import QuadrupedEnv
from tqdm import tqdm

import config
from mpc.dynamics.model import KinoDynamic_Model
from mpc.mpc_opti import QuadrupedMPCOpti
from utils.inv_dyn import compute_joint_torques
from utils.visualization import render_planned_trajectory


def print_orange(text):
    print("\033[1m\033[38;5;208m" + text + "\033[0m")


def print_red(text):
    print("\033[1m\033[38;5;196m" + text + "\033[0m")


def print_green(text):
    print("\033[1m\033[38;5;46m" + text + "\033[0m")


def main():
    parser = argparse.ArgumentParser(description="Quadruped Hopping MPC")
    parser.add_argument(
        "--solver",
        choices=["acados", "opti"],
        default="opti",
        help="Choose solver: acados (original) or opti (new)",
    )
    args = parser.parse_args()

    # ----------------------------------------------------------------------------------------------------------------
    # STAGE 0: Create the model and the simulation environment
    # ----------------------------------------------------------------------------------------------------------------
    print_orange("--- Stage 0: Creating the model and the simulation environment ---")
    print(f"Using solver: {args.solver}")

    # Create the model
    kinodynamic_model = KinoDynamic_Model(config)

    mpc = QuadrupedMPCOpti(model=kinodynamic_model, config=config, build=True)
    suffix = "_opti"  # Add suffix for opti files

    env = QuadrupedEnv(
        robot=config.robot,
        scene="flat",
        ground_friction_coeff=config.sim_params[
            "ground_friction_coeff"
        ],  # pass a float for a fixed value
        sim_dt=config.sim_params["sim_dt"],
    )

    # ----------------------------------------------------------------------------------------------------------------
    # STAGE 1: Trajectory Optimization
    # ----------------------------------------------------------------------------------------------------------------
    print_orange(
        f"--- Stage 1: Solving Kinodynamic Trajectory Optimization with {args.solver.upper()} ---"
    )

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

    target_jump_height = 0.15  # Target 15cm jump height
    reference = {
        "ref_position": config.initial_qpos[0:3]
        + np.array([0.1, 0.0, target_jump_height]),
        "ref_linear_velocity": np.array([0.0, 0.0, 0.0]),
        "ref_orientation": np.zeros(3),
        "ref_angular_velocity": np.zeros(3),
        "ref_joints": config.initial_qpos[7:19],
    }

    if args.solver == "acados":
        # Original Acados format
        ref = np.concatenate(
            [
                reference["ref_position"],
                reference["ref_linear_velocity"],
                reference["ref_orientation"],
                reference["ref_angular_velocity"],
                reference["ref_joints"],
                np.zeros(24),  # Reference for inputs
            ]
        )
    else:
        # Opti format with integral states
        ref = np.concatenate(
            [
                reference["ref_position"],
                reference["ref_linear_velocity"],
                reference["ref_orientation"],
                reference["ref_angular_velocity"],
                reference["ref_joints"],
                np.zeros(6),  # Reference for integral states
                np.zeros(24),  # Reference for inputs (joint velocities + forces)
            ]
        )

    state_traj, grf_traj, joint_vel_traj, status = mpc.solve_trajectory(
        initial_state, ref, config.contact_sequence
    )

    if status != 0:
        print_red(f"Optimization failed with status: {status}")
    else:
        print_green("Optimization successful. Extracted trajectory of states and GRFs.")

    # Save results with appropriate suffix
    np.save(f"results/state_traj{suffix}.npy", state_traj)
    np.save(f"results/joint_vel_traj{suffix}.npy", joint_vel_traj)
    np.save(f"results/grf_traj{suffix}.npy", grf_traj)
    np.save("results/contact_sequence.npy", config.contact_sequence)

    print("Rendering planned trajectory...")
    planned_traj_images = render_planned_trajectory(state_traj, joint_vel_traj, env)
    imageio.mimsave(
        f"results/planned_traj{suffix}.mp4", planned_traj_images, fps=1 / config.mpc_dt
    )

    # ----------------------------------------------------------------------------------------------------------------
    # STAGE 2: Inverse Dynamics to find Joint Torques
    # ----------------------------------------------------------------------------------------------------------------
    print_orange("--- Stage 2: Computing Joint Torques via Inverse Dynamics ---")

    joint_torques_traj = compute_joint_torques(
        kinodynamic_model, state_traj, grf_traj, config.contact_sequence, config.mpc_dt
    )
    np.save(f"results/joint_torques_traj{suffix}.npy", joint_torques_traj)

    # ----------------------------------------------------------------------------------------------------------------
    # STAGE 3: Simulate the trajectory
    # ----------------------------------------------------------------------------------------------------------------
    print_orange("--- Stage 3: Simulating the trajectory ---")
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
        overplotted_image = np.uint8(
            0.7 * image + 0.3 * planned_traj_images[action_index]
        )
        images.append(overplotted_image)

    fps = 1 / config.sim_dt
    imageio.mimsave(f"results/trajectory{suffix}.mp4", images, fps=fps)
    env.close()

    print_green(
        f"✅ Complete! Generated files with suffix '{suffix}' in results/ directory"
    )


if __name__ == "__main__":
    main()
