from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from gym_quadruped.quadruped_env import QuadrupedEnv

from .conversion import mpc_to_sim
from .logging import color_print


def render_planned_trajectory(
    state_traj: np.ndarray, joint_vel_traj: np.ndarray, env: QuadrupedEnv
) -> list[np.ndarray]:
    """
    Renders the planned trajectory.
    Args:
        state_traj: (N, 12)
        joint_vel_traj: (N, 12)
        env: gym.Env
    """

    images = []
    for i in range(state_traj.shape[0]):
        state = state_traj[i]
        joint_vel = (
            joint_vel_traj[i] if i < joint_vel_traj.shape[0] else joint_vel_traj[-1]
        )
        qpos, qvel = mpc_to_sim(state, joint_vel)
        env.reset(qpos=qpos, qvel=qvel)
        images.append(env.render(mode="rgb_array"))
    return images


def render_and_save_planned_trajectory(
    state_traj: np.ndarray,
    input_traj: np.ndarray,
    env: QuadrupedEnv,
    suffix: str = "",
    fps: float | None = None,
) -> list[np.ndarray] | None:
    """
    Render planned trajectory and save as video if rendering is enabled.

    Args:
        state_traj: State trajectory from MPC
        input_traj: Input trajectory (joint velocities + forces)
        env: Quadruped simulation environment
        suffix: File suffix for saving
        fps: Frames per second for video (if None, uses MPC dt)

    Returns:
        List of rendered images if rendering is enabled, None otherwise
    """
    import imageio

    import config

    if not config.experiment.render:
        return None

    print("Rendering planned trajectory...")
    joint_vel_traj = input_traj[:, :12]  # Extract joint velocities
    planned_traj_images = render_planned_trajectory(state_traj, joint_vel_traj, env)

    if fps is None:
        fps = 1 / config.mpc_config.mpc_dt

    imageio.mimsave(
        f"results/planned_traj{suffix}.mp4",
        planned_traj_images,
        fps=fps,
    )
    return planned_traj_images


# Compute Mean Squared Error for all available quantities using aligned timesteps
def compute_aligned_mse(
    mpc_time_arr: np.ndarray,
    sim_time_arr: np.ndarray,
    mpc_arr: np.ndarray,
    sim_arr: np.ndarray,
) -> float | None:
    """
    Compute MSE between MPC and simulation arrays by comparing only aligned timesteps.

    Alignment strategy: for each MPC time t, pick the nearest simulation index via rounding
    assuming uniform sampling. A small tolerance guards against floating point drift.
    """
    if mpc_arr.ndim == 1:
        mpc_arr = mpc_arr[:, None]
    if sim_arr.ndim == 1:
        sim_arr = sim_arr[:, None]

    if len(mpc_time_arr) == 0 or len(sim_time_arr) < 2:
        return None

    # Estimate dt from provided time arrays to avoid reliance on function args
    sim_dt_local = (
        float(np.median(np.diff(sim_time_arr))) if len(sim_time_arr) > 1 else 0.0
    )
    if sim_dt_local <= 0.0:
        return None

    aligned_mpc = []
    aligned_sim = []
    # Tolerance scales slightly with time to accommodate accumulated fp error
    for i, t in enumerate(mpc_time_arr):
        j = int(round(t / sim_dt_local))
        if 0 <= j < len(sim_time_arr):
            if abs(sim_time_arr[j] - t) <= max(1e-9, 1e-6 * max(1.0, t)):
                aligned_mpc.append(mpc_arr[i])
                aligned_sim.append(sim_arr[j])

    if not aligned_mpc:
        return None

    mpc_stack = np.vstack(aligned_mpc)
    sim_stack = np.vstack(aligned_sim)

    num_cols = min(mpc_stack.shape[1], sim_stack.shape[1])
    if num_cols == 0:
        return None

    diff = mpc_stack[:, :num_cols] - sim_stack[:, :num_cols]
    return float(np.sqrt(np.mean(diff**2)))


def plot_trajectory_comparison(
    mpc_x_traj: np.ndarray,
    mpc_u_traj: np.ndarray,
    sim_qpos_traj: np.ndarray,
    sim_qvel_traj: np.ndarray,
    sim_grf_traj: np.ndarray,
    quantities: list[str] | None = None,
    mpc_dt: float = 0.01,
    sim_dt: float = 0.001,
    save_path: str | None = None,
    show_plot: bool = True,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """
    Plot comparison between planned (MPC) and simulated trajectories.

    Args:
        mpc_x_traj: MPC state trajectory (N_mpc, 30) - [base_pos(3), base_vel(3), base_eul(3), base_ang(3), joints(12), integrals(6)]
        mpc_u_traj: MPC input trajectory (N_mpc, 24) - [joint_vel(12), grf(12)]
        sim_qpos_traj: Simulated position trajectory (N_sim, 19) - [base_pos(3), base_quat(4), joints(12)]
        sim_qvel_traj: Simulated velocity trajectory (N_sim, 18) - [base_lin_vel(3), base_ang_vel(3), joint_vel(12)]
        quantities: List of quantities to plot. Options:
            - 'base_position': Base position (x, y, z)
            - 'base_orientation': Base orientation (roll, pitch, yaw)
            - 'base_linear_velocity': Base linear velocity
            - 'base_angular_velocity': Base angular velocity
            - 'joint_positions': Joint positions
            - 'joint_velocities': Joint velocities
            - 'ground_reaction_forces': Ground reaction forces
            - 'all': Plot all available quantities
        mpc_dt: MPC time step duration
        sim_dt: Simulation time step duration
        save_path: Path to save the plot (if None, doesn't save)
        show_plot: Whether to display the plot
        figsize: Figure size tuple
    """
    from .conversion import (
        MPC_U_QVEL_JOINTS,
        MPC_X_BASE_ANG,
        MPC_X_BASE_EUL,
        MPC_X_BASE_POS,
        MPC_X_BASE_VEL,
        MPC_X_Q_JOINTS,
        euler_to_quaternion,
        quaternion_to_euler,
    )

    if quantities is None:
        quantities = ["base_position", "base_orientation", "joint_positions"]

    # Create time axes for both trajectories
    mpc_time = np.arange(mpc_x_traj.shape[0]) * mpc_dt
    sim_time = np.arange(sim_qpos_traj.shape[0]) * sim_dt
    # Inputs (u) are typically defined for N-1 intervals when there are N states
    # Use a dedicated time vector for inputs to avoid length mismatch in plots
    mpc_u_time = np.arange(mpc_u_traj.shape[0]) * mpc_dt

    # Convert MPC trajectory to simulation format for comparison
    mpc_qpos_list = []
    mpc_qvel_list = []

    for i in range(mpc_x_traj.shape[0]):
        mpc_x = mpc_x_traj[i]
        # Use the last control input for the final state if needed
        if i < mpc_u_traj.shape[0]:
            mpc_u = mpc_u_traj[i]
        else:
            mpc_u = mpc_u_traj[-1]  # Use the last control input

        # Convert MPC state to simulation format
        qpos = np.concatenate(
            [
                mpc_x[MPC_X_BASE_POS],  # base position
                euler_to_quaternion(mpc_x[MPC_X_BASE_EUL]),  # base quaternion
                mpc_x[MPC_X_Q_JOINTS],  # joint positions
            ]
        )

        qvel = np.concatenate(
            [
                mpc_x[MPC_X_BASE_VEL],  # base linear velocity
                mpc_x[MPC_X_BASE_ANG],  # base angular velocity
                mpc_u[MPC_U_QVEL_JOINTS],  # joint velocities
            ]
        )

        mpc_qpos_list.append(qpos)
        mpc_qvel_list.append(qvel)

    mpc_qpos_traj = np.array(mpc_qpos_list)
    mpc_qvel_traj = np.array(mpc_qvel_list)

    # Convert quaternions to euler angles for both trajectories
    mpc_quat = mpc_qpos_traj[:, 3:7]
    mpc_euler = np.array([quaternion_to_euler(q) for q in mpc_quat])

    sim_quat = sim_qpos_traj[:, 3:7]
    sim_euler = np.array([quaternion_to_euler(q) for q in sim_quat])

    # Define available quantities
    available_quantities = {
        "base_position": {
            "mpc_data": mpc_qpos_traj[:, 0:3],
            "sim_data": sim_qpos_traj[:, 0:3],
            "labels": ["X Position (m)", "Y Position (m)", "Z Position (m)"],
            "title": "Base Position Comparison",
            "mpc_time": mpc_time,
            "sim_time": sim_time,
        },
        "base_orientation": {
            "mpc_data": mpc_euler,
            "sim_data": sim_euler,
            "labels": ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"],
            "title": "Base Orientation Comparison",
            "mpc_time": mpc_time,
            "sim_time": sim_time,
        },
        "base_linear_velocity": {
            "mpc_data": mpc_qvel_traj[:, 0:3],
            "sim_data": sim_qvel_traj[:, 0:3],
            "labels": ["Vx (m/s)", "Vy (m/s)", "Vz (m/s)"],
            "title": "Base Linear Velocity Comparison",
            "mpc_time": mpc_time,
            "sim_time": sim_time,
        },
        "base_angular_velocity": {
            "mpc_data": mpc_qvel_traj[:, 3:6],
            "sim_data": sim_qvel_traj[:, 3:6],
            "labels": ["Wx (rad/s)", "Wy (rad/s)", "Wz (rad/s)"],
            "title": "Base Angular Velocity Comparison",
            "mpc_time": mpc_time,
            "sim_time": sim_time,
        },
        "joint_positions": {
            "mpc_data": mpc_qpos_traj[:, 7:19],
            "sim_data": sim_qpos_traj[:, 7:19],
            "labels": [f"Joint {i + 1} (rad)" for i in range(12)],
            "title": "Joint Positions Comparison",
            "mpc_time": mpc_time,
            "sim_time": sim_time,
        },
        "joint_velocities": {
            "mpc_data": mpc_qvel_traj[:, 6:18],
            "sim_data": sim_qvel_traj[:, 6:18],
            "labels": [f"Joint {i + 1} Vel (rad/s)" for i in range(12)],
            "title": "Joint Velocities Comparison",
            "mpc_time": mpc_time,
            "sim_time": sim_time,
        },
        "ground_reaction_forces": {
            "mpc_data": mpc_u_traj[:, 12:24],
            "sim_data": sim_grf_traj,
            "labels": [f"Force {i + 1} (N)" for i in range(12)],
            "title": "Ground Reaction Forces Comparison",
            "mpc_time": mpc_u_time,
            "sim_time": sim_time,
        },
    }

    # Add per-limb GRF entries (FL, FR, RL, RR), each with Fx, Fy, Fz
    leg_names = ["FL", "FR", "RL", "RR"]
    leg_slices = [(0, 3), (3, 6), (6, 9), (9, 12)]
    for leg_index, (start, end) in enumerate(leg_slices):
        leg = leg_names[leg_index]
        available_quantities[f"ground_reaction_forces_{leg}"] = {
            "mpc_data": mpc_u_traj[:, 12 + start : 12 + end],
            "sim_data": sim_grf_traj[:, start:end],
            "labels": ["Fx (N)", "Fy (N)", "Fz (N)"],
            "title": f"Ground Reaction Forces - {leg}",
            "mpc_time": mpc_u_time,
            "sim_time": sim_time,
        }

    # Determine which quantities to plot
    if "all" in quantities:
        quantities_to_plot = list(available_quantities.keys())
    else:
        quantities_to_plot = [q for q in quantities if q in available_quantities]

    # If overall GRF was requested, replace it with per-limb GRF plots
    if "ground_reaction_forces" in quantities_to_plot:
        insert_index = quantities_to_plot.index("ground_reaction_forces")
        quantities_to_plot.pop(insert_index)
        quantities_to_plot[insert_index:insert_index] = [
            f"ground_reaction_forces_{leg}" for leg in leg_names
        ]

    if not quantities_to_plot:
        print("Warning: No valid quantities specified for plotting")
        return

    # Create subplots with 2 columns
    n_plots = len(quantities_to_plot)
    n_rows = (n_plots + 1) // 2  # Ceiling division to handle odd number of plots
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)

    # Convert axes to 2D array if n_rows == 1
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Flatten axes for easier indexing
    axes_flat = axes.flatten()

    for i, quantity in enumerate(quantities_to_plot):
        ax = axes_flat[i]
        meta = available_quantities[quantity]
        mpc_data = meta["mpc_data"]
        sim_data = meta["sim_data"]
        labels = meta["labels"]
        title = meta["title"]
        mpc_time_q = meta["mpc_time"]
        sim_time_q = meta["sim_time"]

        # Plot each component
        for j in range(mpc_data.shape[1]):
            ax.plot(
                mpc_time_q,
                mpc_data[:, j],
                label=f"Planned {labels[j]}",
                linewidth=2,
                linestyle="--",
                marker="^",
                alpha=0.8,
                color=f"C{j}",
            )
            ax.plot(
                sim_time_q,
                sim_data[:, j],
                label=f"Simulated {labels[j]}",
                linewidth=1.5,
                linestyle="-",
                color=f"C{j}",
            )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Value", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide empty subplots if odd number of quantities
    for i in range(n_plots, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    mse_results: dict[str, float] = {}
    for q_name, meta in available_quantities.items():
        mse_val = compute_aligned_mse(
            meta["mpc_time"],
            meta["sim_time"],
            meta["mpc_data"],
            meta["sim_data"],
        )
        if mse_val is not None and np.isfinite(mse_val):
            mse_results[q_name] = mse_val
    if mse_results:
        color_print(
            "orange",
            "RMSE between planned and simulated trajectories (aligned timesteps):",
        )
        # Calculate max width for name column for alignment
        col_width = max(len(q) for q in mse_results.keys())
        for q_name in sorted(mse_results.keys()):
            print(f" - {q_name:<{col_width}} : {mse_results[q_name]:.6g}")
