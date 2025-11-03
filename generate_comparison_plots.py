#!/usr/bin/env python3
"""
Generate comparison plots between original and improved inverse dynamics implementations
Similar to the style of plots in results/ folder
"""

import argparse
import os
import time
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import config
from mpc.dynamics.model import KinoDynamic_Model
from mpc.mpc_opti import QuadrupedMPCOpti
from utils.inv_dyn import compute_joint_torques, compute_joint_torques_improved


def run_trajectory_optimization() -> (
    Optional[Tuple[Any, np.ndarray, np.ndarray, np.ndarray]]
):
    """Run trajectory optimization and return results"""
    print("Setting up kinodynamic model and MPC...")
    kinodynamic_model = KinoDynamic_Model(config)
    mpc = QuadrupedMPCOpti(model=kinodynamic_model, config=config, build=True)

    # Set up initial state and reference (same as main.py)
    initial_state = {
        "position": config.experiment.initial_qpos[0:3],
        "linear_velocity": np.zeros(3),
        "orientation": np.zeros(3),
        "angular_velocity": np.zeros(3),
        "joint_FL": config.experiment.initial_qpos[7:10],
        "joint_FR": config.experiment.initial_qpos[10:13],
        "joint_RL": config.experiment.initial_qpos[13:16],
        "joint_RR": config.experiment.initial_qpos[16:19],
    }

    reference = {
        "ref_position": config.experiment.initial_qpos[0:3] + np.array([0.1, 0.0, 0.0]),
        "ref_linear_velocity": np.array([0.0, 0.0, 0.0]),
        "ref_orientation": np.zeros(3),
        "ref_angular_velocity": np.zeros(3),
        "ref_joints": config.experiment.initial_qpos[7:19],
    }

    ref = np.concatenate(
        [
            reference["ref_position"],
            reference["ref_linear_velocity"],
            reference["ref_orientation"],
            reference["ref_angular_velocity"],
            reference["ref_joints"],
            np.zeros(6),  # Reference for integral states
            np.zeros(24),  # Reference for inputs
        ]
    )

    print("Solving trajectory optimization...")
    state_traj, grf_traj, joint_vel_traj, status = mpc.solve_trajectory(
        initial_state, ref, config.mpc_config.contact_sequence
    )

    if status != 0:
        print(f"Optimization failed with status: {status}")
        return None

    print("Optimization successful!")
    return kinodynamic_model, state_traj, grf_traj, joint_vel_traj


def compute_both_implementations(
    kinodynamic_model: Any,
    state_traj: np.ndarray,
    grf_traj: np.ndarray,
    joint_vel_traj: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute joint torques using both implementations"""
    print("\nComputing joint torques with both implementations...")

    # Original implementation
    print("Running original implementation...")
    start_time = time.time()
    torques_original = compute_joint_torques(
        kinodynamic_model,
        state_traj,
        grf_traj,
        config.mpc_config.contact_sequence,
        config.mpc_config.mpc_dt,
    )
    time_original = time.time() - start_time

    # Improved implementation
    print("Running improved implementation...")
    input_traj = np.concatenate([joint_vel_traj, grf_traj], axis=1)
    start_time = time.time()
    torques_improved = compute_joint_torques_improved(
        kinodynamic_model,
        state_traj,
        input_traj,
        config.mpc_config.contact_sequence,
        config.mpc_config.mpc_dt,
    )
    time_improved = time.time() - start_time

    print(f"Original computation time: {time_original:.3f}s")
    print(f"Improved computation time: {time_improved:.3f}s")

    return torques_original, torques_improved, time_original, time_improved


def create_comparison_plot(
    torques_original: np.ndarray,
    torques_improved: np.ndarray,
    save_path: str = "results/inverse_dynamics_comparison.png",
) -> str:
    """Create comparison plot in the style of existing debug plots"""

    # Create figure with 3x4 subplots (12 joints)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(
        "Inverse Dynamics Comparison: Original vs Improved Implementation",
        fontsize=16,
        fontweight="bold",
    )

    # Time vector (assuming MPC timesteps)
    time_steps = np.arange(len(torques_original))

    for i in range(12):
        row = i // 4
        col = i % 4
        ax = axes[row, col]

        # Plot both implementations
        ax.plot(
            time_steps,
            torques_original[:, i],
            "b-",
            linewidth=2,
            label="original",
            alpha=0.8,
        )
        ax.plot(
            time_steps,
            torques_improved[:, i],
            "r-",
            linewidth=2,
            label="improved",
            alpha=0.8,
        )

        # Formatting
        ax.set_title(f"Joint {i+1}", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Torque (Nm)")

        # Set consistent y-axis limits for better comparison
        y_min = min(torques_original[:, i].min(), torques_improved[:, i].min())
        y_max = max(torques_original[:, i].max(), torques_improved[:, i].max())
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved to: {save_path}")
    return save_path


def create_difference_plot(
    torques_original: np.ndarray,
    torques_improved: np.ndarray,
    save_path: str = "results/inverse_dynamics_differences.png",
) -> str:
    """Create plot showing the differences between implementations"""

    differences = torques_improved - torques_original

    # Create figure with 3x4 subplots (12 joints)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(
        "Inverse Dynamics Differences: Improved - Original",
        fontsize=16,
        fontweight="bold",
    )

    # Time vector
    time_steps = np.arange(len(differences))

    for i in range(12):
        row = i // 4
        col = i % 4
        ax = axes[row, col]

        # Plot differences
        ax.plot(time_steps, differences[:, i], "g-", linewidth=2, alpha=0.8)
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)

        # Formatting
        ax.set_title(f"Joint {i+1}", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Torque Difference (Nm)")

        # Add RMSE for this joint
        rmse_joint = np.sqrt(np.mean(differences[:, i] ** 2))
        ax.text(
            0.02,
            0.98,
            f"RMSE: {rmse_joint:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Difference plot saved to: {save_path}")
    return save_path


def create_statistics_plot(
    torques_original: np.ndarray,
    torques_improved: np.ndarray,
    time_original: float,
    time_improved: float,
    save_path: str = "results/inverse_dynamics_statistics.png",
) -> str:
    """Create statistical comparison plot"""

    differences = torques_improved - torques_original

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Inverse Dynamics Statistical Analysis", fontsize=16, fontweight="bold"
    )

    # 1. RMSE per joint
    rmse_per_joint = [np.sqrt(np.mean(differences[:, i] ** 2)) for i in range(12)]
    ax1.bar(range(1, 13), rmse_per_joint, color="skyblue", alpha=0.7)
    ax1.set_title("RMSE per Joint")
    ax1.set_xlabel("Joint Number")
    ax1.set_ylabel("RMSE (Nm)")
    ax1.grid(True, alpha=0.3)

    # 2. Histogram of all differences
    ax2.hist(differences.flatten(), bins=50, alpha=0.7, color="lightcoral")
    ax2.set_title("Distribution of Torque Differences")
    ax2.set_xlabel("Torque Difference (Nm)")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)

    # 3. Max torque magnitudes comparison
    max_orig = np.max(np.abs(torques_original), axis=0)
    max_improved = np.max(np.abs(torques_improved), axis=0)
    joints = np.arange(1, 13)
    width = 0.35
    ax3.bar(
        joints - width / 2, max_orig, width, label="Original", alpha=0.7, color="blue"
    )
    ax3.bar(
        joints + width / 2,
        max_improved,
        width,
        label="Improved",
        alpha=0.7,
        color="red",
    )
    ax3.set_title("Maximum Torque Magnitudes")
    ax3.set_xlabel("Joint Number")
    ax3.set_ylabel("Max |Torque| (Nm)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Performance comparison
    methods = ["Original", "Improved"]
    times = [time_original, time_improved]
    colors = ["blue", "red"]
    ax4.bar(methods, times, color=colors, alpha=0.7)
    ax4.set_title("Computation Time Comparison")
    ax4.set_ylabel("Time (seconds)")
    ax4.grid(True, alpha=0.3)

    # Add text annotations
    for i, time_val in enumerate(times):
        ax4.text(
            i,
            time_val + 0.01,
            f"{time_val:.3f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Statistics plot saved to: {save_path}")
    return save_path


def print_summary_statistics(
    torques_original: np.ndarray,
    torques_improved: np.ndarray,
    time_original: float,
    time_improved: float,
) -> None:
    """Print comprehensive summary statistics"""
    differences = torques_improved - torques_original

    print("\n" + "=" * 60)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("=" * 60)

    print("\n📊 Shape Analysis:")
    print(f"   Original shape: {torques_original.shape}")
    print(f"   Improved shape: {torques_improved.shape}")

    print("\n🔢 Statistical Analysis:")
    overall_rmse = np.sqrt(np.mean(differences**2))
    max_abs_diff = np.max(np.abs(differences))
    mean_orig = np.mean(np.abs(torques_original))
    mean_improved = np.mean(np.abs(torques_improved))

    print(f"   Overall RMSE: {overall_rmse:.6f} Nm")
    print(f"   Max absolute difference: {max_abs_diff:.6f} Nm")
    print(f"   Mean |original| torque: {mean_orig:.3f} Nm")
    print(f"   Mean |improved| torque: {mean_improved:.3f} Nm")
    print(f"   Relative error: {(overall_rmse/mean_orig)*100:.2f}%")

    print("\n⏱️  Performance Analysis:")
    print(f"   Original time: {time_original:.3f} seconds")
    print(f"   Improved time: {time_improved:.3f} seconds")
    speedup = time_original / time_improved if time_improved > 0 else float("inf")
    print(f"   Speedup factor: {speedup:.2f}x")

    print("\n🎯 Per-Joint RMSE:")
    for i in range(12):
        joint_rmse = np.sqrt(np.mean(differences[:, i] ** 2))
        print(f"   Joint {i+1:2d}: {joint_rmse:.4f} Nm")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate comparison plots for inverse dynamics implementations"
    )
    parser.add_argument(
        "--output-dir", default="results", help="Output directory for plots"
    )
    parser.add_argument(
        "--show-plots", action="store_true", help="Display plots interactively"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print("🚀 Starting inverse dynamics comparison plot generation...")

    # Run trajectory optimization
    results = run_trajectory_optimization()
    if results is None:
        print("❌ Trajectory optimization failed. Exiting.")
        return

    kinodynamic_model, state_traj, grf_traj, joint_vel_traj = results

    # Compute with both implementations
    torques_original, torques_improved, time_original, time_improved = (
        compute_both_implementations(
            kinodynamic_model, state_traj, grf_traj, joint_vel_traj
        )
    )

    # Generate plots
    print("\nGenerating comparison plots...")

    comparison_path = create_comparison_plot(
        torques_original,
        torques_improved,
        os.path.join(args.output_dir, "inverse_dynamics_comparison.png"),
    )

    difference_path = create_difference_plot(
        torques_original,
        torques_improved,
        os.path.join(args.output_dir, "inverse_dynamics_differences.png"),
    )

    statistics_path = create_statistics_plot(
        torques_original,
        torques_improved,
        time_original,
        time_improved,
        os.path.join(args.output_dir, "inverse_dynamics_statistics.png"),
    )

    # Print summary
    print_summary_statistics(
        torques_original, torques_improved, time_original, time_improved
    )

    # Save data for future use
    np.save(
        os.path.join(args.output_dir, "torques_original_comparison.npy"),
        torques_original,
    )
    np.save(
        os.path.join(args.output_dir, "torques_improved_comparison.npy"),
        torques_improved,
    )

    print("\n🎉 Plot generation complete!")
    print("📁 Generated files:")
    print(f"   - {comparison_path}")
    print(f"   - {difference_path}")
    print(f"   - {statistics_path}")
    print(f"   - {os.path.join(args.output_dir, 'torques_original_comparison.npy')}")
    print(f"   - {os.path.join(args.output_dir, 'torques_improved_comparison.npy')}")

    if args.show_plots:
        plt.show()


if __name__ == "__main__":
    main()
