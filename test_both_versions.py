#!/usr/bin/env python3
"""
Script to test and compare both inverse dynamics implementations
"""

import argparse
import time
from typing import Any

import numpy as np

import config
from mpc.dynamics.model import KinoDynamic_Model
from mpc.mpc_opti import QuadrupedMPCOpti
from utils.inv_dyn import compute_joint_torques, compute_joint_torques_improved


def run_trajectory_optimization() -> tuple[Any, ...] | None:
    """Run the trajectory optimization and return the results"""
    print("Setting up kinodynamic model and MPC...")
    kinodynamic_model = KinoDynamic_Model(config)
    mpc = QuadrupedMPCOpti(model=kinodynamic_model, config=config, build=True)

    # Set up initial state and reference
    from utils.conversion import sim_to_mpc
    from utils.simulation import create_reference_trajectory

    initial_state, _ = sim_to_mpc(
        config.experiment.initial_qpos, config.experiment.initial_qvel
    )
    ref = create_reference_trajectory(config.experiment.initial_qpos)

    print("Solving trajectory optimization...")
    state_traj, grf_traj, joint_vel_traj, status = mpc.solve_trajectory(
        initial_state, ref, config.mpc_config.contact_sequence
    )

    if status != 0:
        print(f"❌ Optimization failed with status: {status}")
        return None

    print("✅ Optimization successful!")
    return kinodynamic_model, state_traj, grf_traj, joint_vel_traj


def test_original_inverse_dynamics(
    kinodynamic_model: Any, state_traj: Any, grf_traj: Any
) -> tuple[Any, Any] | tuple[None, None]:
    """Test the original inverse dynamics implementation"""
    print("\n" + "=" * 60)
    print("TESTING ORIGINAL INVERSE DYNAMICS")
    print("=" * 60)

    start_time = time.time()
    try:
        joint_torques_orig = compute_joint_torques(
            kinodynamic_model,
            state_traj,
            grf_traj,
            config.mpc_config.contact_sequence,
            config.mpc_config.mpc_dt,
        )
        computation_time = time.time() - start_time

        print("✅ Original inverse dynamics completed successfully!")
        print(f"⏱️  Computation time: {computation_time:.3f} seconds")
        print(f"📊 Output shape: {joint_torques_orig.shape}")
        print(
            f"📈 Torque range: [{joint_torques_orig.min():.3f}, {joint_torques_orig.max():.3f}] Nm"
        )
        return joint_torques_orig, computation_time

    except Exception as e:
        print(f"❌ Original inverse dynamics failed: {e}")
        return None, None


def test_improved_inverse_dynamics(
    kinodynamic_model: Any, state_traj: Any, grf_traj: Any, joint_vel_traj: Any
) -> tuple[Any, Any] | tuple[None, None]:
    """Test the improved inverse dynamics implementation"""
    print("\n" + "=" * 60)
    print("TESTING IMPROVED INVERSE DYNAMICS")
    print("=" * 60)

    # Create proper input trajectory (joint velocities + GRFs)
    input_traj = np.concatenate([joint_vel_traj, grf_traj], axis=1)

    start_time = time.time()
    try:
        joint_torques_improved = compute_joint_torques_improved(
            kinodynamic_model,
            state_traj,
            input_traj,
            config.mpc_config.contact_sequence,
            config.mpc_config.mpc_dt,
        )
        computation_time = time.time() - start_time

        print("✅ Improved inverse dynamics completed successfully!")
        print(f"⏱️  Computation time: {computation_time:.3f} seconds")
        print(f"📊 Output shape: {joint_torques_improved.shape}")
        print(
            f"📈 Torque range: [{joint_torques_improved.min():.3f}, {joint_torques_improved.max():.3f}] Nm"
        )
        return joint_torques_improved, computation_time

    except Exception as e:
        print(f"❌ Improved inverse dynamics failed: {e}")
        return None, None


def compare_results(
    torques_orig: Any, torques_improved: Any, time_orig: Any, time_improved: Any
) -> None:
    """Compare the results from both implementations"""
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    if torques_orig is None or torques_improved is None:
        print("❌ Cannot compare - one or both implementations failed")
        return

    # Compute differences
    diff = torques_improved - torques_orig
    rmse = np.sqrt(np.mean(diff**2))
    max_abs_diff = np.max(np.abs(diff))

    print("📊 Shape comparison:")
    print(f"   Original: {torques_orig.shape}")
    print(f"   Improved: {torques_improved.shape}")

    print("\n🔢 Numerical comparison:")
    print(f"   RMSE: {rmse:.6f} Nm")
    print(f"   Max absolute difference: {max_abs_diff:.6f} Nm")
    print(f"   Mean original torque: {np.mean(np.abs(torques_orig)):.3f} Nm")
    print(f"   Mean improved torque: {np.mean(np.abs(torques_improved)):.3f} Nm")

    print("\n⏱️  Performance comparison:")
    print(f"   Original time: {time_orig:.3f} seconds")
    print(f"   Improved time: {time_improved:.3f} seconds")
    speedup = time_orig / time_improved if time_improved > 0 else float("inf")
    print(f"   Speedup: {speedup:.2f}x")

    # Assessment
    print("\n🎯 Assessment:")
    if rmse < 1e-3:
        print("   ✅ Results are very similar (RMSE < 1e-3)")
    elif rmse < 1e-2:
        print("   ⚠️  Results have small differences (RMSE < 1e-2)")
    else:
        print("   ❌ Results have significant differences (RMSE >= 1e-2)")

    relative_error = rmse / np.mean(np.abs(torques_orig)) * 100
    print(f"   📊 Relative error: {relative_error:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare inverse dynamics implementations"
    )
    parser.add_argument(
        "--test",
        choices=["original", "improved", "both"],
        default="both",
        help="Which implementation to test",
    )
    args = parser.parse_args()

    print("🚀 Starting inverse dynamics comparison test...")

    # Run trajectory optimization
    results = run_trajectory_optimization()
    if results is None:
        print("❌ Trajectory optimization failed. Exiting.")
        return

    kinodynamic_model, state_traj, grf_traj, joint_vel_traj = results

    # Test implementations based on argument
    torques_orig, time_orig = None, None
    torques_improved, time_improved = None, None

    if args.test in ["original", "both"]:
        torques_orig, time_orig = test_original_inverse_dynamics(
            kinodynamic_model, state_traj, grf_traj
        )

    if args.test in ["improved", "both"]:
        torques_improved, time_improved = test_improved_inverse_dynamics(
            kinodynamic_model, state_traj, grf_traj, joint_vel_traj
        )

    # Compare if both were run
    if args.test == "both":
        compare_results(torques_orig, torques_improved, time_orig, time_improved)

    print("\n🎉 Testing complete!")


if __name__ == "__main__":
    main()
