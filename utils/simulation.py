"""
Simulation utilities for trajectory data.
"""

from __future__ import annotations

import numpy as np


def create_reference_trajectory(
    initial_qpos: np.ndarray, target_jump_height: float = 0.15
) -> np.ndarray:
    """
    Create reference trajectory for MPC optimization.

    Args:
        initial_qpos: Initial joint positions from simulation
        target_jump_height: Target jump height in meters

    Returns:
        Reference trajectory vector for MPC optimization
    """
    reference = {
        "ref_position": initial_qpos[0:3] + np.array([0.1, 0.0, target_jump_height]),
        "ref_linear_velocity": np.array([0.0, 0.0, 0.0]),
        "ref_orientation": np.zeros(3),
        "ref_angular_velocity": np.zeros(3),
        "ref_joints": initial_qpos[7:19],
    }

    return np.concatenate(
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


def save_trajectory_results(
    state_traj: np.ndarray,
    joint_vel_traj: np.ndarray,
    grf_traj: np.ndarray,
    contact_sequence: np.ndarray,
    suffix: str = "",
) -> None:
    """
    Save all trajectory optimization results to files.

    Args:
        state_traj: State trajectory from MPC optimization
        joint_vel_traj: Joint velocity trajectory
        grf_traj: Ground reaction force trajectory
        contact_sequence: Contact sequence matrix
        suffix: File suffix for saving
    """
    np.save(f"results/state_traj{suffix}.npy", state_traj)
    np.save(f"results/joint_vel_traj{suffix}.npy", joint_vel_traj)
    np.save(f"results/grf_traj{suffix}.npy", grf_traj)
    np.save("results/contact_sequence.npy", contact_sequence)
