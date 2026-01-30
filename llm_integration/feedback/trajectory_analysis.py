"""Trajectory analysis utilities for enhanced feedback."""

from __future__ import annotations

from typing import Any

import numpy as np


def analyze_phase_metrics(
    state_traj: np.ndarray,
    contact_sequence: np.ndarray,
    mpc_dt: float,
    grf_traj: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Analyze trajectory metrics for each motion phase (stance, flight, landing).

    Args:
        state_traj: State trajectory (N x state_dim)
        contact_sequence: Contact sequence (4 x horizon)
        mpc_dt: MPC time step
        grf_traj: Ground reaction force trajectory (optional)

    Returns:
        Dictionary with phase-specific metrics
    """
    if state_traj.shape[0] == 0:
        return {"error": "Empty trajectory"}

    # Detect phases from contact sequence
    horizon = contact_sequence.shape[1]
    all_contact = np.all(contact_sequence == 1, axis=0)  # All feet in contact
    no_contact = np.all(contact_sequence == 0, axis=0)  # All feet in air

    phases: dict[str, list[int]] = {"stance_pre": [], "flight": [], "stance_post": []}

    _in_flight = False
    flight_started = False
    for k in range(horizon):
        if no_contact[k]:
            phases["flight"].append(k)
            _in_flight = True
            flight_started = True
        elif all_contact[k]:
            if not flight_started:
                phases["stance_pre"].append(k)
            else:
                phases["stance_post"].append(k)
            _in_flight = False

    # Extract state components
    com_z = state_traj[:, 2]  # Height
    com_vz = state_traj[:, 5]  # Vertical velocity
    roll = state_traj[:, 6]  # Roll angle
    pitch = state_traj[:, 7]  # Pitch angle
    yaw = state_traj[:, 8]  # Yaw angle
    roll_rate = state_traj[:, 9]  # Roll angular velocity
    pitch_rate = state_traj[:, 10]  # Pitch angular velocity
    yaw_rate = state_traj[:, 11]  # Yaw angular velocity

    metrics: dict[str, Any] = {}

    # Pre-flight stance analysis
    if phases["stance_pre"]:
        pre_indices = phases["stance_pre"]
        start_idx = pre_indices[0]
        end_idx = min(pre_indices[-1] + 1, len(com_z))
        metrics["stance_pre"] = {
            "duration": len(pre_indices) * mpc_dt,
            "crouch_depth": float(com_z[start_idx] - np.min(com_z[start_idx:end_idx])),
            "final_vz": float(com_vz[end_idx - 1]) if end_idx > 0 else 0.0,
        }
        if grf_traj is not None and len(grf_traj) > 0:
            # Sum vertical GRFs (indices 2, 5, 8, 11 for z-components)
            grf_z = grf_traj[:, 2] + grf_traj[:, 5] + grf_traj[:, 8] + grf_traj[:, 11]
            if end_idx <= len(grf_z):
                metrics["stance_pre"]["peak_grf"] = float(
                    np.max(grf_z[start_idx:end_idx])
                )

    # Flight phase analysis
    if phases["flight"]:
        flight_indices = phases["flight"]
        start_idx = flight_indices[0]
        end_idx = min(flight_indices[-1] + 1, len(com_z))
        peak_idx = start_idx + np.argmax(com_z[start_idx:end_idx])
        metrics["flight"] = {
            "duration": len(flight_indices) * mpc_dt,
            "peak_height": float(np.max(com_z[start_idx:end_idx])),
            "time_to_peak": (peak_idx - start_idx) * mpc_dt,
            "avg_roll_rate": float(np.mean(np.abs(roll_rate[start_idx:end_idx]))),
            "avg_pitch_rate": float(np.mean(np.abs(pitch_rate[start_idx:end_idx]))),
            "avg_yaw_rate": float(np.mean(np.abs(yaw_rate[start_idx:end_idx]))),
            "total_roll_change": float(roll[end_idx - 1] - roll[start_idx]),
            "total_pitch_change": float(pitch[end_idx - 1] - pitch[start_idx]),
            "total_yaw_change": float(yaw[end_idx - 1] - yaw[start_idx]),
        }

    # Post-flight landing analysis
    if phases["stance_post"]:
        post_indices = phases["stance_post"]
        start_idx = post_indices[0]
        end_idx = min(post_indices[-1] + 1, len(com_z))
        metrics["stance_post"] = {
            "duration": len(post_indices) * mpc_dt,
            "impact_velocity": float(com_vz[start_idx]),
            "final_height": float(com_z[end_idx - 1]),
            "height_recovery": float(com_z[end_idx - 1] - com_z[start_idx]),
        }
        if grf_traj is not None and len(grf_traj) > start_idx:
            grf_z = grf_traj[:, 2] + grf_traj[:, 5] + grf_traj[:, 8] + grf_traj[:, 11]
            if start_idx < len(grf_z):
                metrics["stance_post"]["impact_grf"] = float(grf_z[start_idx])

    return metrics


def analyze_grf_profile(
    grf_traj: np.ndarray, contact_sequence: np.ndarray, robot_mass: float = 15.0
) -> dict[str, Any]:
    """
    Analyze ground reaction force profile.

    Args:
        grf_traj: GRF trajectory (N x 12) - [FL_xyz, FR_xyz, RL_xyz, RR_xyz]
        contact_sequence: Contact sequence (4 x horizon)
        robot_mass: Robot mass in kg

    Returns:
        Dictionary with GRF analysis
    """
    if grf_traj.shape[0] == 0:
        return {"error": "Empty GRF trajectory"}

    # Extract vertical forces for each foot
    grf_z = np.zeros((grf_traj.shape[0], 4))
    grf_z[:, 0] = grf_traj[:, 2]  # FL
    grf_z[:, 1] = grf_traj[:, 5]  # FR
    grf_z[:, 2] = grf_traj[:, 8]  # RL
    grf_z[:, 3] = grf_traj[:, 11]  # RR

    total_grf_z = np.sum(grf_z, axis=1)
    weight = robot_mass * 9.81

    # Find takeoff and landing (transitions to/from zero total GRF)
    contact_mask = total_grf_z > 10  # Some threshold
    takeoff_idx = None
    landing_idx = None

    for i in range(1, len(contact_mask)):
        if contact_mask[i - 1] and not contact_mask[i]:
            takeoff_idx = i - 1
        if not contact_mask[i - 1] and contact_mask[i]:
            landing_idx = i

    metrics = {
        "max_total_grf": float(np.max(total_grf_z)),
        "max_grf_ratio": float(np.max(total_grf_z) / weight),  # Multiple of body weight
        "avg_stance_grf": float(np.mean(total_grf_z[contact_mask]))
        if np.any(contact_mask)
        else 0.0,
    }

    # Left-right asymmetry
    left_grf = grf_z[:, 0] + grf_z[:, 2]  # FL + RL
    right_grf = grf_z[:, 1] + grf_z[:, 3]  # FR + RR
    total = left_grf + right_grf + 1e-6  # Avoid division by zero
    asymmetry = np.abs(left_grf - right_grf) / total
    metrics["avg_asymmetry"] = (
        float(np.mean(asymmetry[contact_mask])) if np.any(contact_mask) else 0.0
    )

    if takeoff_idx is not None:
        metrics["grf_at_takeoff"] = float(total_grf_z[takeoff_idx])
    if landing_idx is not None and landing_idx < len(total_grf_z):
        metrics["grf_at_landing"] = float(total_grf_z[landing_idx])

    return metrics


def analyze_actuator_saturation(
    joint_vel_traj: np.ndarray,
    joint_torques_traj: np.ndarray | None,
    velocity_limit: float = 10.0,
    torque_limit: float = 33.5,
) -> dict[str, Any]:
    """
    Analyze actuator saturation (joints at limits, torque saturation).

    Args:
        joint_vel_traj: Joint velocity trajectory (N x 12)
        joint_torques_traj: Joint torque trajectory (N x 12), optional
        velocity_limit: Joint velocity limit (rad/s)
        torque_limit: Torque limit (Nm)

    Returns:
        Dictionary with saturation analysis
    """
    metrics: dict[str, Any] = {}

    if joint_vel_traj.shape[0] > 0:
        # Velocity saturation
        vel_saturation = np.abs(joint_vel_traj) / velocity_limit
        metrics["max_velocity_ratio"] = float(np.max(vel_saturation))
        metrics["avg_velocity_ratio"] = float(np.mean(vel_saturation))

        # Per-joint saturation (fraction of time near limit)
        near_limit = vel_saturation > 0.9
        joint_names = [
            "FL_hip",
            "FL_thigh",
            "FL_calf",
            "FR_hip",
            "FR_thigh",
            "FR_calf",
            "RL_hip",
            "RL_thigh",
            "RL_calf",
            "RR_hip",
            "RR_thigh",
            "RR_calf",
        ]
        metrics["velocity_saturation_by_joint"] = {
            joint_names[i]: float(np.mean(near_limit[:, i]))
            for i in range(min(12, near_limit.shape[1]))
        }

    if joint_torques_traj is not None and joint_torques_traj.shape[0] > 0:
        # Torque saturation
        torque_saturation = np.abs(joint_torques_traj) / torque_limit
        metrics["max_torque_ratio"] = float(np.max(torque_saturation))
        metrics["avg_torque_ratio"] = float(np.mean(torque_saturation))

        # Check if torques are clipping
        clipping = torque_saturation > 0.95
        metrics["torque_clipping_fraction"] = float(np.mean(clipping))

    return metrics
