"""Surface-contact sections: smoothness, ground penetration, GRF-contact."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._helpers import _build_H, _eval_fk

# Foot names in order matching the 4x3 GRF layout and contact_sequence rows
_FOOT_NAMES = ("FL", "FR", "RL", "RR")


def _section_smoothness(
    state_traj: np.ndarray,
    mpc_dt: float,
    contact_sequence: np.ndarray | None,
) -> list[str]:
    """A. Smoothness (jerk-based)."""
    lines: list[str] = []

    # COM position jerk
    com_pos = state_traj[:, 0:3]
    com_vel = np.diff(com_pos, axis=0) / mpc_dt
    com_accel = np.diff(com_vel, axis=0) / mpc_dt
    com_jerk = np.diff(com_accel, axis=0) / mpc_dt
    com_jerk_mag = np.linalg.norm(com_jerk, axis=1)
    com_jerk_rms = float(np.sqrt(np.mean(com_jerk_mag**2)))
    com_jerk_max = float(np.max(com_jerk_mag)) if len(com_jerk_mag) > 0 else 0.0

    # Joint angle jerk
    joint_angles = state_traj[:, 12:24]
    joint_vel = np.diff(joint_angles, axis=0) / mpc_dt
    joint_accel = np.diff(joint_vel, axis=0) / mpc_dt
    joint_jerk = np.diff(joint_accel, axis=0) / mpc_dt
    if joint_jerk.shape[0] > 0:
        joint_jerk_rms = float(np.sqrt(np.mean(joint_jerk**2)))
        joint_jerk_max_val = float(np.max(np.abs(joint_jerk)))
        joint_jerk_max_idx = np.unravel_index(
            np.argmax(np.abs(joint_jerk)), joint_jerk.shape
        )
        joint_jerk_max_time = float(joint_jerk_max_idx[0]) * mpc_dt
        joint_jerk_max_joint = int(joint_jerk_max_idx[1])
    else:
        joint_jerk_rms = 0.0
        joint_jerk_max_val = 0.0
        joint_jerk_max_time = 0.0
        joint_jerk_max_joint = 0

    # Angular jerk
    euler = state_traj[:, 6:9]
    euler_vel = np.diff(euler, axis=0) / mpc_dt
    euler_accel = np.diff(euler_vel, axis=0) / mpc_dt
    euler_jerk = np.diff(euler_accel, axis=0) / mpc_dt
    euler_jerk_mag = (
        np.linalg.norm(euler_jerk, axis=1)
        if euler_jerk.shape[0] > 0
        else np.array([0.0])
    )
    euler_jerk_rms = float(np.sqrt(np.mean(euler_jerk_mag**2)))

    # Phase-aware COM jerk breakdown
    stance_jerk_str = ""
    flight_jerk_str = ""
    transition_jerk_str = ""
    if contact_sequence is not None and com_jerk.shape[0] > 0:
        # contact_sequence is (4, N). A timestep is stance if any foot is in contact.
        n_jerk = com_jerk.shape[0]
        # Jerk at timestep k corresponds roughly to states k..k+3; use k+1 as representative
        all_contact = np.sum(contact_sequence, axis=0)  # (N,)
        # Classify each jerk timestep
        stance_mask = np.zeros(n_jerk, dtype=bool)
        flight_mask = np.zeros(n_jerk, dtype=bool)
        transition_mask = np.zeros(n_jerk, dtype=bool)
        for k in range(n_jerk):
            idx = min(k + 1, len(all_contact) - 1)
            idx_prev = max(idx - 1, 0)
            c_now = all_contact[idx] > 0
            c_prev = all_contact[idx_prev] > 0
            if c_now != c_prev:
                transition_mask[k] = True
            elif c_now:
                stance_mask[k] = True
            else:
                flight_mask[k] = True

        def _rms(arr: np.ndarray) -> float:
            return float(np.sqrt(np.mean(arr**2))) if len(arr) > 0 else 0.0

        stance_jerk_str = f"{_rms(com_jerk_mag[stance_mask]):.1f}"
        flight_jerk_str = f"{_rms(com_jerk_mag[flight_mask]):.1f}"
        transition_jerk_str = f"{_rms(com_jerk_mag[transition_mask]):.1f}"

    phase_detail = ""
    if stance_jerk_str:
        phase_detail = (
            f" (stance: {stance_jerk_str}, flight: {flight_jerk_str},"
            f" transitions: {transition_jerk_str})"
        )

    lines.append(
        f"  COM jerk RMS: {com_jerk_rms:.1f} m/s3, "
        f"max: {com_jerk_max:.1f} m/s3{phase_detail}"
    )
    lines.append(
        f"  Joint jerk RMS: {joint_jerk_rms:.1f} rad/s3, "
        f"max: {joint_jerk_max_val:.1f} rad/s3 "
        f"(joint {joint_jerk_max_joint}, at t={joint_jerk_max_time:.3f}s)"
    )
    lines.append(f"  Angular jerk RMS: {euler_jerk_rms:.1f} rad/s3")

    return lines


def _section_ground_penetration(
    state_traj: np.ndarray,
    contact_sequence: np.ndarray | None,
    kindyn_model: Any,
) -> list[str]:
    """B. Ground Penetration & Swing Clearance (FK-based)."""
    lines: list[str] = []
    N = state_traj.shape[0] - 1

    fk_funs = [
        kindyn_model.forward_kinematics_FL_fun,
        kindyn_model.forward_kinematics_FR_fun,
        kindyn_model.forward_kinematics_RL_fun,
        kindyn_model.forward_kinematics_RR_fun,
    ]

    # Compute foot positions at each timestep
    foot_heights = np.zeros((4, state_traj.shape[0]))
    for t in range(state_traj.shape[0]):
        com_pos = state_traj[t, 0:3]
        euler = state_traj[t, 6:9]
        joint_pos = state_traj[t, 12:24]
        H = _build_H(com_pos, euler)
        for f_idx, fk_fun in enumerate(fk_funs):
            pos = _eval_fk(fk_fun, H, joint_pos)
            foot_heights[f_idx, t] = pos[2]

    # Ground penetration
    min_height = float(np.min(foot_heights))
    penetration_threshold = -0.005  # 5mm tolerance
    penetration_mask = foot_heights < penetration_threshold
    n_penetration = int(np.sum(np.any(penetration_mask, axis=0)))
    total_steps = state_traj.shape[0]

    worst_pen_foot = -1
    worst_pen_depth = 0.0
    worst_pen_t = 0
    if np.any(penetration_mask):
        flat_idx = np.argmin(foot_heights)
        worst_pen_foot, worst_pen_t = np.unravel_index(flat_idx, foot_heights.shape)
        worst_pen_depth = float(foot_heights[worst_pen_foot, worst_pen_t])

    if n_penetration > 0:
        lines.append(
            f"  Max penetration: {worst_pen_depth:.4f}m "
            f"({_FOOT_NAMES[worst_pen_foot]} foot at step {worst_pen_t})"
        )
        lines.append(
            f"  Timesteps with penetration: {n_penetration}/{total_steps} "
            f"({100 * n_penetration / total_steps:.0f}%)"
        )
    else:
        lines.append(
            f"  No ground penetration detected (min foot height: {min_height:.4f}m)"
        )

    # Swing clearance
    swing_warnings = []
    if contact_sequence is not None:
        for f_idx in range(4):
            for t in range(min(N, contact_sequence.shape[1])):
                if contact_sequence[f_idx, t] < 0.5:  # swing phase
                    if t < foot_heights.shape[1] and foot_heights[f_idx, t] < 0.005:
                        swing_warnings.append(
                            f"{_FOOT_NAMES[f_idx]} foot at step {t}: "
                            f"z={foot_heights[f_idx, t]:.4f}m"
                        )
        if swing_warnings:
            lines.append(
                f"  Swing clearance warnings "
                f"({len(swing_warnings)} instances of foot < 5mm during swing):"
            )
            for w in swing_warnings[:5]:  # cap display
                lines.append(f"    {w}")
            if len(swing_warnings) > 5:
                lines.append(f"    ... and {len(swing_warnings) - 5} more")
        else:
            lines.append("  Swing clearance: OK (all feet > 5mm during swing phases)")

    return lines


def _section_grf_contact(
    grf_traj: np.ndarray,
    contact_sequence: np.ndarray | None,
) -> list[str]:
    """C. GRF-Contact Consistency."""
    lines: list[str] = []

    if contact_sequence is None:
        lines.append(
            "  Contact sequence not available — skipping GRF consistency check."
        )
        return lines

    N = min(grf_traj.shape[0], contact_sequence.shape[1])
    phantom_count = 0
    missing_count = 0
    worst_phantom_mag = 0.0
    worst_phantom_foot = ""

    for t in range(N):
        for f_idx in range(4):
            grf_foot = grf_traj[t, f_idx * 3 : (f_idx + 1) * 3]
            grf_mag = float(np.linalg.norm(grf_foot))
            contact = contact_sequence[f_idx, t] > 0.5

            if not contact and grf_mag > 1.0:  # phantom force: GRF during flight
                phantom_count += 1
                if grf_mag > worst_phantom_mag:
                    worst_phantom_mag = grf_mag
                    worst_phantom_foot = f"{_FOOT_NAMES[f_idx]} at step {t}"

            if contact and grf_foot[2] < 0.1:  # missing force: no GRF_z during stance
                missing_count += 1

    # Contact chattering
    transitions_per_foot = np.sum(
        np.abs(np.diff(contact_sequence[:, :N], axis=1)), axis=1
    )
    chattering_feet = []
    for f_idx in range(4):
        if transitions_per_foot[f_idx] > 4:
            chattering_feet.append(
                f"{_FOOT_NAMES[f_idx]} ({int(transitions_per_foot[f_idx])} transitions)"
            )

    if phantom_count > 0:
        lines.append(f"  Phantom force violations: {phantom_count} (GRF during flight)")
        lines.append(f"    Worst: {worst_phantom_mag:.1f}N ({worst_phantom_foot})")
    else:
        lines.append("  No phantom force violations (GRF correctly zero during flight)")

    if missing_count > 0:
        lines.append(
            f"  Missing force violations: {missing_count} (no GRF_z during stance)"
        )
    else:
        lines.append("  No missing force violations")

    if chattering_feet:
        lines.append(f"  Contact chattering: {', '.join(chattering_feet)}")
    else:
        lines.append("  Contact chattering: none (clean phase transitions)")

    return lines
