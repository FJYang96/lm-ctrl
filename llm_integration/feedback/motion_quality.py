"""Compute motion quality metrics from trajectory data (pure NumPy, no LLM).

Replaces the Gemini video summary with deterministic, computed metrics
covering smoothness, ground penetration, GRF consistency, friction cone,
angular momentum, energy continuity, terminal stability, contact quality,
joint quality, and manipulability.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..logging_config import logger

# Foot names in order matching the 4x3 GRF layout and contact_sequence rows
_FOOT_NAMES = ("FL", "FR", "RL", "RR")


# ---------------------------------------------------------------------------
# Helper: build 4x4 homogeneous transform from COM pos + euler angles
# ---------------------------------------------------------------------------


def _euler_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX intrinsic euler angles to 3x3 rotation matrix (world <- body)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    return R


def _build_H(com_pos: np.ndarray, euler: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transform matching model.py convention."""
    H = np.eye(4)
    H[0:3, 0:3] = _euler_to_rotation(euler[0], euler[1], euler[2])
    H[0:3, 3] = com_pos
    return H


# ---------------------------------------------------------------------------
# Helper: evaluate CasADi FK/Jacobian functions with NumPy inputs
# ---------------------------------------------------------------------------


def _eval_fk(fk_fun: Any, H: np.ndarray, joint_pos: np.ndarray) -> np.ndarray:
    """Evaluate a CasADi FK function and return foot position as (3,) ndarray."""
    result = fk_fun(H, joint_pos)
    return np.array(result[0:3, 3]).flatten()


def _eval_jacobian(jac_fun: Any, H: np.ndarray, joint_pos: np.ndarray) -> np.ndarray:
    """Evaluate a CasADi Jacobian function and return (3, 18) translational Jacobian."""
    result = jac_fun(H, joint_pos)
    return np.array(result[0:3, :]).reshape(3, -1)


# ---------------------------------------------------------------------------
# Section computers — each returns (severity, lines) tuple
# ---------------------------------------------------------------------------


def _section_smoothness(
    state_traj: np.ndarray,
    mpc_dt: float,
    contact_sequence: np.ndarray | None,
) -> tuple[str, list[str]]:
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
        phase_detail = f" (stance: {stance_jerk_str}, flight: {flight_jerk_str}, transitions: {transition_jerk_str})"

    lines.append(
        f"  COM jerk RMS: {com_jerk_rms:.1f} m/s3, max: {com_jerk_max:.1f} m/s3{phase_detail}"
    )
    lines.append(
        f"  Joint jerk RMS: {joint_jerk_rms:.1f} rad/s3, max: {joint_jerk_max_val:.1f} rad/s3 (joint {joint_jerk_max_joint}, at t={joint_jerk_max_time:.3f}s)"
    )
    lines.append(f"  Angular jerk RMS: {euler_jerk_rms:.1f} rad/s3")

    # Severity
    severity = "OK"
    if com_jerk_rms > 500 or joint_jerk_max_val > 5000:
        severity = "CRITICAL"
    elif com_jerk_rms > 200 or joint_jerk_max_val > 2000:
        severity = "WARNING"

    return severity, lines


def _section_ground_penetration(
    state_traj: np.ndarray,
    contact_sequence: np.ndarray | None,
    kindyn_model: Any,
) -> tuple[str, list[str]]:
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
            f"  Max penetration: {worst_pen_depth:.4f}m ({_FOOT_NAMES[worst_pen_foot]} foot at step {worst_pen_t})"
        )
        lines.append(
            f"  Timesteps with penetration: {n_penetration}/{total_steps} ({100 * n_penetration / total_steps:.0f}%)"
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
                            f"{_FOOT_NAMES[f_idx]} foot at step {t}: z={foot_heights[f_idx, t]:.4f}m"
                        )
        if swing_warnings:
            lines.append(
                f"  Swing clearance warnings ({len(swing_warnings)} instances of foot < 5mm during swing):"
            )
            for w in swing_warnings[:5]:  # cap display
                lines.append(f"    {w}")
            if len(swing_warnings) > 5:
                lines.append(f"    ... and {len(swing_warnings) - 5} more")
        else:
            lines.append("  Swing clearance: OK (all feet > 5mm during swing phases)")

    severity = "OK"
    if worst_pen_depth < -0.02:
        severity = "CRITICAL"
    elif worst_pen_depth < -0.005 or swing_warnings:
        severity = "WARNING"

    return severity, lines


def _section_grf_contact(
    grf_traj: np.ndarray,
    contact_sequence: np.ndarray | None,
) -> tuple[str, list[str]]:
    """C. GRF-Contact Consistency."""
    lines: list[str] = []

    if contact_sequence is None:
        lines.append(
            "  Contact sequence not available — skipping GRF consistency check."
        )
        return "OK", lines

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

    severity = "OK"
    if phantom_count > N * 0.1 or missing_count > N * 0.2:
        severity = "CRITICAL"
    elif phantom_count > 0 or missing_count > 0 or chattering_feet:
        severity = "WARNING"

    return severity, lines


def _section_friction_cone(
    grf_traj: np.ndarray,
    contact_sequence: np.ndarray | None,
    mu_friction: float,
) -> tuple[str, list[str]]:
    """D. Friction Cone."""
    lines: list[str] = []
    N = grf_traj.shape[0]

    violations = 0
    worst_ratio = 0.0
    worst_foot = ""

    for t in range(N):
        for f_idx in range(4):
            # Skip if not in contact
            if contact_sequence is not None:
                if t < contact_sequence.shape[1] and contact_sequence[f_idx, t] < 0.5:
                    continue

            grf_foot = grf_traj[t, f_idx * 3 : (f_idx + 1) * 3]
            fz = grf_foot[2]
            if fz < 1.0:  # skip negligible forces
                continue

            tangential = np.sqrt(grf_foot[0] ** 2 + grf_foot[1] ** 2)
            ratio = tangential / fz

            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_foot = f"{_FOOT_NAMES[f_idx]} at step {t}"

            if ratio > mu_friction:
                violations += 1

    if violations == 0:
        lines.append(f"  All timesteps within friction cone (mu={mu_friction})")
    else:
        lines.append(f"  Friction cone violations: {violations}")

    lines.append(f"  Worst tangential/normal ratio: {worst_ratio:.3f} ({worst_foot})")

    severity = "OK"
    if violations > N * 0.1:
        severity = "CRITICAL"
    elif violations > 0:
        severity = "WARNING"

    return severity, lines


def _section_angular_momentum(
    state_traj: np.ndarray,
    contact_sequence: np.ndarray | None,
    mpc_dt: float,
) -> tuple[str, list[str]]:
    """E. Angular Momentum During Flight."""
    lines: list[str] = []

    if contact_sequence is None:
        lines.append(
            "  Contact sequence not available — skipping angular momentum check."
        )
        return "OK", lines

    omega = state_traj[:, 9:12]  # angular velocity
    N = contact_sequence.shape[1]

    # Find flight segments (all 4 feet off ground)
    all_off = np.all(contact_sequence[:, :N] < 0.5, axis=0)

    # Group consecutive flight timesteps into segments
    flight_segments: list[list[int]] = []
    current_seg: list[int] = []
    for t in range(N):
        if all_off[t]:
            current_seg.append(t)
        else:
            if current_seg:
                flight_segments.append(current_seg)
                current_seg = []
    if current_seg:
        flight_segments.append(current_seg)

    if not flight_segments:
        lines.append(
            "  No full-flight phases detected (at least one foot always in contact)"
        )
        return "OK", lines

    max_deviation = 0.0
    for i, seg in enumerate(flight_segments):
        seg_omega = omega[seg]
        mean_omega = np.mean(seg_omega, axis=0)
        deviations = np.linalg.norm(seg_omega - mean_omega, axis=1)
        seg_max_dev = float(np.max(deviations))
        if seg_max_dev > max_deviation:
            max_deviation = seg_max_dev

        t_start = seg[0] * mpc_dt
        t_end = seg[-1] * mpc_dt
        lines.append(
            f"  Flight segment {i + 1} (t={t_start:.3f}-{t_end:.3f}s, {len(seg)} steps): "
            f"omega deviation max={seg_max_dev:.4f} rad/s"
        )

    severity = "OK"
    if max_deviation > 1.0:
        severity = "CRITICAL"
    elif max_deviation > 0.3:
        severity = "WARNING"

    return severity, lines


def _section_energy_continuity(
    state_traj: np.ndarray,
    robot_mass: float,
    mpc_dt: float,
) -> tuple[str, list[str]]:
    """F. Energy Continuity."""
    lines: list[str] = []
    g = 9.81

    com_pos = state_traj[:, 0:3]
    com_vel = state_traj[:, 3:6]
    omega = state_traj[:, 9:12]

    KE_linear = 0.5 * robot_mass * np.sum(com_vel**2, axis=1)
    # Approximate rotational KE using scalar inertia ~ m * 0.01 (rough Go2)
    I_approx = robot_mass * 0.01
    KE_rot = 0.5 * I_approx * np.sum(omega**2, axis=1)
    PE = robot_mass * g * com_pos[:, 2]

    total_E = KE_linear + KE_rot + PE

    # Energy changes between consecutive steps
    dE = np.diff(total_E)
    E_mean = np.mean(np.abs(total_E))
    if E_mean > 1e-6:
        dE_pct = np.abs(dE) / E_mean * 100
    else:
        dE_pct = np.zeros_like(dE)

    max_discontinuity = float(np.max(dE_pct)) if len(dE_pct) > 0 else 0.0
    mean_change_rate = float(np.mean(np.abs(dE))) / mpc_dt if len(dE) > 0 else 0.0

    lines.append(
        f"  Max energy discontinuity: {max_discontinuity:.1f}% (between consecutive steps)"
    )
    lines.append(f"  Mean energy change rate: {mean_change_rate:.1f} J/s")

    severity = "OK"
    if max_discontinuity > 50:
        severity = "CRITICAL"
    elif max_discontinuity > 20:
        severity = "WARNING"

    return severity, lines


def _section_terminal_stability(
    state_traj: np.ndarray,
) -> tuple[str, list[str]]:
    """G. Terminal Stability."""
    lines: list[str] = []

    final = state_traj[-1]
    initial = state_traj[0]

    final_com_vel = np.linalg.norm(final[3:6])
    final_omega = np.linalg.norm(final[9:12])
    height_diff = final[2] - initial[2]
    final_roll = float(np.abs(final[6]))
    final_pitch = float(np.abs(final[7]))

    lines.append(f"  Final COM velocity: {final_com_vel:.4f} m/s")
    lines.append(f"  Final angular velocity: {final_omega:.4f} rad/s")
    lines.append(f"  Height change (final - initial): {height_diff:+.4f} m")
    lines.append(
        f"  Final orientation: roll={np.degrees(final_roll):.1f}deg, pitch={np.degrees(final_pitch):.1f}deg"
    )

    severity = "OK"
    if (
        final_com_vel > 2.0
        or final_omega > 3.0
        or final_roll > 0.5
        or final_pitch > 0.5
    ):
        severity = "CRITICAL"
    elif (
        final_com_vel > 0.5
        or final_omega > 1.0
        or final_roll > 0.2
        or final_pitch > 0.2
    ):
        severity = "WARNING"

    return severity, lines


def _section_contact_quality(
    state_traj: np.ndarray,
    contact_sequence: np.ndarray | None,
    kindyn_model: Any,
    mpc_dt: float,
) -> tuple[str, list[str]]:
    """H. Contact Quality."""
    lines: list[str] = []

    if contact_sequence is None:
        lines.append(
            "  Contact sequence not available — skipping contact quality check."
        )
        return "OK", lines

    N = contact_sequence.shape[1]

    # COM impact velocity: downward velocity at flight->stance transition
    all_off = np.all(contact_sequence[:, :N] < 0.5, axis=0)
    com_vel_z = state_traj[:N, 5]  # vz component

    impact_velocities = []
    for t in range(1, N):
        if all_off[t - 1] and not all_off[t]:  # flight -> at least one foot contacts
            impact_velocities.append((t, float(com_vel_z[t])))

    if impact_velocities:
        for t, vz in impact_velocities:
            lines.append(
                f"  Impact at step {t} (t={t * mpc_dt:.3f}s): COM vz = {vz:.4f} m/s"
            )
    else:
        lines.append("  No flight-to-stance transitions detected")

    # Landing symmetry: variance of per-foot contact onset times
    landing_times = []
    for f_idx in range(4):
        for t in range(1, N):
            if (
                contact_sequence[f_idx, t - 1] < 0.5
                and contact_sequence[f_idx, t] > 0.5
            ):
                landing_times.append(t * mpc_dt)
                break  # first landing per foot

    if len(landing_times) >= 2:
        landing_var = float(np.var(landing_times))
        lines.append(
            f"  Landing timing variance: {landing_var:.6f} s2 ({len(landing_times)} feet)"
        )
    elif landing_times:
        lines.append(
            f"  Only {len(landing_times)} foot landing detected — cannot assess symmetry"
        )
    else:
        lines.append("  No individual foot landings detected")

    severity = "OK"
    if impact_velocities and any(vz < -2.0 for _, vz in impact_velocities):
        severity = "CRITICAL"
    elif impact_velocities and any(vz < -1.0 for _, vz in impact_velocities):
        severity = "WARNING"

    return severity, lines


def _section_joint_quality(
    state_traj: np.ndarray,
    grf_traj: np.ndarray,
    joint_limits_lower: np.ndarray,
    joint_limits_upper: np.ndarray,
    kindyn_model: Any,
    torque_limits: np.ndarray | None,
) -> tuple[str, list[str]]:
    """I. Joint Quality."""
    lines: list[str] = []

    joint_angles = state_traj[:, 12:24]
    joint_range = joint_limits_upper - joint_limits_lower
    joint_range = np.maximum(joint_range, 1e-6)  # avoid division by zero

    # Joint limit proximity
    dist_lower = joint_angles - joint_limits_lower
    dist_upper = joint_limits_upper - joint_angles
    min_dist = np.minimum(dist_lower, dist_upper)
    min_frac = min_dist / (joint_range / 2)

    worst_proximity = float(np.min(min_frac))
    worst_idx = np.unravel_index(np.argmin(min_frac), min_frac.shape)
    worst_joint = int(worst_idx[1])
    worst_time_step = int(worst_idx[0])

    lines.append(
        f"  Joint limit proximity (0=at limit, 1=centered): min={worst_proximity:.3f} (joint {worst_joint} at step {worst_time_step})"
    )

    # Torque feasibility via J^T * F
    if torque_limits is not None:
        jac_funs = [
            kindyn_model.jacobian_FL_fun,
            kindyn_model.jacobian_FR_fun,
            kindyn_model.jacobian_RL_fun,
            kindyn_model.jacobian_RR_fun,
        ]

        torque_violations = 0
        worst_torque_ratio = 0.0
        worst_torque_joint = 0
        N_input = grf_traj.shape[0]

        for t in range(N_input):
            com_pos = state_traj[t, 0:3]
            euler = state_traj[t, 6:9]
            joint_pos = state_traj[t, 12:24]
            H = _build_H(com_pos, euler)

            for f_idx, jac_fun in enumerate(jac_funs):
                grf_foot = grf_traj[t, f_idx * 3 : (f_idx + 1) * 3]
                if np.linalg.norm(grf_foot) < 1.0:
                    continue

                J_full = _eval_jacobian(jac_fun, H, joint_pos)
                # J_full is (3, 18): first 6 cols = base DOFs, next 12 = joints
                # Foot f_idx uses joint columns 6 + f_idx*3 : 6 + f_idx*3 + 3
                j_start = 6 + f_idx * 3
                J_leg = J_full[:, j_start : j_start + 3]
                tau_leg = J_leg.T @ grf_foot  # (3,)

                leg_torque_limits = torque_limits[f_idx * 3 : (f_idx + 1) * 3]
                ratios = np.abs(tau_leg) / np.maximum(leg_torque_limits, 1e-6)
                max_ratio = float(np.max(ratios))
                if max_ratio > worst_torque_ratio:
                    worst_torque_ratio = max_ratio
                    worst_torque_joint = f_idx * 3 + int(np.argmax(ratios))

                if max_ratio > 1.0:
                    torque_violations += 1

        lines.append(
            f"  Torque feasibility: {torque_violations} violations, worst ratio={worst_torque_ratio:.3f} (joint {worst_torque_joint})"
        )
    else:
        lines.append("  Torque limits not provided — skipping torque feasibility check")

    severity = "OK"
    if worst_proximity < 0.02:
        severity = "CRITICAL"
    elif worst_proximity < 0.1:
        severity = "WARNING"

    return severity, lines


def _section_manipulability(
    state_traj: np.ndarray,
    kindyn_model: Any,
) -> tuple[str, list[str]]:
    """J. Manipulability."""
    lines: list[str] = []

    jac_funs = [
        kindyn_model.jacobian_FL_fun,
        kindyn_model.jacobian_FR_fun,
        kindyn_model.jacobian_RL_fun,
        kindyn_model.jacobian_RR_fun,
    ]

    min_manip = float("inf")
    min_manip_foot = 0
    min_manip_step = 0
    warning_threshold = 0.001
    below_threshold_count = 0
    total_count = 0

    N = state_traj.shape[0]
    # Sample every few steps to keep computation reasonable
    step_size = max(1, N // 50)

    for t in range(0, N, step_size):
        com_pos = state_traj[t, 0:3]
        euler = state_traj[t, 6:9]
        joint_pos = state_traj[t, 12:24]
        H = _build_H(com_pos, euler)

        for f_idx, jac_fun in enumerate(jac_funs):
            J_full = _eval_jacobian(jac_fun, H, joint_pos)
            j_start = 6 + f_idx * 3
            J_leg = J_full[:, j_start : j_start + 3]

            # Manipulability: sqrt(det(J @ J.T))
            JJT = J_leg @ J_leg.T
            det_val = np.linalg.det(JJT)
            manip = np.sqrt(max(det_val, 0.0))

            total_count += 1
            if manip < warning_threshold:
                below_threshold_count += 1

            if manip < min_manip:
                min_manip = manip
                min_manip_foot = f_idx
                min_manip_step = t

    if total_count > 0:
        frac_below = below_threshold_count / total_count
        lines.append(
            f"  Min manipulability: {min_manip:.6f} ({_FOOT_NAMES[min_manip_foot]} foot at step {min_manip_step})"
        )
        lines.append(
            f"  Below threshold ({warning_threshold}): {below_threshold_count}/{total_count} ({100 * frac_below:.0f}%)"
        )
    else:
        lines.append("  No manipulability data (empty trajectory)")
        frac_below = 0.0

    severity = "OK"
    if min_manip < 1e-6 or frac_below > 0.3:
        severity = "CRITICAL"
    elif min_manip < warning_threshold or frac_below > 0.1:
        severity = "WARNING"

    return severity, lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_motion_quality_report(
    state_traj: np.ndarray,
    grf_traj: np.ndarray,
    joint_vel_traj: np.ndarray,
    mpc_dt: float,
    contact_sequence: np.ndarray | None,
    kindyn_model: Any,
    joint_limits_lower: np.ndarray,
    joint_limits_upper: np.ndarray,
    robot_mass: float,
    mu_friction: float,
    torque_limits: np.ndarray | None = None,
) -> str:
    """Compute a comprehensive motion quality report from trajectory data.

    Args:
        state_traj: (N+1, 24) state trajectory — pos(3), vel(3), euler(3), omega(3), joints(12).
        grf_traj: (N, 12) ground reaction forces — 4 feet x 3 forces.
        joint_vel_traj: (N, 12) joint velocities.
        mpc_dt: Timestep duration in seconds.
        contact_sequence: (4, N) binary contact sequence, or None.
        kindyn_model: KinoDynamic_Model instance with FK/Jacobian functions.
        joint_limits_lower: (12,) per-joint lower limits.
        joint_limits_upper: (12,) per-joint upper limits.
        robot_mass: Robot mass in kg.
        mu_friction: Ground friction coefficient.
        torque_limits: (12,) per-joint torque limits, or None.

    Returns:
        Formatted text report with OK/WARNING/CRITICAL per section.
    """
    sections: list[tuple[str, str, list[str]]] = []

    try:
        sev, lns = _section_smoothness(state_traj, mpc_dt, contact_sequence)
        sections.append(("SMOOTHNESS", sev, lns))
    except Exception as e:
        logger.warning(f"Motion quality: smoothness computation failed: {e}")
        sections.append(("SMOOTHNESS", "ERROR", [f"  Computation failed: {e}"]))

    try:
        sev, lns = _section_ground_penetration(
            state_traj, contact_sequence, kindyn_model
        )
        sections.append(("GROUND PENETRATION", sev, lns))
    except Exception as e:
        logger.warning(f"Motion quality: ground penetration computation failed: {e}")
        sections.append(("GROUND PENETRATION", "ERROR", [f"  Computation failed: {e}"]))

    try:
        sev, lns = _section_grf_contact(grf_traj, contact_sequence)
        sections.append(("GRF-CONTACT CONSISTENCY", sev, lns))
    except Exception as e:
        logger.warning(f"Motion quality: GRF-contact computation failed: {e}")
        sections.append(
            ("GRF-CONTACT CONSISTENCY", "ERROR", [f"  Computation failed: {e}"])
        )

    try:
        sev, lns = _section_friction_cone(grf_traj, contact_sequence, mu_friction)
        sections.append(("FRICTION CONE", sev, lns))
    except Exception as e:
        logger.warning(f"Motion quality: friction cone computation failed: {e}")
        sections.append(("FRICTION CONE", "ERROR", [f"  Computation failed: {e}"]))

    try:
        sev, lns = _section_angular_momentum(state_traj, contact_sequence, mpc_dt)
        sections.append(("ANGULAR MOMENTUM (FLIGHT)", sev, lns))
    except Exception as e:
        logger.warning(f"Motion quality: angular momentum computation failed: {e}")
        sections.append(
            ("ANGULAR MOMENTUM (FLIGHT)", "ERROR", [f"  Computation failed: {e}"])
        )

    try:
        sev, lns = _section_energy_continuity(state_traj, robot_mass, mpc_dt)
        sections.append(("ENERGY CONTINUITY", sev, lns))
    except Exception as e:
        logger.warning(f"Motion quality: energy continuity computation failed: {e}")
        sections.append(("ENERGY CONTINUITY", "ERROR", [f"  Computation failed: {e}"]))

    try:
        sev, lns = _section_terminal_stability(state_traj)
        sections.append(("TERMINAL STABILITY", sev, lns))
    except Exception as e:
        logger.warning(f"Motion quality: terminal stability computation failed: {e}")
        sections.append(("TERMINAL STABILITY", "ERROR", [f"  Computation failed: {e}"]))

    try:
        sev, lns = _section_contact_quality(
            state_traj, contact_sequence, kindyn_model, mpc_dt
        )
        sections.append(("CONTACT QUALITY", sev, lns))
    except Exception as e:
        logger.warning(f"Motion quality: contact quality computation failed: {e}")
        sections.append(("CONTACT QUALITY", "ERROR", [f"  Computation failed: {e}"]))

    try:
        sev, lns = _section_joint_quality(
            state_traj,
            grf_traj,
            joint_limits_lower,
            joint_limits_upper,
            kindyn_model,
            torque_limits,
        )
        sections.append(("JOINT QUALITY", sev, lns))
    except Exception as e:
        logger.warning(f"Motion quality: joint quality computation failed: {e}")
        sections.append(("JOINT QUALITY", "ERROR", [f"  Computation failed: {e}"]))

    try:
        sev, lns = _section_manipulability(state_traj, kindyn_model)
        sections.append(("MANIPULABILITY", sev, lns))
    except Exception as e:
        logger.warning(f"Motion quality: manipulability computation failed: {e}")
        sections.append(("MANIPULABILITY", "ERROR", [f"  Computation failed: {e}"]))

    # Assemble report
    report_lines: list[str] = []
    for name, sev, lns in sections:
        report_lines.append(f"{name}: {sev}")
        report_lines.extend(lns)
        report_lines.append("")

    return "\n".join(report_lines).strip()
