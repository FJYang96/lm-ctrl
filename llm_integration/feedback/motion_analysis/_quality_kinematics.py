"""Quality & kinematics sections: contact quality, joint quality, manipulability."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._helpers import _build_H, _eval_fk, _eval_jacobian

# Foot names in order matching the 4x3 GRF layout and contact_sequence rows
_FOOT_NAMES = ("FL", "FR", "RL", "RR")


def _section_contact_quality(
    state_traj: np.ndarray,
    contact_sequence: np.ndarray | None,
    kindyn_model: Any,
    mpc_dt: float,
    joint_limits_lower: np.ndarray | None = None,
    joint_limits_upper: np.ndarray | None = None,
) -> list[str]:
    """H. Contact Quality."""
    lines: list[str] = []

    if contact_sequence is None:
        lines.append(
            "  Contact sequence not available — skipping contact quality check."
        )
        return lines

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

    # ── Landing foot placement & leg quality ──
    # Find the last timestep where all 4 feet are in contact (final stance)
    all_on = np.all(contact_sequence[:, :N] > 0.5, axis=0)
    landing_step = None
    for t in range(N - 1, -1, -1):
        if all_on[t]:
            landing_step = t
            break

    # Fallback: last timestep with any contact
    if landing_step is None:
        any_on = np.any(contact_sequence[:, :N] > 0.5, axis=0)
        for t in range(N - 1, -1, -1):
            if any_on[t]:
                landing_step = t
                break

    if landing_step is not None:
        fk_funs = [
            kindyn_model.forward_kinematics_FL_fun,
            kindyn_model.forward_kinematics_FR_fun,
            kindyn_model.forward_kinematics_RL_fun,
            kindyn_model.forward_kinematics_RR_fun,
        ]

        # Compute foot positions at landing
        com_pos_land = state_traj[landing_step, 0:3]
        euler_land = state_traj[landing_step, 6:9]
        joint_pos_land = state_traj[landing_step, 12:24]
        H_land = _build_H(com_pos_land, euler_land)

        foot_positions = np.zeros((4, 3))
        for f_idx, fk_fun in enumerate(fk_funs):
            foot_positions[f_idx] = _eval_fk(fk_fun, H_land, joint_pos_land)

        # Compute initial foot positions for nominal stance reference
        com_pos_init = state_traj[0, 0:3]
        euler_init = state_traj[0, 6:9]
        joint_pos_init = state_traj[0, 12:24]
        H_init = _build_H(com_pos_init, euler_init)

        init_foot_positions = np.zeros((4, 3))
        for f_idx, fk_fun in enumerate(fk_funs):
            init_foot_positions[f_idx] = _eval_fk(fk_fun, H_init, joint_pos_init)

        # Report foot heights at landing
        lines.append(f"  Landing foot placement (step {landing_step}):")
        for f_idx in range(4):
            z = foot_positions[f_idx, 2]
            status = "OK" if abs(z) < 0.01 else f"OFF ({z:.4f}m)"
            lines.append(
                f"    {_FOOT_NAMES[f_idx]}: x={foot_positions[f_idx, 0]:.3f} "
                f"y={foot_positions[f_idx, 1]:.3f} z={z:.4f}m [{status}]"
            )

        # Foot spread: diagonal distances
        diag_fl_rr = float(
            np.linalg.norm(foot_positions[0, :2] - foot_positions[3, :2])
        )
        diag_fr_rl = float(
            np.linalg.norm(foot_positions[1, :2] - foot_positions[2, :2])
        )
        init_diag_fl_rr = float(
            np.linalg.norm(init_foot_positions[0, :2] - init_foot_positions[3, :2])
        )
        init_diag_fr_rl = float(
            np.linalg.norm(init_foot_positions[1, :2] - init_foot_positions[2, :2])
        )

        def _spread_status(current: float, nominal: float) -> str:
            if nominal < 1e-6:
                return "N/A"
            ratio = current / nominal
            if 0.7 <= ratio <= 1.3:
                return "OK"
            return f"{'wide' if ratio > 1.3 else 'narrow'} ({ratio:.2f}x nominal)"

        lines.append(
            f"  Foot spread: FL-RR={diag_fl_rr:.3f}m "
            f"(nominal {init_diag_fl_rr:.3f}m, {_spread_status(diag_fl_rr, init_diag_fl_rr)}), "
            f"FR-RL={diag_fr_rl:.3f}m "
            f"(nominal {init_diag_fr_rl:.3f}m, {_spread_status(diag_fr_rl, init_diag_fr_rl)})"
        )

        # Support polygon: convex hull of foot XY, check if COM XY is inside
        foot_xy = foot_positions[:, :2]
        com_xy = com_pos_land[:2]
        _report_support_polygon(lines, foot_xy, com_xy)

        # Leg joint configuration at landing
        if joint_limits_lower is not None and joint_limits_upper is not None:
            joint_range = joint_limits_upper - joint_limits_lower
            joint_range = np.maximum(joint_range, 1e-6)
            dist_lower = joint_pos_land - joint_limits_lower
            dist_upper = joint_limits_upper - joint_pos_land
            min_dist = np.minimum(dist_lower, dist_upper)
            proximity = min_dist / (joint_range / 2)  # 0 = at limit, 1 = centered

            near_limit_threshold = 0.05
            near_limit_joints = []
            for j in range(len(proximity)):
                if proximity[j] < near_limit_threshold:
                    near_limit_joints.append(f"joint {j} ({proximity[j]:.3f})")

            if near_limit_joints:
                lines.append(
                    f"  Landing joint limits: {len(near_limit_joints)} joint(s) "
                    f"near limits (<{near_limit_threshold}):"
                )
                for entry in near_limit_joints:
                    lines.append(f"    {entry}")
            else:
                lines.append("  Landing joint limits: OK (no joints near limits)")

    return lines


def _report_support_polygon(
    lines: list[str],
    foot_xy: np.ndarray,
    com_xy: np.ndarray,
) -> None:
    """Compute convex hull of foot XY positions and check COM containment."""
    # Simple 2D convex-hull containment using cross-product winding
    # Order points by angle from centroid to form convex hull
    centroid = np.mean(foot_xy, axis=0)
    angles = np.arctan2(foot_xy[:, 1] - centroid[1], foot_xy[:, 0] - centroid[0])
    order = np.argsort(angles)
    hull = foot_xy[order]

    # Check if COM is inside convex hull (cross product method)
    n = len(hull)
    inside = True
    min_margin = float("inf")
    for i in range(n):
        j = (i + 1) % n
        edge = hull[j] - hull[i]
        to_point = com_xy - hull[i]
        cross = edge[0] * to_point[1] - edge[1] * to_point[0]
        # Signed distance from COM to edge line
        edge_len = np.linalg.norm(edge)
        if edge_len > 1e-9:
            margin = cross / edge_len
            min_margin = min(min_margin, margin)
        if cross < 0:
            inside = False

    if inside:
        lines.append(f"  Support polygon: COM inside (margin {min_margin:.4f}m)")
    else:
        lines.append(f"  Support polygon: COM OUTSIDE (margin {min_margin:.4f}m)")


def _section_joint_quality(
    state_traj: np.ndarray,
    grf_traj: np.ndarray,
    joint_limits_lower: np.ndarray,
    joint_limits_upper: np.ndarray,
    kindyn_model: Any,
    torque_limits: np.ndarray | None,
) -> list[str]:
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
        f"  Joint limit proximity (0=at limit, 1=centered): "
        f"min={worst_proximity:.3f} (joint {worst_joint} at step {worst_time_step})"
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
            f"  Torque feasibility: {torque_violations} violations, "
            f"worst ratio={worst_torque_ratio:.3f} (joint {worst_torque_joint})"
        )
    else:
        lines.append("  Torque limits not provided — skipping torque feasibility check")

    return lines


def _section_manipulability(
    state_traj: np.ndarray,
    kindyn_model: Any,
) -> list[str]:
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
            f"  Min manipulability: {min_manip:.6f} "
            f"({_FOOT_NAMES[min_manip_foot]} foot at step {min_manip_step})"
        )
        lines.append(
            f"  Below threshold ({warning_threshold}): "
            f"{below_threshold_count}/{total_count} ({100 * frac_below:.0f}%)"
        )
    else:
        lines.append("  No manipulability data (empty trajectory)")
        frac_below = 0.0

    return lines
