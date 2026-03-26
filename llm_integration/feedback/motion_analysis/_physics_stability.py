"""Physics & stability sections: friction cone, angular momentum, energy, terminal."""

from __future__ import annotations

import numpy as np

import go2_config

# Foot names in order matching the 4x3 GRF layout and contact_sequence rows
_FOOT_NAMES = ("FL", "FR", "RL", "RR")


def _section_friction_cone(
    grf_traj: np.ndarray,
    contact_sequence: np.ndarray | None,
    mu_friction: float,
) -> list[str]:
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
            if fz < go2_config.analysis_thresholds["negligible_force_threshold"]:
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

    return lines


def _section_angular_momentum(
    state_traj: np.ndarray,
    contact_sequence: np.ndarray | None,
    mpc_dt: float,
) -> list[str]:
    """E. Angular Momentum During Flight."""
    lines: list[str] = []

    if contact_sequence is None:
        lines.append(
            "  Contact sequence not available — skipping angular momentum check."
        )
        return lines

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
        return lines

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

    return lines


def _section_energy_continuity(
    state_traj: np.ndarray,
    robot_mass: float,
    mpc_dt: float,
) -> list[str]:
    """F. Energy Continuity."""
    lines: list[str] = []
    import go2_config

    g = go2_config.experiment.gravity_constant

    com_pos = state_traj[:, 0:3]
    com_vel = state_traj[:, 3:6]
    omega = state_traj[:, 9:12]

    KE_linear = 0.5 * robot_mass * np.sum(com_vel**2, axis=1)
    # Use average diagonal of composite inertia for scalar approximation
    I_approx = float(np.mean(np.diag(go2_config.composite_inertia)))
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

    return lines


def _section_terminal_stability(
    state_traj: np.ndarray,
) -> list[str]:
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
        f"  Final orientation: roll={np.degrees(final_roll):.1f}deg, "
        f"pitch={np.degrees(final_pitch):.1f}deg"
    )

    return lines
