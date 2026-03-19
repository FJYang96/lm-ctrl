"""Metrics section formatters for feedback generation."""

from typing import Any


def format_trajectory_metrics_section(
    trajectory_analysis: dict[str, Any], opt_success: bool = True
) -> list[str]:
    """Format trajectory metrics section (comprehensive, used by feedback context)."""
    lines = []

    if not opt_success:
        lines.append("")
        lines.append(
            "!! SOLVER DID NOT CONVERGE — METRICS ARE FROM LAST (INFEASIBLE) ITERATE !!"
        )
        lines.append(
            "These values may violate physics. "
            "Use only to diagnose failure, not as achieved results."
        )
        lines.append("")

    lines.append("\n" + "-" * 60)
    lines.append("TRAJECTORY METRICS")
    lines.append("-" * 60)

    ta = trajectory_analysis

    if "error" in ta:
        return [f"Trajectory analysis error: {ta['error']}"]

    # Position metrics
    lines.append("Position:")
    lines.append(
        f"  Height: initial={ta['initial_com_height']:.3f}m, "
        f"max={ta['max_com_height']:.3f}m, "
        f"min={ta['min_com_height']:.3f}m, "
        f"final={ta['final_com_height']:.3f}m"
    )
    lines.append(f"  Height gain: {ta['height_gain']:.3f}m")
    # flight_peak_height is conditionally set (only when flight phase exists)
    flight_peak = ta.get("flight_peak_height", ta["max_com_height"])
    lines.append(f"  Flight peak height: {flight_peak:.3f}m")
    lines.append(f"  X displacement: {ta['com_displacement_x']:.3f}m")
    lines.append(f"  Y displacement: {ta['com_displacement_y']:.3f}m")
    lines.append(f"  Total distance: {ta['total_distance']:.3f}m")

    # Velocity metrics
    lines.append("Velocity:")
    lines.append(f"  Max COM velocity: {ta['max_com_velocity']:.2f} m/s")
    lines.append(f"  Final COM velocity: {ta['final_com_velocity']:.2f} m/s")
    lines.append(f"  Max angular velocity: {ta['max_angular_vel']:.2f} rad/s")
    lines.append(f"  Max acceleration: {ta['max_acceleration']:.2f} m/s2")

    # Terminal velocity (per-component)
    lines.append("Terminal velocity:")
    lines.append(
        f"  vx={ta['final_vx']:.3f} m/s, "
        f"vy={ta['final_vy']:.3f} m/s, "
        f"vz={ta['final_vz']:.3f} m/s"
    )
    lines.append(
        f"  wx={ta['final_wx']:.3f} rad/s, "
        f"wy={ta['final_wy']:.3f} rad/s, "
        f"wz={ta['final_wz']:.3f} rad/s"
    )

    # Orientation metrics
    lines.append("Orientation:")
    max_roll = ta["max_roll"]
    total_roll = ta["total_roll_rotation"]
    lines.append(
        f"  Roll: max={max_roll:.2f} rad ({max_roll * 57.3:.0f} deg), "
        f"total_change={total_roll:.2f} rad ({total_roll * 57.3:.0f} deg)"
    )
    max_pitch = ta["max_pitch"]
    total_pitch = ta["total_pitch_rotation"]
    lines.append(
        f"  Pitch: max={max_pitch:.2f} rad ({max_pitch * 57.3:.0f} deg), "
        f"total_change={total_pitch:.2f} rad ({total_pitch * 57.3:.0f} deg)"
    )
    max_yaw = ta["max_yaw"]
    total_yaw = ta["total_yaw_rotation"]
    lines.append(
        f"  Yaw: max_change={max_yaw:.2f} rad ({max_yaw * 57.3:.0f} deg), "
        f"total_change={total_yaw:.2f} rad ({total_yaw * 57.3:.0f} deg)"
    )

    # Timing metrics
    lines.append("Timing:")
    lines.append(f"  Duration: {ta['trajectory_duration']:.2f}s")
    lines.append(f"  Flight duration: {ta['flight_duration']:.2f}s")
    lines.append(f"  Flight start: {ta['flight_start_time']:.2f}s")

    # GRF metrics
    lines.append("Ground Reaction Forces:")
    lines.append(f"  Max total vertical GRF: {ta['max_total_grf_z']:.1f} N")
    lines.append(f"  Mean total vertical GRF: {ta['mean_total_grf_z']:.1f} N")
    lines.append(f"  Max single-foot vertical GRF: {ta['max_single_foot_grf_z']:.1f} N")
    lines.append(f"  GRF active fraction: {ta['grf_active_fraction']:.1%}")

    # Actuator metrics
    lines.append("Actuator (joint velocities):")
    lines.append(f"  Max joint velocity: {ta['max_joint_velocity']:.2f} rad/s")
    lines.append(f"  Mean joint velocity: {ta['mean_joint_velocity']:.2f} rad/s")
    lines.append(
        f"  Joint velocity utilization: {ta['joint_vel_utilization']:.2f} rad/s"
    )

    # Joint range metrics
    lines.append("Joints:")
    lines.append(f"  Max joint range: {ta['max_joint_range']:.2f} rad")
    lines.append(f"  Avg joint range: {ta['avg_joint_range']:.2f} rad")

    # Per-phase breakdown (stance vs flight)
    if "n_stance_steps" in ta:
        lines.append("Phase breakdown (stance vs flight):")
        n_stance = ta["n_stance_steps"]
        n_flight = ta["n_flight_steps"]
        lines.append(f"  Stance steps: {n_stance}, Flight steps: {n_flight}")

        for axis in ["roll", "pitch", "yaw"]:
            s = ta.get(f"{axis}_change_stance", 0)
            f = ta.get(f"{axis}_change_flight", 0)
            total = s + f
            if total > 0.01:  # only show if meaningful rotation occurred
                stance_pct = s / total * 100
                flight_pct = f / total * 100
                lines.append(
                    f"  {axis.capitalize()} change: stance={s:.2f} rad ({s * 57.3:.0f} deg, {stance_pct:.0f}%), "
                    f"flight={f:.2f} rad ({f * 57.3:.0f} deg, {flight_pct:.0f}%)"
                )

        s_vel = ta.get("max_angular_vel_stance", 0)
        f_vel = ta.get("max_angular_vel_flight", 0)
        lines.append(
            f"  Max angular velocity: stance={s_vel:.2f}, flight={f_vel:.2f} rad/s"
        )

        sh = ta.get("height_change_stance", 0)
        fh = ta.get("height_change_flight", 0)
        lines.append(f"  Height change: stance={sh:+.3f}m, flight={fh:+.3f}m")

    return lines


def format_trajectory_metrics_text(
    trajectory_analysis: dict[str, Any], opt_success: bool = True
) -> str:
    """Format trajectory metrics as a single text string.

    This is the shared comprehensive formatter used by all LLM calls
    (scoring, iteration summary, codegen).
    """
    if not trajectory_analysis:
        return "No trajectory data available"
    return "\n".join(
        format_trajectory_metrics_section(trajectory_analysis, opt_success)
    )
