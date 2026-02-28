"""Metrics section formatters for feedback generation."""

from typing import Any


def format_trajectory_metrics_section(trajectory_analysis: dict[str, Any]) -> list[str]:
    """Format trajectory metrics section (comprehensive, used by feedback context)."""
    lines = []
    lines.append("\n" + "-" * 60)
    lines.append("TRAJECTORY METRICS")
    lines.append("-" * 60)

    ta = trajectory_analysis

    # Position metrics
    lines.append("Position:")
    lines.append(
        f"  Height: initial={ta.get('initial_com_height', 0):.3f}m, "
        f"max={ta.get('max_com_height', 0):.3f}m, "
        f"min={ta.get('min_com_height', 0):.3f}m, "
        f"final={ta.get('final_com_height', 0):.3f}m"
    )
    lines.append(f"  Height gain: {ta.get('height_gain', 0):.3f}m")
    flight_peak = ta.get("flight_peak_height", ta.get("max_com_height", 0))
    lines.append(f"  Flight peak height: {flight_peak:.3f}m")
    lines.append(f"  X displacement: {ta.get('com_displacement_x', 0):.3f}m")
    lines.append(f"  Y displacement: {ta.get('com_displacement_y', 0):.3f}m")
    lines.append(f"  Total distance: {ta.get('total_distance', 0):.3f}m")

    # Velocity metrics
    lines.append("Velocity:")
    lines.append(f"  Max COM velocity: {ta.get('max_com_velocity', 0):.2f} m/s")
    lines.append(f"  Final COM velocity: {ta.get('final_com_velocity', 0):.2f} m/s")
    lines.append(f"  Max angular velocity: {ta.get('max_angular_vel', 0):.2f} rad/s")
    lines.append(f"  Max acceleration: {ta.get('max_acceleration', 0):.2f} m/s2")

    # Terminal velocity (per-component)
    lines.append("Terminal velocity:")
    lines.append(
        f"  vx={ta.get('final_vx', 0):.3f} m/s, "
        f"vy={ta.get('final_vy', 0):.3f} m/s, "
        f"vz={ta.get('final_vz', 0):.3f} m/s"
    )
    lines.append(
        f"  wx={ta.get('final_wx', 0):.3f} rad/s, "
        f"wy={ta.get('final_wy', 0):.3f} rad/s, "
        f"wz={ta.get('final_wz', 0):.3f} rad/s"
    )

    # Orientation metrics
    lines.append("Orientation:")
    max_roll = ta.get("max_roll", 0)
    total_roll = ta.get("total_roll_rotation", 0)
    lines.append(
        f"  Roll: max={max_roll:.2f} rad ({max_roll * 57.3:.0f} deg), "
        f"total_change={total_roll:.2f} rad ({total_roll * 57.3:.0f} deg)"
    )
    max_pitch = ta.get("max_pitch", 0)
    total_pitch = ta.get("total_pitch_rotation", 0)
    lines.append(
        f"  Pitch: max={max_pitch:.2f} rad ({max_pitch * 57.3:.0f} deg), "
        f"total_change={total_pitch:.2f} rad ({total_pitch * 57.3:.0f} deg)"
    )
    max_yaw = ta.get("max_yaw", 0)
    total_yaw = ta.get("total_yaw_rotation", 0)
    lines.append(
        f"  Yaw: max_change={max_yaw:.2f} rad ({max_yaw * 57.3:.0f} deg), "
        f"total_change={total_yaw:.2f} rad ({total_yaw * 57.3:.0f} deg)"
    )

    # Timing metrics
    lines.append("Timing:")
    lines.append(f"  Duration: {ta.get('trajectory_duration', 0):.2f}s")
    lines.append(f"  Flight duration: {ta.get('flight_duration', 0):.2f}s")
    lines.append(f"  Flight start: {ta.get('flight_start_time', 0):.2f}s")

    # GRF metrics
    lines.append("Ground Reaction Forces:")
    lines.append(f"  Max total vertical GRF: {ta.get('max_total_grf_z', 0):.1f} N")
    lines.append(f"  Mean total vertical GRF: {ta.get('mean_total_grf_z', 0):.1f} N")
    lines.append(
        f"  Max single-foot vertical GRF: {ta.get('max_single_foot_grf_z', 0):.1f} N"
    )
    lines.append(f"  GRF active fraction: {ta.get('grf_active_fraction', 0):.1%}")

    # Actuator metrics
    lines.append("Actuator (joint velocities):")
    lines.append(f"  Max joint velocity: {ta.get('max_joint_velocity', 0):.2f} rad/s")
    lines.append(f"  Mean joint velocity: {ta.get('mean_joint_velocity', 0):.2f} rad/s")
    lines.append(
        f"  Joint velocity utilization: {ta.get('joint_vel_utilization', 0):.2f} rad/s"
    )

    # Joint range metrics
    lines.append("Joints:")
    lines.append(f"  Max joint range: {ta.get('max_joint_range', 0):.2f} rad")
    lines.append(f"  Avg joint range: {ta.get('avg_joint_range', 0):.2f} rad")

    return lines


def format_trajectory_metrics_text(trajectory_analysis: dict[str, Any]) -> str:
    """Format trajectory metrics as a single text string.

    This is the shared comprehensive formatter used by all LLM calls
    (scoring, constraint feedback, reference feedback, iteration summary).
    """
    if not trajectory_analysis:
        return "No trajectory data available"
    return "\n".join(format_trajectory_metrics_section(trajectory_analysis))
