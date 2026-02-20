"""Metrics section formatters for feedback generation."""

from typing import Any


def format_trajectory_metrics_section(trajectory_analysis: dict[str, Any]) -> list[str]:
    """Format trajectory metrics section."""
    lines = []
    lines.append("\n" + "-" * 60)
    lines.append("TRAJECTORY METRICS")
    lines.append("-" * 60)

    # Position metrics
    lines.append("Position:")
    lines.append(
        f"  Height: initial={trajectory_analysis.get('initial_com_height', 0):.3f}m, "
        f"max={trajectory_analysis.get('max_com_height', 0):.3f}m, "
        f"min={trajectory_analysis.get('min_com_height', 0):.3f}m, "
        f"final={trajectory_analysis.get('final_com_height', 0):.3f}m"
    )
    lines.append(f"  Height gain: {trajectory_analysis.get('height_gain', 0):.3f}m")
    lines.append(
        f"  X displacement: {trajectory_analysis.get('com_displacement_x', 0):.3f}m"
    )
    lines.append(
        f"  Y displacement: {trajectory_analysis.get('com_displacement_y', 0):.3f}m"
    )
    lines.append(
        f"  Total distance: {trajectory_analysis.get('total_distance', 0):.3f}m"
    )

    # Velocity metrics
    lines.append("Velocity:")
    lines.append(
        f"  Max COM velocity: {trajectory_analysis.get('max_com_velocity', 0):.2f} m/s"
    )
    lines.append(
        f"  Final COM velocity: {trajectory_analysis.get('final_com_velocity', 0):.2f} m/s"
    )
    lines.append(
        f"  Max angular velocity: {trajectory_analysis.get('max_angular_vel', 0):.2f} rad/s"
    )
    lines.append(
        f"  Max acceleration: {trajectory_analysis.get('max_acceleration', 0):.2f} m/s²"
    )

    # Orientation metrics
    lines.append("Orientation:")
    max_roll = trajectory_analysis.get("max_roll", 0)
    total_roll = trajectory_analysis.get("total_roll_rotation", 0)
    lines.append(
        f"  Roll: max={max_roll:.2f} rad ({max_roll * 57.3:.0f}°), "
        f"total_change={total_roll:.2f} rad ({total_roll * 57.3:.0f}°)"
    )
    max_pitch = trajectory_analysis.get("max_pitch", 0)
    total_pitch = trajectory_analysis.get("total_pitch_rotation", 0)
    lines.append(
        f"  Pitch: max={max_pitch:.2f} rad ({max_pitch * 57.3:.0f}°), "
        f"total_change={total_pitch:.2f} rad ({total_pitch * 57.3:.0f}°)"
    )
    max_yaw = trajectory_analysis.get("max_yaw", 0)
    lines.append(f"  Yaw: max_change={max_yaw:.2f} rad ({max_yaw * 57.3:.0f}°)")

    # Timing metrics
    lines.append("Timing:")
    lines.append(
        f"  Duration: {trajectory_analysis.get('trajectory_duration', 0):.2f}s"
    )
    lines.append(
        f"  Flight duration: {trajectory_analysis.get('flight_duration', 0):.2f}s"
    )
    lines.append(
        f"  Flight start: {trajectory_analysis.get('flight_start_time', 0):.2f}s"
    )

    # Joint metrics
    lines.append("Joints:")
    lines.append(
        f"  Max joint range: {trajectory_analysis.get('max_joint_range', 0):.2f} rad"
    )
    lines.append(
        f"  Avg joint range: {trajectory_analysis.get('avg_joint_range', 0):.2f} rad"
    )

    return lines
