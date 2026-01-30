"""Metrics section formatters for feedback generation."""

import math
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


def format_terminal_state_section(trajectory_analysis: dict[str, Any]) -> list[str]:
    """Format terminal state section for constraint tuning feedback.

    This section shows the final state values so the LLM can adjust terminal constraints.
    Critical for tasks like backflips where terminal pitch should be ~2*pi.
    """
    lines = []
    lines.append("\n" + "-" * 60)
    lines.append("TERMINAL STATE (for terminal constraint tuning)")
    lines.append("-" * 60)

    # Final position
    final_z = trajectory_analysis.get("final_com_height", 0)
    lines.append(f"Position: z={final_z:.3f}m")

    # Final orientation (absolute values - critical for backflips)
    final_roll = trajectory_analysis.get("final_roll", 0)
    final_pitch = trajectory_analysis.get("final_pitch", 0)
    final_yaw = trajectory_analysis.get("final_yaw", 0)
    lines.append(
        f"Orientation: roll={final_roll:.2f} rad ({final_roll * 57.3:.0f}°), "
        f"pitch={final_pitch:.2f} rad ({final_pitch * 57.3:.0f}°), "
        f"yaw={final_yaw:.2f} rad ({final_yaw * 57.3:.0f}°)"
    )

    # Helpful hint for backflip
    if abs(final_pitch) > 0.5:  # Significant pitch rotation
        target_2pi = 2 * math.pi
        diff_from_2pi = abs(abs(final_pitch) - target_2pi)
        if diff_from_2pi < 1.0:  # Within 1 rad of full rotation
            lines.append(
                f"  -> Pitch is {diff_from_2pi:.2f} rad "
                f"({diff_from_2pi * 57.3:.0f}°) from full rotation (2*pi)"
            )

    # Final linear velocity components
    final_vx = trajectory_analysis.get("final_vx", 0)
    final_vy = trajectory_analysis.get("final_vy", 0)
    final_vz = trajectory_analysis.get("final_vz", 0)
    lines.append(
        f"Velocity: vx={final_vx:.2f}, vy={final_vy:.2f}, vz={final_vz:.2f} m/s"
    )

    # Final angular velocity components
    final_wx = trajectory_analysis.get("final_wx", 0)
    final_wy = trajectory_analysis.get("final_wy", 0)
    final_wz = trajectory_analysis.get("final_wz", 0)
    lines.append(
        f"Angular vel: wx={final_wx:.2f}, wy={final_wy:.2f}, wz={final_wz:.2f} rad/s"
    )

    # Landing stability assessment
    vel_magnitude = (final_vx**2 + final_vy**2 + final_vz**2) ** 0.5
    omega_magnitude = (final_wx**2 + final_wy**2 + final_wz**2) ** 0.5
    if vel_magnitude > 1.0 or omega_magnitude > 1.0:
        lines.append(
            "  -> WARNING: High terminal velocity/rotation may cause unstable landing"
        )

    return lines


def format_phase_analysis_section(phase_metrics: dict[str, Any]) -> list[str]:
    """Format phase analysis section."""
    lines = []
    if phase_metrics and "error" not in phase_metrics:
        lines.append("\n" + "-" * 60)
        lines.append("PHASE ANALYSIS")
        lines.append("-" * 60)

        if "stance_pre" in phase_metrics:
            sp = phase_metrics["stance_pre"]
            lines.append(
                f"Pre-flight Stance: {sp['duration']:.2f}s, "
                f"crouch: {sp.get('crouch_depth', 0)*100:.1f}cm, "
                f"takeoff velocity: {sp.get('final_vz', 0):.2f} m/s"
            )
            if "peak_grf" in sp:
                lines.append(f"  Peak GRF: {sp['peak_grf']:.1f}N")

        if "flight" in phase_metrics:
            fp = phase_metrics["flight"]
            lines.append(
                f"Flight: {fp['duration']:.2f}s, "
                f"peak height: {fp['peak_height']:.3f}m"
            )
            lines.append(
                f"  Roll rate: {fp.get('avg_roll_rate', 0):.2f} rad/s, "
                f"Pitch rate: {fp.get('avg_pitch_rate', 0):.2f} rad/s, "
                f"Yaw rate: {fp.get('avg_yaw_rate', 0):.2f} rad/s"
            )
            roll_change = fp.get("total_roll_change", 0)
            pitch_change = fp.get("total_pitch_change", 0)
            yaw_change = fp.get("total_yaw_change", 0)
            lines.append(
                f"  Roll change: {roll_change:.2f} rad ({roll_change * 57.3:.0f}°), "
                f"Pitch change: {pitch_change:.2f} rad ({pitch_change * 57.3:.0f}°), "
                f"Yaw change: {yaw_change:.2f} rad ({yaw_change * 57.3:.0f}°)"
            )

        if "stance_post" in phase_metrics:
            sl = phase_metrics["stance_post"]
            lines.append(
                f"Landing: {sl['duration']:.2f}s, "
                f"impact velocity: {sl.get('impact_velocity', 0):.2f} m/s"
            )
            if "impact_grf" in sl:
                lines.append(f"  Impact GRF: {sl['impact_grf']:.1f}N")

    return lines


def format_grf_section(grf_metrics: dict[str, Any]) -> list[str]:
    """Format GRF analysis section."""
    lines = []
    if grf_metrics and "error" not in grf_metrics:
        lines.append("\n" + "-" * 60)
        lines.append("GROUND REACTION FORCES")
        lines.append("-" * 60)
        lines.append(
            f"Max GRF: {grf_metrics.get('max_total_grf', 0):.1f}N "
            f"({grf_metrics.get('max_grf_ratio', 0):.1f}x body weight)"
        )
        if "grf_at_takeoff" in grf_metrics:
            lines.append(f"GRF at takeoff: {grf_metrics['grf_at_takeoff']:.1f}N")
        if "grf_at_landing" in grf_metrics:
            lines.append(f"GRF at landing: {grf_metrics['grf_at_landing']:.1f}N")
        lines.append(
            f"Left-right asymmetry: {grf_metrics.get('avg_asymmetry', 0)*100:.1f}%"
        )
    return lines


def format_actuator_section(actuator_metrics: dict[str, Any]) -> list[str]:
    """Format actuator status section."""
    lines = []
    if actuator_metrics:
        lines.append("\n" + "-" * 60)
        lines.append("ACTUATOR STATUS")
        lines.append("-" * 60)
        if "max_torque_ratio" in actuator_metrics:
            torque_pct = actuator_metrics["max_torque_ratio"] * 100
            status = "⚠️ NEAR LIMIT" if torque_pct > 90 else "OK"
            lines.append(f"Torque utilization: {torque_pct:.0f}% of limit {status}")
            if "torque_clipping_fraction" in actuator_metrics:
                clip_pct = actuator_metrics["torque_clipping_fraction"] * 100
                if clip_pct > 0:
                    lines.append(f"  ⚠️ Torque clipping: {clip_pct:.1f}% of timesteps")
        if "max_velocity_ratio" in actuator_metrics:
            vel_pct = actuator_metrics["max_velocity_ratio"] * 100
            status = "⚠️ NEAR LIMIT" if vel_pct > 90 else "OK"
            lines.append(
                f"Joint velocity utilization: {vel_pct:.0f}% of limit {status}"
            )

        if "velocity_saturation_by_joint" in actuator_metrics:
            saturated_joints = {
                k: v
                for k, v in actuator_metrics["velocity_saturation_by_joint"].items()
                if v > 0.1
            }
            if saturated_joints:
                lines.append("  Saturated joints (>10% of time at limit):")
                for joint, fraction in sorted(
                    saturated_joints.items(), key=lambda x: -x[1]
                ):
                    lines.append(f"    {joint}: {fraction * 100:.0f}%")
    return lines


def format_comparison_section(
    trajectory_analysis: dict[str, Any],
    previous_iteration_analysis: dict[str, Any] | None,
) -> list[str]:
    """Format comparison to previous iteration section."""
    lines = []
    if previous_iteration_analysis:
        lines.append("\n" + "-" * 60)
        lines.append("VS PREVIOUS ITERATION")
        lines.append("-" * 60)
        prev_height = previous_iteration_analysis.get("height_gain", 0)
        curr_height = trajectory_analysis.get("height_gain", 0)
        height_delta = curr_height - prev_height
        arrow = "↑" if height_delta > 0 else "↓" if height_delta < 0 else "→"
        lines.append(f"Height gain: {height_delta:+.3f}m {arrow}")

        prev_pitch = abs(previous_iteration_analysis.get("total_pitch_rotation", 0))
        curr_pitch = abs(trajectory_analysis.get("total_pitch_rotation", 0))
        pitch_delta = curr_pitch - prev_pitch
        arrow = "↑" if pitch_delta > 0 else "↓" if pitch_delta < 0 else "→"
        lines.append(
            f"Pitch rotation: {pitch_delta:+.2f} rad ({pitch_delta*57.3:+.0f}°) {arrow}"
        )

        prev_yaw = abs(previous_iteration_analysis.get("max_yaw", 0))
        curr_yaw = abs(trajectory_analysis.get("max_yaw", 0))
        yaw_delta = curr_yaw - prev_yaw
        arrow = "↑" if yaw_delta > 0 else "↓" if yaw_delta < 0 else "→"
        lines.append(
            f"Yaw rotation: {yaw_delta:+.2f} rad ({yaw_delta*57.3:+.0f}°) {arrow}"
        )

        prev_roll = abs(previous_iteration_analysis.get("max_roll", 0))
        curr_roll = abs(trajectory_analysis.get("max_roll", 0))
        roll_delta = curr_roll - prev_roll
        arrow = "↑" if roll_delta > 0 else "↓" if roll_delta < 0 else "→"
        lines.append(f"Roll: {roll_delta:+.2f} rad ({roll_delta * 57.3:+.0f}°) {arrow}")
    return lines
