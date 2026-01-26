"""
Enhanced feedback generation for LLM constraint refinement.

This module provides rich, structured feedback to help the LLM understand:
- What the robot actually did (phase-by-phase analysis)
- How close it got to the goal (task progress metrics)
- What's limiting performance (actuator saturation, constraint binding)
- Visual feedback via key video frames
"""

import base64
from pathlib import Path
from typing import Any

import numpy as np

# Try to import cv2 for video frame extraction
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Video frame extraction disabled.")


def extract_key_frames(
    video_path: str | Path, num_frames: int = 4, resize: tuple[int, int] = (320, 240)
) -> list[str]:
    """
    Extract evenly-spaced key frames from a video as base64-encoded images.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        resize: Target size (width, height) for efficiency

    Returns:
        List of base64-encoded PNG images
    """
    if not CV2_AVAILABLE:
        return []

    video_path = Path(video_path)
    if not video_path.exists():
        return []

    try:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < num_frames:
            num_frames = max(1, total_frames)

        # Get evenly spaced frame indices
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames_base64 = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize for efficiency
                frame = cv2.resize(frame, resize)
                _, buffer = cv2.imencode(".png", frame)
                img_base64 = base64.b64encode(buffer).decode("utf-8")
                frames_base64.append(img_base64)

        cap.release()
        return frames_base64

    except Exception as e:
        print(f"Warning: Failed to extract frames from {video_path}: {e}")
        return []


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


def compute_task_progress(
    command: str, trajectory_analysis: dict[str, Any]
) -> dict[str, Any]:
    """
    Compute progress toward task-specific goals.

    Args:
        command: Natural language command
        trajectory_analysis: Trajectory analysis from constraint_generator

    Returns:
        Dictionary with task progress metrics
    """
    command_lower = command.lower()
    progress: dict[str, Any] = {"command": command, "criteria": []}

    height_gain = trajectory_analysis.get("height_gain", 0)
    total_pitch = abs(trajectory_analysis.get("total_pitch_rotation", 0))
    max_yaw = abs(trajectory_analysis.get("max_yaw", 0))
    displacement_x = abs(trajectory_analysis.get("com_displacement_x", 0))
    displacement_y = abs(trajectory_analysis.get("com_displacement_y", 0))

    # Determine task type and compute progress
    if any(word in command_lower for word in ["backflip", "flip", "somersault"]):
        # Backflip: needs ~2π pitch rotation + height
        target_rotation = 6.28
        target_height = 0.3
        progress["criteria"].append(
            {
                "name": "Pitch Rotation",
                "required": f"{target_rotation:.2f} rad (360°)",
                "achieved": f"{total_pitch:.2f} rad ({total_pitch * 57.3:.0f}°)",
                "progress": min(1.0, total_pitch / target_rotation),
            }
        )
        progress["criteria"].append(
            {
                "name": "Height Gain",
                "required": f"{target_height:.2f}m",
                "achieved": f"{height_gain:.2f}m",
                "progress": min(1.0, max(0, height_gain) / target_height),
            }
        )

    elif any(
        word in command_lower for word in ["turn around", "spin", "rotate", "turn"]
    ):
        # Turn: needs yaw rotation
        if "360" in command_lower or "full" in command_lower:
            target_yaw = 6.28
        elif "180" in command_lower or "around" in command_lower:
            target_yaw = 3.14
        else:
            target_yaw = 3.14  # Default to 180°

        progress["criteria"].append(
            {
                "name": "Yaw Rotation",
                "required": f"{target_yaw:.2f} rad ({target_yaw * 57.3:.0f}°)",
                "achieved": f"{max_yaw:.2f} rad ({max_yaw * 57.3:.0f}°)",
                "progress": min(1.0, max_yaw / target_yaw),
            }
        )

        # If "jump" is also mentioned, add height criterion
        if "jump" in command_lower:
            target_height = 0.15
            progress["criteria"].append(
                {
                    "name": "Height Gain",
                    "required": f"{target_height:.2f}m",
                    "achieved": f"{height_gain:.2f}m",
                    "progress": min(1.0, max(0, height_gain) / target_height),
                }
            )

    elif any(word in command_lower for word in ["jump", "hop", "leap"]):
        # Jump: needs height
        if "high" in command_lower:
            target_height = 0.4
        else:
            target_height = 0.2

        progress["criteria"].append(
            {
                "name": "Height Gain",
                "required": f"{target_height:.2f}m",
                "achieved": f"{height_gain:.2f}m",
                "progress": min(1.0, max(0, height_gain) / target_height),
            }
        )

        # Directional jumps
        if any(word in command_lower for word in ["forward", "ahead"]):
            target_x = 0.3
            progress["criteria"].append(
                {
                    "name": "Forward Displacement",
                    "required": f"{target_x:.2f}m",
                    "achieved": f"{displacement_x:.2f}m",
                    "progress": min(1.0, displacement_x / target_x),
                }
            )
        elif any(word in command_lower for word in ["left", "right", "side"]):
            target_y = 0.2
            progress["criteria"].append(
                {
                    "name": "Lateral Displacement",
                    "required": f"{target_y:.2f}m",
                    "achieved": f"{displacement_y:.2f}m",
                    "progress": min(1.0, displacement_y / target_y),
                }
            )

    elif any(word in command_lower for word in ["squat", "crouch", "lower"]):
        # Squat: needs negative height change
        target_crouch = 0.1
        crouch = -height_gain if height_gain < 0 else 0
        progress["criteria"].append(
            {
                "name": "Crouch Depth",
                "required": f"{target_crouch:.2f}m",
                "achieved": f"{crouch:.2f}m",
                "progress": min(1.0, crouch / target_crouch),
            }
        )

    else:
        # Unknown task - report general motion
        progress["criteria"].append(
            {
                "name": "Height Change",
                "required": "Any motion",
                "achieved": f"{height_gain:.2f}m",
                "progress": min(1.0, abs(height_gain) / 0.1) if height_gain != 0 else 0,
            }
        )

    # Compute overall progress
    if progress["criteria"]:
        progress["overall_progress"] = sum(
            c["progress"] for c in progress["criteria"]
        ) / len(progress["criteria"])
    else:
        progress["overall_progress"] = 0.0

    return progress


def format_enhanced_feedback(
    iteration: int,
    command: str,
    optimization_status: dict[str, Any],
    simulation_results: dict[str, Any],
    trajectory_analysis: dict[str, Any],
    phase_metrics: dict[str, Any],
    grf_metrics: dict[str, Any],
    actuator_metrics: dict[str, Any],
    task_progress: dict[str, Any],
    previous_constraints: str,
    previous_iteration_analysis: dict[str, Any] | None = None,
    initial_height: float = 0.2117,
) -> str:
    """
    Format all feedback into a structured string for the LLM.

    Args:
        initial_height: Robot's initial COM height from config

    Returns:
        Formatted feedback string
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"ITERATION {iteration} FEEDBACK")
    lines.append("=" * 60)

    # MPC Configuration Summary
    if "config_summary" in optimization_status:
        config = optimization_status["config_summary"]
        lines.append("\n" + "-" * 60)
        lines.append("MPC CONFIGURATION")
        lines.append("-" * 60)
        lines.append(f"Task: {config.get('task_name', 'unknown')}")
        lines.append(f"Duration: {config.get('duration', 0):.2f}s")
        lines.append(f"Time step: {config.get('time_step', 0.02):.3f}s")
        lines.append(f"Horizon: {config.get('horizon', 0)} steps")
        lines.append(f"Constraints: {config.get('num_constraints', 0)}")
        if "contact_phases" in config:
            lines.append("Contact phases:")
            for phase in config["contact_phases"]:
                pattern = phase.get("contact_pattern", phase.get("pattern", []))
                phase_end = phase["start_time"] + phase["duration"]
                lines.append(
                    f"  {phase['phase_type']}: {phase['start_time']:.2f}-{phase_end:.2f}s {pattern}"
                )

    # Optimization status
    if optimization_status.get("converged", False):
        lines.append("\n✅ OPTIMIZATION: SUCCESS")
        if "solver_iterations" in optimization_status:
            lines.append(
                f"  Solver converged in {optimization_status['solver_iterations']} iterations"
            )
    else:
        lines.append("\n❌ OPTIMIZATION: FAILED")
        # Detailed error info
        if (
            "error_message" in optimization_status
            and optimization_status["error_message"]
        ):
            lines.append(f"  Error: {optimization_status['error_message']}")
        if "solver_iterations" in optimization_status:
            lines.append(
                f"  Solver stopped at {optimization_status['solver_iterations']} iterations"
            )
        if "infeasibility_info" in optimization_status:
            lines.append(
                f"  Infeasibility: {optimization_status['infeasibility_info']}"
            )
        lines.append("")
        lines.append("COMMON FAILURE CAUSES:")
        lines.append(
            f"  1. Constraints violate initial state (t=0) - robot starts at height={initial_height:.4f}m"
        )
        lines.append(
            "  2. Mutually exclusive constraints (e.g., height>0.5 AND height<0.3)"
        )
        lines.append("  3. Contact sequence doesn't match constraint timing")
        lines.append("  4. Bounds too tight - try loosening by 20%")
        lines.append("")
        lines.append(
            "Fix: SIMPLIFY and LOOSEN constraints. Start with just one key constraint."
        )
        lines.append("")
        lines.append("⚠️  DON'T BE AFRAID TO CHANGE YOUR APPROACH/CODE DRASTICALLY:")
        lines.append(
            "  - If optimization failed, the constraint STRUCTURE may be fundamentally flawed"
        )
        lines.append(
            "  - Small tweaks to bad structure won't help - RETHINK the ENTIRE approach"
        )
        lines.append(
            "  - Consider: constrain ONLY the final state (e.g., final yaw = -π)"
        )
        lines.append(
            "  - Avoid progressive bounds that create 'traps' (loose early, tight late)"
        )

    # Simulation status
    if simulation_results.get("success", False):
        tracking_error = simulation_results.get("tracking_error", 0)
        lines.append(f"✅ SIMULATION: SUCCESS (tracking error: {tracking_error:.3f})")
    else:
        lines.append("❌ SIMULATION: FAILED")
        # Detailed failure info
        if "error" in simulation_results:
            lines.append(f"  Error: {simulation_results['error']}")
        sim_analysis = simulation_results.get("simulation_analysis", {})
        if sim_analysis.get("issues"):
            lines.append("  Issues detected:")
            for issue in sim_analysis["issues"]:
                lines.append(f"    - {issue}")

    # Task Progress Table
    lines.append("\n" + "-" * 60)
    lines.append("TASK PROGRESS")
    lines.append("-" * 60)
    lines.append(
        f"{'Criterion':<25} {'Required':<15} {'Achieved':<15} {'Progress':<10}"
    )
    lines.append("-" * 60)
    for criterion in task_progress.get("criteria", []):
        progress_pct = criterion["progress"] * 100
        status = "✓" if progress_pct >= 90 else "⚠️" if progress_pct >= 50 else "✗"
        lines.append(
            f"{criterion['name']:<25} {criterion['required']:<15} "
            f"{criterion['achieved']:<15} {progress_pct:>5.0f}% {status}"
        )
    lines.append("-" * 60)
    overall = task_progress.get("overall_progress", 0) * 100
    lines.append(f"{'OVERALL TASK COMPLETION:':<55} {overall:>5.0f}%")

    # Comprehensive Trajectory Metrics
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

    # Phase Analysis
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

    # GRF Analysis
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

    # Actuator Analysis
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

        # Per-joint breakdown
        if "velocity_saturation_by_joint" in actuator_metrics:
            saturated_joints = {
                k: v
                for k, v in actuator_metrics["velocity_saturation_by_joint"].items()
                if v > 0.1  # Only show joints saturated >10% of time
            }
            if saturated_joints:
                lines.append("  Saturated joints (>10% of time at limit):")
                for joint, fraction in sorted(
                    saturated_joints.items(), key=lambda x: -x[1]
                ):
                    lines.append(f"    {joint}: {fraction * 100:.0f}%")

    # Comparison to previous iteration
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

    # Initial State Reminder
    lines.append("\n" + "-" * 60)
    lines.append("REMINDER: INITIAL STATE")
    lines.append("-" * 60)
    lines.append(
        f"Robot starts at: height={initial_height:.4f}m, roll=0, pitch=0, yaw=0"
    )
    lines.append("Constraints at k=0 MUST allow this state!")

    # Previous code
    lines.append("\n" + "-" * 60)
    lines.append("PREVIOUS CODE")
    lines.append("-" * 60)
    lines.append(previous_constraints)

    # Instructions
    lines.append("\n" + "=" * 60)
    lines.append("TASK: Generate improved constraints based on this feedback.")
    lines.append("Return ONLY Python code.")
    lines.append("=" * 60)

    return "\n".join(lines)


def create_visual_feedback(
    run_dir: Path, iteration: int, num_frames: int = 4
) -> list[str]:
    """
    Extract key frames from planned and simulated trajectory videos.

    Args:
        run_dir: Directory containing iteration results
        iteration: Iteration number
        num_frames: Number of frames to extract per video

    Returns:
        List of base64-encoded images (planned frames first, then simulated)
    """
    images: list[str] = []

    # Extract from planned trajectory
    planned_video = run_dir / f"planned_traj_iter_{iteration}.mp4"
    if planned_video.exists():
        planned_frames = extract_key_frames(planned_video, num_frames)
        images.extend(planned_frames)

    # Extract from simulation
    sim_video = run_dir / f"simulation_iter_{iteration}.mp4"
    if sim_video.exists():
        sim_frames = extract_key_frames(sim_video, num_frames)
        images.extend(sim_frames)

    return images


def generate_enhanced_feedback(
    iteration: int,
    command: str,
    state_traj: np.ndarray,
    grf_traj: np.ndarray,
    joint_vel_traj: np.ndarray,
    joint_torques_traj: np.ndarray | None,
    contact_sequence: np.ndarray,
    mpc_dt: float,
    optimization_status: dict[str, Any],
    simulation_results: dict[str, Any],
    trajectory_analysis: dict[str, Any],
    previous_constraints: str,
    previous_iteration_analysis: dict[str, Any] | None = None,
    robot_mass: float = 15.0,
    initial_height: float = 0.2117,
) -> str:
    """
    Generate comprehensive enhanced feedback for the LLM.

    This is the main entry point that combines all analysis.

    Args:
        initial_height: Robot's initial COM height from config

    Returns:
        Formatted feedback string
    """
    # Phase analysis
    phase_metrics = analyze_phase_metrics(
        state_traj, contact_sequence, mpc_dt, grf_traj
    )

    # GRF analysis
    grf_metrics = analyze_grf_profile(grf_traj, contact_sequence, robot_mass)

    # Actuator analysis
    actuator_metrics = analyze_actuator_saturation(joint_vel_traj, joint_torques_traj)

    # Task progress
    task_progress = compute_task_progress(command, trajectory_analysis)

    # Format everything
    return format_enhanced_feedback(
        iteration=iteration,
        command=command,
        optimization_status=optimization_status,
        simulation_results=simulation_results,
        trajectory_analysis=trajectory_analysis,
        phase_metrics=phase_metrics,
        grf_metrics=grf_metrics,
        actuator_metrics=actuator_metrics,
        task_progress=task_progress,
        previous_constraints=previous_constraints,
        previous_iteration_analysis=previous_iteration_analysis,
        initial_height=initial_height,
    )


def generate_failure_feedback(
    iteration: int,
    command: str,
    optimization_metrics: dict[str, Any],
    constraint_violations: dict[str, Any],
    trajectory_analysis: dict[str, Any] | None,
    previous_constraints: str,
    state_traj: np.ndarray | None = None,
    initial_height: float = 0.2117,
) -> str:
    """
    Generate feedback when optimization fails to converge.

    This provides the LLM with actionable information about why the optimization
    failed and what to try differently.

    Args:
        iteration: Current iteration number
        command: The task command
        optimization_metrics: Solver metrics (iterations, error messages)
        constraint_violations: Detailed constraint violation info from MPC
        trajectory_analysis: Analysis of the (infeasible) trajectory if available
        previous_constraints: The constraint code that failed
        state_traj: The debug trajectory from failed optimization (if available)
        initial_height: Robot's initial COM height from config

    Returns:
        Formatted feedback string for the LLM
    """
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append(f"ITERATION {iteration} - OPTIMIZATION FAILED")
    lines.append("=" * 60)
    lines.append(f"Task: {command}")

    # Solver status
    lines.append("\n" + "-" * 60)
    lines.append("SOLVER STATUS")
    lines.append("-" * 60)
    lines.append("❌ OPTIMIZATION DID NOT CONVERGE")

    if optimization_metrics.get("solver_iterations"):
        lines.append(
            f"  Solver stopped after {optimization_metrics['solver_iterations']} iterations"
        )

    if optimization_metrics.get("error_message"):
        lines.append(f"  Error: {optimization_metrics['error_message']}")

    if optimization_metrics.get("infeasibility_info"):
        lines.append(f"  Infeasibility: {optimization_metrics['infeasibility_info']}")

    # Constraint violations analysis
    lines.append("\n" + "-" * 60)
    lines.append("CONSTRAINT VIOLATION ANALYSIS")
    lines.append("-" * 60)

    if constraint_violations.get("terminal_constraints"):
        lines.append("\n⚠️ TERMINAL CONSTRAINT VIOLATIONS:")
        for violation in constraint_violations["terminal_constraints"]:
            lines.append(f"  • {violation}")

    if constraint_violations.get("state_bounds"):
        lines.append("\n⚠️ STATE BOUND VIOLATIONS:")
        # Only show first 5 to avoid overwhelming
        for violation in constraint_violations["state_bounds"][:5]:
            lines.append(f"  • {violation}")
        if len(constraint_violations["state_bounds"]) > 5:
            lines.append(
                f"  ... and {len(constraint_violations['state_bounds']) - 5} more"
            )

    if constraint_violations.get("llm_constraints"):
        lines.append("\n🔴 YOUR LLM CONSTRAINT VIOLATIONS:")
        # Show first 10 LLM constraint violations
        for violation in constraint_violations["llm_constraints"][:10]:
            lines.append(f"  • {violation}")
        if len(constraint_violations["llm_constraints"]) > 10:
            lines.append(
                f"  ... and {len(constraint_violations['llm_constraints']) - 10} more"
            )

    if constraint_violations.get("llm_summary"):
        lines.append("\nLLM CONSTRAINT SUMMARY:")
        for summary in constraint_violations["llm_summary"]:
            lines.append(f"  • {summary}")

    if constraint_violations.get("summary"):
        lines.append("\nSYSTEM CONSTRAINT SUMMARY:")
        for summary in constraint_violations["summary"]:
            lines.append(f"  • {summary}")

    # Trajectory analysis from the failed attempt (if available)
    if trajectory_analysis and state_traj is not None and state_traj.size > 0:
        # Check if we have non-zero trajectory data
        if np.any(state_traj != 0):
            lines.append("\n" + "-" * 60)
            lines.append("FAILED TRAJECTORY ANALYSIS (solver's last attempt)")
            lines.append("-" * 60)
            lines.append(
                f"  Height range: {trajectory_analysis.get('min_com_height', 0):.3f}m - "
                f"{trajectory_analysis.get('max_com_height', 0):.3f}m"
            )
            lines.append(
                f"  Final height: {trajectory_analysis.get('final_com_height', 0):.3f}m"
            )
            lines.append(
                f"  Max pitch: {trajectory_analysis.get('max_pitch', 0):.3f}rad "
                f"({trajectory_analysis.get('max_pitch', 0) * 57.3:.1f}°)"
            )
            lines.append(
                f"  Max yaw: {trajectory_analysis.get('max_yaw', 0):.3f}rad "
                f"({trajectory_analysis.get('max_yaw', 0) * 57.3:.1f}°)"
            )

            # Analyze what the solver was trying to do
            if trajectory_analysis.get("max_com_height", 0) > 0.5:
                lines.append("\n  ℹ️ Solver was attempting a high jump")
            if abs(trajectory_analysis.get("max_yaw", 0)) > 0.5:
                lines.append("\n  ℹ️ Solver was attempting significant yaw rotation")

    # Common failure patterns and fixes
    lines.append("\n" + "-" * 60)
    lines.append("LIKELY CAUSES & SUGGESTED FIXES")
    lines.append("-" * 60)

    # Analyze the violations to give specific advice
    violations_text = str(constraint_violations)

    if "terminal" in violations_text.lower() or "landing" in violations_text.lower():
        lines.append("\n🔧 TERMINAL STATE CONFLICT:")
        lines.append("  Your constraints may conflict with landing requirements.")
        lines.append("  The robot MUST land with:")
        lines.append("    - vz in [-0.5, 0.3] m/s (not falling too fast)")
        lines.append("    - roll, pitch in [-0.2, 0.2] rad (upright)")
        lines.append("    - angular velocities in [-0.5, 0.5] rad/s")
        lines.append("  FIX: Relax your constraints near the end of the trajectory")
        lines.append("       or only apply constraints during flight phase.")

    if "underground" in violations_text.lower() or "height" in violations_text.lower():
        lines.append("\n🔧 HEIGHT CONSTRAINT CONFLICT:")
        lines.append("  Your constraints may be forcing impossible heights.")
        lines.append("  FIX: Ensure height constraints are achievable given")
        lines.append("       the contact sequence and physics.")

    if not constraint_violations.get(
        "terminal_constraints"
    ) and not constraint_violations.get("state_bounds"):
        lines.append("\n🔧 LLM CONSTRAINT LIKELY INFEASIBLE:")
        lines.append("  No obvious system constraint violations detected.")
        lines.append("  Your custom constraints are likely the issue.")
        lines.append("  Common problems:")
        lines.append(
            f"    1. Constraints at k=0 don't allow initial state (height={initial_height:.4f}m)"
        )
        lines.append(
            "    2. Mutually exclusive bounds (e.g., height>0.5 AND height<0.3)"
        )
        lines.append("    3. Constraints too tight - try loosening by 20-50%")
        lines.append("    4. Wrong timing - check contact_k for stance vs flight")

    # General advice
    lines.append("\n" + "-" * 60)
    lines.append("GENERAL DEBUGGING TIPS")
    lines.append("-" * 60)
    lines.append("1. START SIMPLE: Use only 1-2 constraints, not many")
    lines.append(
        f"2. CHECK k=0: Your constraints MUST allow height={initial_height:.4f}m at k=0"
    )
    lines.append("3. USE PHASES: Apply constraints only during relevant phases")
    lines.append("   Example: if contact_k.sum() == 0:  # Only during flight")
    lines.append("4. LOOSEN BOUNDS: If in doubt, make bounds 2x wider")
    lines.append("5. FINAL STATE ONLY: Consider constraining only the final state")
    lines.append("   Example: if k == horizon - 1:  # Only at the end")

    # Initial state reminder
    lines.append("\n" + "-" * 60)
    lines.append("REMINDER: INITIAL STATE")
    lines.append("-" * 60)
    lines.append(
        f"Robot starts at: height={initial_height:.4f}m, roll=0, pitch=0, yaw=0"
    )
    lines.append("Constraints at k=0 MUST allow this state!")

    # Previous code
    lines.append("\n" + "-" * 60)
    lines.append("PREVIOUS CODE (FAILED)")
    lines.append("-" * 60)
    lines.append(previous_constraints)

    # Instructions
    lines.append("\n" + "=" * 60)
    lines.append("TASK: Fix the constraints based on this failure analysis.")
    lines.append("Consider a COMPLETELY DIFFERENT approach if needed.")
    lines.append("Return ONLY Python code.")
    lines.append("=" * 60)

    return "\n".join(lines)
