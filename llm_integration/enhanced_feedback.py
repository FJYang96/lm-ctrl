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
    pitch = state_traj[:, 7]  # Pitch angle
    yaw = state_traj[:, 8]  # Yaw angle
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
            "avg_pitch_rate": float(np.mean(np.abs(pitch_rate[start_idx:end_idx]))),
            "avg_yaw_rate": float(np.mean(np.abs(yaw_rate[start_idx:end_idx]))),
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
) -> str:
    """
    Format all feedback into a structured string for the LLM.

    Returns:
        Formatted feedback string
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"ITERATION {iteration} FEEDBACK")
    lines.append("=" * 60)

    # Optimization status
    if optimization_status.get("converged", False):
        lines.append("\n✅ OPTIMIZATION: SUCCESS")
    else:
        lines.append("\n❌ OPTIMIZATION: FAILED")
        lines.append(
            "Your constraints are likely MUTUALLY EXCLUSIVE or PHYSICALLY IMPOSSIBLE."
        )
        lines.append(
            "Fix: SIMPLIFY and LOOSEN constraints. Start with just one key constraint."
        )

    # Simulation status
    if simulation_results.get("success", False):
        tracking_error = simulation_results.get("tracking_error", 0)
        lines.append(f"✅ SIMULATION: SUCCESS (tracking error: {tracking_error:.3f})")
    else:
        lines.append("❌ SIMULATION: FAILED")

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
                f"  Pitch rate: {fp.get('avg_pitch_rate', 0):.2f} rad/s, "
                f"Yaw rate: {fp.get('avg_yaw_rate', 0):.2f} rad/s"
            )
            lines.append(
                f"  Pitch change: {fp.get('total_pitch_change', 0):.2f} rad "
                f"({fp.get('total_pitch_change', 0)*57.3:.0f}°), "
                f"Yaw change: {fp.get('total_yaw_change', 0):.2f} rad "
                f"({fp.get('total_yaw_change', 0)*57.3:.0f}°)"
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
        if "max_velocity_ratio" in actuator_metrics:
            vel_pct = actuator_metrics["max_velocity_ratio"] * 100
            status = "⚠️ NEAR LIMIT" if vel_pct > 90 else "OK"
            lines.append(
                f"Joint velocity utilization: {vel_pct:.0f}% of limit {status}"
            )

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
) -> str:
    """
    Generate comprehensive enhanced feedback for the LLM.

    This is the main entry point that combines all analysis.

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
    )
