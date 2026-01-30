"""Task progress computation for enhanced feedback."""

from typing import Any


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
