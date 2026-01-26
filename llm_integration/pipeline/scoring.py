"""Iteration scoring functions for the feedback pipeline."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .feedback_pipeline import FeedbackPipeline


def score_iteration(
    self: "FeedbackPipeline", iteration_result: dict[str, Any]
) -> float:
    """
    Score an iteration based on optimization success, simulation quality, AND task-specific behavior.

    Returns:
        Score between 0 and 1 (higher is better)
    """
    score = 0.0

    # Check if optimization succeeded (20% of score)
    if iteration_result.get("optimization", {}).get("success", False):
        score += 0.20

    # Check if simulation succeeded (20% of score)
    if iteration_result.get("simulation", {}).get("success", False):
        score += 0.20

    # Check simulation realism (20% of score)
    if iteration_result.get("simulation", {}).get("realistic", False):
        score += 0.20

    # Check trajectory quality based on tracking error (20% of score)
    tracking_error = iteration_result.get("simulation", {}).get(
        "tracking_error", float("inf")
    )
    if tracking_error < float("inf"):
        # Convert tracking error to score (lower error = higher score)
        error_score = max(0, 1 - min(tracking_error, 1.0))
        score += 0.20 * error_score

    # NEW: Task-specific behavior scoring (20% of score)
    task_score = score_task_specific_behavior(self, iteration_result)
    score += 0.20 * task_score

    return score


def score_task_specific_behavior(
    self: "FeedbackPipeline", iteration_result: dict[str, Any]
) -> float:
    """
    Score based on whether the robot actually performed the intended behavior.

    Returns:
        Score between 0 and 1 based on task-specific metrics
    """
    command = iteration_result.get("command", "").lower()
    trajectory_analysis = iteration_result.get("optimization", {}).get(
        "trajectory_analysis", {}
    )

    if not trajectory_analysis or "error" in trajectory_analysis:
        return 0.0

    try:
        # Extract key metrics
        height_gain = trajectory_analysis.get("height_gain", 0)
        total_pitch_rotation = abs(trajectory_analysis.get("total_pitch_rotation", 0))
        total_yaw_change = abs(trajectory_analysis.get("max_yaw", 0))
        com_displacement_x = abs(trajectory_analysis.get("com_displacement_x", 0))
        com_displacement_y = abs(trajectory_analysis.get("com_displacement_y", 0))

        # Task-specific scoring
        if any(word in command for word in ["backflip", "flip", "somersault"]):
            # Backflip: needs significant pitch rotation (close to 2π ≈ 6.28)
            target_rotation: float = 6.28  # 2π radians
            rotation_score: float = min(1.0, total_pitch_rotation / target_rotation)
            height_score: float = (
                min(1.0, height_gain / 0.3) if height_gain > 0.1 else 0
            )
            return 0.7 * rotation_score + 0.3 * height_score

        elif any(word in command for word in ["turn", "spin", "rotate", "around"]):
            # Turn around: needs yaw rotation (π for 180°, 2π for 360°)
            if "360" in command or "full" in command or "720" in command:
                target_rotation = (
                    6.28 if "360" in command else 12.56
                )  # 2π or 4π for 720°
            else:
                target_rotation = 3.14  # π for turn around (180°)

            yaw_score: float = min(1.0, total_yaw_change / target_rotation)
            # Penalize if it just jumps instead of turning
            if height_gain > 0.3 and yaw_score < 0.3:
                return 0.1  # Probably just jumped
            return yaw_score

        elif any(word in command for word in ["jump", "leap", "hop"]):
            if any(word in command for word in ["high", "up"]):
                # Jump high: needs significant height gain
                target_height = 0.4  # 40cm gain
                return float(
                    min(1.0, height_gain / target_height) if height_gain > 0.1 else 0
                )

            elif any(
                word in command
                for word in ["left", "right", "forward", "backward", "side"]
            ):
                # Directional jump: needs both height and displacement
                height_score = min(1.0, height_gain / 0.2) if height_gain > 0.05 else 0
                displacement = max(com_displacement_x, com_displacement_y)
                displacement_score = (
                    min(1.0, displacement / 0.3) if displacement > 0.1 else 0
                )
                return float(0.5 * height_score + 0.5 * displacement_score)

            else:
                # Generic jump: just needs height
                return float(min(1.0, height_gain / 0.2) if height_gain > 0.05 else 0)

        elif any(word in command for word in ["squat", "crouch", "lower"]):
            # Lowering motion: needs negative height change
            return float(min(1.0, abs(height_gain) / 0.1) if height_gain < -0.05 else 0)

        else:
            # Unknown command: give partial credit for any significant motion
            motion_score = 0.0
            if height_gain > 0.1:
                motion_score += 0.3
            if total_pitch_rotation > 0.5:
                motion_score += 0.3
            if total_yaw_change > 0.5:
                motion_score += 0.3
            if max(com_displacement_x, com_displacement_y) > 0.1:
                motion_score += 0.1
            return min(1.0, motion_score)

    except Exception as e:
        print(f"Warning: Task scoring failed: {e}")
        return 0.0
