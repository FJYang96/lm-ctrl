"""Reference-trajectory-specific LLM feedback generation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..logging_config import logger
from .llm_evaluation import get_evaluator


def _compute_reference_metrics(
    ref_trajectory_data: dict[str, Any] | None,
    state_trajectory: np.ndarray | None,
    mpc_dt: float = 0.02,
) -> str:
    """Compute metrics comparing reference trajectory to actual result.

    Returns a formatted text string with RMSE and plausibility checks.
    """
    if ref_trajectory_data is None:
        return "No reference trajectory data available."

    lines = []
    X_ref = ref_trajectory_data.get("X_ref")
    if X_ref is None:
        return "Reference trajectory X_ref not available."

    lines.append("Reference trajectory shape: " + str(X_ref.shape))

    # Height comparison
    ref_height = X_ref[2, :]
    lines.append(
        f"Ref height: min={ref_height.min():.3f}m, max={ref_height.max():.3f}m, "
        f"final={ref_height[-1]:.3f}m"
    )

    # Pitch comparison
    ref_pitch = X_ref[7, :]
    lines.append(
        f"Ref pitch: min={ref_pitch.min():.2f}rad, max={ref_pitch.max():.2f}rad, "
        f"final={ref_pitch[-1]:.2f}rad "
        f"({ref_pitch[-1] * 57.3:.0f} deg)"
    )

    # Velocity comparison
    ref_vz = X_ref[5, :]
    lines.append(
        f"Ref vertical velocity: min={ref_vz.min():.2f}, max={ref_vz.max():.2f} m/s"
    )

    # RMSE against actual trajectory
    if state_trajectory is not None:
        X_actual = state_trajectory.T  # (states_dim, horizon+1)
        # Match shapes
        min_cols = min(X_ref.shape[1], X_actual.shape[1])

        # Height RMSE
        height_rmse = np.sqrt(
            np.mean((X_ref[2, :min_cols] - X_actual[2, :min_cols]) ** 2)
        )
        lines.append(f"Height RMSE (ref vs actual): {height_rmse:.4f}m")

        # Pitch RMSE
        pitch_rmse = np.sqrt(
            np.mean((X_ref[7, :min_cols] - X_actual[7, :min_cols]) ** 2)
        )
        lines.append(
            f"Pitch RMSE (ref vs actual): {pitch_rmse:.4f}rad "
            f"({pitch_rmse * 57.3:.1f} deg)"
        )

        # Vertical velocity RMSE
        vz_rmse = np.sqrt(np.mean((X_ref[5, :min_cols] - X_actual[5, :min_cols]) ** 2))
        lines.append(f"Vz RMSE (ref vs actual): {vz_rmse:.4f} m/s")

    # Raw plausibility metrics (LLM interprets these)
    lines.append("\nPlausibility metrics:")
    ref_vz_diff = np.diff(ref_vz)
    lines.append(
        f"  Max vertical velocity increase between timesteps: {ref_vz_diff.max():.4f} m/s"
    )
    lines.append(
        f"  Max vertical velocity decrease between timesteps: {ref_vz_diff.min():.4f} m/s"
    )

    ref_z_diff = np.diff(ref_height)
    if X_ref.shape[1] > 1:
        expected_dz = ref_vz[:-1] * mpc_dt
        max_inconsistency = float(np.max(np.abs(ref_z_diff - expected_dz)))
        lines.append(
            f"  Position-velocity consistency (max |dz - vz*dt|): {max_inconsistency:.4f}m"
        )

    return "\n".join(lines)


def generate_reference_feedback(
    command: str,
    constraint_code: str,
    images: list[str] | None,
    visual_summary: str,
    ref_trajectory_data: dict[str, Any] | None,
    trajectory_analysis: dict[str, Any],
    state_trajectory: np.ndarray | None,
    opt_success: bool,
    pivot_signal: str | None,
    mpc_dt: float = 0.02,
) -> str:
    """Generate reference-trajectory-specific feedback via LLM.

    Args:
        command: The task command
        constraint_code: Full constraint code (for context)
        images: Video frames from the trajectory
        visual_summary: Text summary of the video frames
        ref_trajectory_data: Dict with X_ref, U_ref arrays
        trajectory_analysis: Trajectory metrics dict
        state_trajectory: Actual state trajectory (horizon+1, states_dim)
        opt_success: Whether the solver converged
        pivot_signal: "pivot", "tweak", or None
        mpc_dt: MPC time step in seconds

    Returns:
        Multi-paragraph analysis text for the code-gen LLM
    """
    system_prompt = """You are an expert analyzing reference trajectory design for quadruped MPC trajectory optimization.

The reference trajectory is used ONLY as an initial guess (warmstart) for the solver.
It does NOT change the cost function — the phase-aware cost and slack constraints remain unchanged.
A good reference helps the solver converge faster and find better solutions.

=== CONSTRAINT-REFERENCE INTERPLAY ===

The reference trajectory should sit roughly in the CENTER of the constraint bounds.
If constraints force a specific rotation, the reference must show that rotation.
If constraints define a flight phase, the reference must have ballistic motion during that phase.
Phase timing in the reference must match the contact sequence exactly.

Key principles:
- Reference should be physically plausible (respect gravity, momentum conservation)
- Velocities must be consistent with positions (no teleportation)
- GRF should be zero during flight phases, ~mg/n_feet during stance
- Angular velocity during flight should be constant (momentum conservation)
- Angles should integrate from angular velocities

=== PLAUSIBILITY DATA ===

You will receive raw plausibility metrics: velocity changes between timesteps, position-velocity
consistency values, and RMSE comparisons. YOU must interpret these numbers to determine whether
the reference trajectory is physically plausible. There are no pre-classified warnings or OK labels.
For example, a large positive vertical velocity increase may indicate a gravity violation during flight.
A high position-velocity inconsistency may indicate the reference has discontinuities.

=== OUTPUT FORMAT ===

Write multi-paragraph analysis. Be specific about:
1. How well the reference matches the task requirements
2. Physics plausibility assessment from the raw metrics (gravity, momentum, velocity-position consistency)
3. Phase timing alignment with contact sequence
4. Specific parameter changes (peak height, rotation rate, timing) with concrete numbers

Do NOT return JSON. Return readable analysis text."""

    mode_text = ""
    if pivot_signal == "pivot":
        mode_text = """MODE: MANDATORY PIVOT
The current approach has stagnated. Suggest a fundamentally different reference trajectory
shape — different peak values, different phase timing, different interpolation strategy."""
    elif pivot_signal == "tweak":
        mode_text = """MODE: ADJUSTMENT SUGGESTED
The current reference shows some promise. Suggest incremental changes — adjust peak values,
shift timing, tune interpolation parameters."""
    else:
        mode_text = """MODE: FIRST ITERATION
This is the first attempt. Analyze the reference trajectory design and suggest improvements."""

    # Compute reference metrics
    ref_metrics = _compute_reference_metrics(
        ref_trajectory_data, state_trajectory, mpc_dt
    )

    # Format trajectory metrics
    metrics_text = "No trajectory data available"
    if trajectory_analysis:
        ta = trajectory_analysis
        metrics_text = (
            f"Height: initial={ta.get('initial_com_height', 0):.3f}m, "
            f"max={ta.get('max_com_height', 0):.3f}m, gain={ta.get('height_gain', 0):.3f}m\n"
            f"Pitch: {ta.get('total_pitch_rotation', 0):.2f} rad "
            f"({abs(ta.get('total_pitch_rotation', 0)) * 57.3:.0f} deg)\n"
            f"Flight duration: {ta.get('flight_duration', 0):.2f}s\n"
            f"Max angular velocity: {ta.get('max_angular_vel', 0):.2f} rad/s\n"
            f"Final COM velocity: {ta.get('final_com_velocity', 0):.2f} m/s"
        )

    user_message = f"""COMMAND: {command}

{mode_text}

SOLVER STATUS: {"CONVERGED" if opt_success else "FAILED"}

CONSTRAINT CODE (for context):
```python
{constraint_code}
```

ACTUAL TRAJECTORY METRICS:
{metrics_text}

REFERENCE TRAJECTORY ANALYSIS:
{ref_metrics}

VISUAL SUMMARY:
{visual_summary if visual_summary else "Not available"}

Provide targeted feedback on the reference trajectory design."""

    try:
        evaluator = get_evaluator()
        response = evaluator._call_llm(system_prompt, user_message, images)
        return response.strip()
    except Exception as e:
        logger.error(f"Reference feedback generation failed: {e}")
        return ""
