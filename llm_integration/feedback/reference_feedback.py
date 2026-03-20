"""Reference trajectory metrics computation."""

from __future__ import annotations

from typing import Any

import numpy as np


def _compute_reference_metrics(
    ref_trajectory_data: dict[str, Any] | None,
    state_trajectory: np.ndarray | None,
    mpc_dt: float | None = None,
) -> str:
    """Compute metrics comparing reference trajectory to actual result.

    Returns a formatted text string with RMSE and plausibility checks.
    """
    if mpc_dt is None:
        raise ValueError(
            "_compute_reference_metrics: 'mpc_dt' must be explicitly provided "
            "(go2_config.mpc_config.mpc_dt may be stale after restore_base_config)."
        )
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
    if len(ref_vz_diff) > 0:
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
