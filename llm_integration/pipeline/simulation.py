"""Simulation execution functions for the feedback pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from utils.visualization import render_planned_trajectory

from ..logging_config import logger

if TYPE_CHECKING:
    from .feedback_pipeline import FeedbackPipeline


def get_video_dir(run_dir: Path) -> Path:
    """Get the video directory, using VIDEO_DIR env var if set, otherwise run_dir."""
    video_dir = os.environ.get("VIDEO_DIR")
    if video_dir:
        video_path = Path(video_dir)
        video_path.mkdir(parents=True, exist_ok=True)
        return video_path
    return run_dir


def execute_simulation(
    self: FeedbackPipeline,
    optimization_result: dict[str, Any],
    iteration: int,
    run_dir: Path,
) -> dict[str, Any]:
    """Render the planned trajectory from MPC optimization."""

    if not optimization_result["success"]:
        # Still render the debug trajectory for visualization/debugging
        return render_failed_trajectory(self, optimization_result, iteration, run_dir)

    if self.env is None:
        return {
            "success": False,
            "error": "Cannot render - environment not available",
        }

    try:
        state_traj = optimization_result["state_trajectory"]
        joint_vel_traj = optimization_result["joint_vel_trajectory"]

        # Render planned trajectory
        planned_traj_images = render_planned_trajectory(
            state_traj, joint_vel_traj, self.env
        )

        # Save planned trajectory video to run dir
        if planned_traj_images:
            import imageio

            video_dir = get_video_dir(run_dir)
            mpc_dt = (
                self.current_task_mpc.mpc_dt
                if self.current_task_mpc is not None
                else self._last_mpc_dt
            )
            fps = 1 / mpc_dt
            video_path = video_dir / f"planned_traj_iter_{iteration}.mp4"
            imageio.mimsave(str(video_path), planned_traj_images, fps=fps)

        return {
            "success": True,
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Rendering failed: {str(e)}",
        }


def render_failed_trajectory(
    self: FeedbackPipeline,
    optimization_result: dict[str, Any],
    iteration: int,
    run_dir: Path,
) -> dict[str, Any]:
    """
    Render the debug trajectory from a failed optimization for debugging.

    Even when the solver doesn't converge, the debug trajectory shows what
    the solver was attempting. This is invaluable for understanding if the
    approach was on the right track (e.g., 77% of a backflip).
    """

    if self.env is None:
        return {
            "success": False,
            "error": "Cannot render - environment not available",
            "debug_video_saved": False,
        }

    state_traj = optimization_result.get("state_trajectory")
    if state_traj is None or state_traj.size == 0:
        return {
            "success": False,
            "error": "Optimization failed - no debug trajectory available",
            "debug_video_saved": False,
        }

    # Check if trajectory has any meaningful data (not all zeros)
    if np.allclose(state_traj, 0):
        return {
            "success": False,
            "error": "Optimization failed - debug trajectory is all zeros",
            "debug_video_saved": False,
        }

    try:
        joint_vel_traj = optimization_result.get("joint_vel_trajectory")
        if joint_vel_traj is None:
            return {
                "success": False,
                "error": "Optimization failed - no joint velocity trajectory available",
                "debug_video_saved": False,
            }

        # Render the debug trajectory (what solver was attempting)
        logger.info("Rendering debug trajectory from failed optimization...")
        debug_traj_images = render_planned_trajectory(
            state_traj, joint_vel_traj, self.env
        )

        # Get the video directory (uses VIDEO_DIR env var if set)
        video_dir = get_video_dir(run_dir)

        video_path = None
        if debug_traj_images and len(debug_traj_images) > 0:
            import imageio

            mpc_dt = (
                self.current_task_mpc.mpc_dt
                if self.current_task_mpc is not None
                else self._last_mpc_dt
            )
            fps = 1 / mpc_dt
            video_path = video_dir / f"debug_trajectory_iter_{iteration}.mp4"
            imageio.mimsave(str(video_path), debug_traj_images, fps=fps)
            logger.info(f"Saved debug trajectory video: {video_path}")

        trajectory_analysis = optimization_result.get("trajectory_analysis", {})
        pitch_achieved = abs(trajectory_analysis.get("total_pitch_rotation", 0))
        height_gain = trajectory_analysis.get("height_gain", 0)

        return {
            "success": False,  # Still marked as failed (optimization didn't converge)
            "error": "Optimization failed - debug trajectory rendered for analysis",
            "debug_video_saved": video_path is not None,
            "debug_video_path": str(video_path) if video_path else None,
            "debug_trajectory_metrics": {
                "pitch_achieved_rad": pitch_achieved,
                "pitch_achieved_pct": (pitch_achieved / 6.28) * 100
                if pitch_achieved > 0
                else 0,
                "height_gain_m": height_gain,
            },
        }

    except Exception as e:
        logger.warning(f"Could not render debug trajectory: {e}")
        return {
            "success": False,
            "error": f"Optimization failed, debug render also failed: {e}",
            "debug_video_saved": False,
        }
