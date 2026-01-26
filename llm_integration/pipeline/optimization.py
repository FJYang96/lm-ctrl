"""Trajectory optimization functions for the feedback pipeline."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from utils import conversion
from utils.simulation import create_reference_trajectory

if TYPE_CHECKING:
    from .feedback_pipeline import FeedbackPipeline


def solve_trajectory_optimization(
    self: "FeedbackPipeline",
    mpc_config_code: str,
    task_name: str,
    iteration: int,
    run_dir: Path,
) -> dict[str, Any]:
    """Solve trajectory optimization with LLM-configured MPC."""

    if self.current_task_mpc is None:
        return {
            "success": False,
            "error": "No LLM MPC configured for this iteration",
            "converged": False,
        }

    # Print the MPC configuration for debugging
    print("📝 Generated MPC Configuration:")
    print("=" * 50)
    print(mpc_config_code)
    print("=" * 50)

    # Show MPC configuration summary
    config_summary = self.current_task_mpc.get_configuration_summary()
    print("🔧 MPC Configuration Summary:")
    print(f"  Task: {config_summary['task_name']}")
    print(f"  Duration: {config_summary['duration']:.2f}s")
    print(f"  Horizon: {config_summary['horizon']} steps")
    print(f"  Constraints: {config_summary['num_constraints']}")
    print(f"  Contact Phases: {len(config_summary['contact_phases'])}")
    for phase in config_summary.get("contact_phases", []):
        print(
            f"    {phase['phase_type']}: {phase['start_time']:.2f}-{phase['start_time'] + phase['duration']:.2f}s"
        )

    # Setup initial conditions
    initial_state, _ = conversion.sim_to_mpc(
        self.config.experiment.initial_qpos, self.config.experiment.initial_qvel
    )

    ref = create_reference_trajectory(self.config.experiment.initial_qpos)

    try:
        # Solve trajectory optimization with LLM-configured MPC
        print("⚙️  Solving trajectory optimization with LLM MPC...")
        state_traj, grf_traj, joint_vel_traj, status = (
            self.current_task_mpc.solve_trajectory(initial_state, ref)
        )

        if status == 0:
            print("✅ Optimization converged successfully!")
        else:
            print(f"❌ Optimization failed with status: {status}")

    except Exception as e:
        print(f"❌ LLM MPC failed: {e}")
        print("🔄 Falling back to default MPC...")

        # Fallback to default MPC
        try:
            state_traj, grf_traj, joint_vel_traj, status = (
                self.fallback_mpc.solve_trajectory(
                    initial_state, ref, self.config.mpc_config.contact_sequence
                )
            )
            print(f"Fallback MPC status: {status}")
        except Exception as fallback_error:
            return {
                "success": False,
                "error": f"Both LLM MPC and fallback failed: {e}, {fallback_error}",
                "converged": False,
            }

    # Create metrics dict
    metrics = {
        "converged": status == 0,
        "status": "success" if status == 0 else "failed",
        "objective_value": 0.0,
        "mpc_type": "llm_configured"
        if self.current_task_mpc is not None
        else "fallback",
        "config_summary": config_summary,
        # Solver info for feedback
        "solver_iterations": getattr(self.current_task_mpc, "solver_iterations", None)
        if self.current_task_mpc
        else None,
        "error_message": getattr(self.current_task_mpc, "last_error", None)
        if self.current_task_mpc and status != 0
        else None,
        "infeasibility_info": getattr(self.current_task_mpc, "infeasibility_info", None)
        if self.current_task_mpc and status != 0
        else None,
    }

    # Analyze trajectory using LLM MPC time step
    mpc_dt = config_summary.get("time_step", self.config.mpc_config.mpc_dt)
    trajectory_analysis = self.constraint_generator.analyze_trajectory(
        state_traj, mpc_dt
    )

    # Print trajectory analysis
    if status == 0:
        print("📊 Trajectory Analysis:")
        print(f"  Max Height: {trajectory_analysis.get('max_com_height', 0):.3f}m")
        print(f"  Height Gain: {trajectory_analysis.get('height_gain', 0):.3f}m")
        print(
            f"  Pitch Rotation: {trajectory_analysis.get('total_pitch_rotation', 0):.3f}rad ({trajectory_analysis.get('total_pitch_rotation', 0) * 180 / 3.14159:.1f}°)"
        )
        print(
            f"  Yaw Change: {trajectory_analysis.get('max_yaw', 0):.3f}rad ({trajectory_analysis.get('max_yaw', 0) * 180 / 3.14159:.1f}°)"
        )
        print(
            f"  X Displacement: {trajectory_analysis.get('com_displacement_x', 0):.3f}m"
        )
        print(
            f"  Y Displacement: {trajectory_analysis.get('com_displacement_y', 0):.3f}m"
        )
        print(
            f"  Flight Duration: {trajectory_analysis.get('flight_duration', 0):.3f}s"
        )

    # Save trajectory data
    np.save(run_dir / f"state_traj_iter_{iteration}.npy", state_traj)
    np.save(run_dir / f"grf_traj_iter_{iteration}.npy", grf_traj)
    np.save(run_dir / f"joint_vel_traj_iter_{iteration}.npy", joint_vel_traj)

    result = {
        "success": status == 0,
        "status": status,
        "converged": status == 0,
        "state_trajectory": state_traj,
        "grf_trajectory": grf_traj,
        "joint_vel_trajectory": joint_vel_traj,
        "optimization_metrics": metrics,
        "trajectory_analysis": trajectory_analysis,
        "mpc_config_valid": True,
        "task_name": task_name,
        "mpc_config_code": mpc_config_code,
    }

    return result
