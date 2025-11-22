"""Constraint generation system using LLM feedback."""

from typing import Any, Dict, List

import numpy as np

from . import prompts


class ConstraintGenerator:
    """
    Manages the system prompt and context for LLM constraint generation.
    """

    def __init__(self) -> None:
        """Initialize the constraint generator."""
        self.iteration_history: List[Dict[str, Any]] = []

    def get_system_prompt(self) -> str:
        """
        Get the system prompt that instructs the LLM on constraint generation.

        Returns:
            System prompt string
        """
        return prompts.get_system_prompt()

    def get_user_prompt(self, command: str) -> str:
        """
        Create the initial user prompt from a natural language command.

        Args:
            command: Natural language command (e.g., "do a backflip")

        Returns:
            Formatted user prompt
        """
        return prompts.get_user_prompt(command)

    def create_feedback_context(
        self,
        iteration: int,
        trajectory_data: Dict[str, Any],
        optimization_status: Dict[str, Any],
        simulation_results: Dict[str, Any],
        previous_constraints: str,
    ) -> str:
        """
        Create feedback context for the next LLM iteration.

        Args:
            iteration: Current iteration number
            trajectory_data: Optimized trajectory information
            optimization_status: Solver status and convergence info
            simulation_results: Simulation execution results
            previous_constraints: Previously generated constraints

        Returns:
            Formatted feedback context string
        """
        # Store iteration history
        self.iteration_history.append(
            {
                "iteration": iteration,
                "trajectory_data": trajectory_data,
                "optimization_status": optimization_status,
                "simulation_results": simulation_results,
                "constraints": previous_constraints,
            }
        )

        return prompts.create_feedback_context(
            iteration,
            trajectory_data,
            optimization_status,
            simulation_results,
            previous_constraints,
        )

    def create_repair_prompt(
        self, command: str, failed_code: str, error_message: str, attempt_number: int
    ) -> str:
        """
        Create a prompt to ask the LLM to fix failed constraint code.

        Args:
            command: Original natural language command
            failed_code: The code that failed
            error_message: Error message from SafeExecutor/MPC
            attempt_number: Which attempt this is (1-10)

        Returns:
            Repair prompt string
        """
        return prompts.create_repair_prompt(
            command, failed_code, error_message, attempt_number
        )

    def analyze_trajectory(
        self, state_traj: np.ndarray, mpc_dt: float
    ) -> Dict[str, Any]:
        """
        Analyze trajectory to extract key metrics for feedback.

        Args:
            state_traj: State trajectory array (N x 24)
            mpc_dt: Time step

        Returns:
            Dictionary of trajectory metrics
        """
        if state_traj.shape[0] == 0:
            return {}

        # Extract key state components
        com_positions = state_traj[:, 0:3]  # x, y, z
        com_velocities = state_traj[:, 3:6]
        euler_angles = state_traj[:, 6:9]  # roll, pitch, yaw
        angular_velocities = state_traj[:, 9:12]

        # Calculate metrics
        metrics = {
            "max_com_height": np.max(com_positions[:, 2]),
            "min_com_height": np.min(com_positions[:, 2]),
            "final_pitch": euler_angles[-1, 1],
            "initial_pitch": euler_angles[0, 1],
            "total_pitch_rotation": euler_angles[-1, 1] - euler_angles[0, 1],
            "max_angular_vel": np.max(np.linalg.norm(angular_velocities, axis=1)),
            "trajectory_duration": len(state_traj) * mpc_dt,
            "max_com_velocity": np.max(np.linalg.norm(com_velocities, axis=1)),
        }

        # Estimate flight duration (when COM z-velocity > 0 and height > initial height)
        initial_height = com_positions[0, 2]
        airborne_mask = (com_positions[:, 2] > initial_height + 0.05) & (
            com_velocities[:, 2] > -0.1
        )
        flight_duration = np.sum(airborne_mask) * mpc_dt
        metrics["flight_duration"] = flight_duration

        return metrics
