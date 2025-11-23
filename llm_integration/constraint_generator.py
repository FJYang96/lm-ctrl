"""Constraint generation system using LLM feedback."""

from typing import Any, Dict, List

import numpy as np

from . import prompts


class ConstraintGenerator:
    """
    Manages the system prompt and context for LLM constraint generation.
    Enhanced with robot-specific details and improved trajectory analysis.
    """

    def __init__(self) -> None:
        """Initialize the constraint generator."""
        self.iteration_history: List[Dict[str, Any]] = []
        self.robot_details = self._get_robot_details()

    def _get_robot_details(self) -> Dict[str, Any]:
        """Get robot-specific details for enhanced prompts."""
        return {
            "mass": "~12 kg",
            "dimensions": "Body: ~30cm x 20cm x 10cm",
            "leg_reach": "~30cm leg extension",
            "joint_limits": "Hip: ±45°, Thigh: ±90°, Calf: ±150°",
            "max_jump_height": "~0.5-0.8m realistic",
            "typical_stance_height": "~0.25m COM height",
            "foot_spacing": "Front/rear: ~30cm, Left/right: ~20cm"
        }

    def get_system_prompt(self) -> str:
        """
        Get the system prompt that instructs the LLM on constraint generation.

        Returns:
            System prompt string with robot details
        """
        base_prompt = prompts.get_system_prompt()

        # Add robot-specific context
        robot_context = f"""

ROBOT PHYSICAL DETAILS:
- Mass: {self.robot_details['mass']}
- {self.robot_details['dimensions']}
- Leg reach: {self.robot_details['leg_reach']}
- Joint limits: {self.robot_details['joint_limits']}
- Realistic jump height: {self.robot_details['max_jump_height']}
- Typical stance COM height: {self.robot_details['typical_stance_height']}
- Foot spacing: {self.robot_details['foot_spacing']}

Use these physical limits to create realistic constraints."""

        return base_prompt + robot_context

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
        Enhanced trajectory analysis to extract key metrics for feedback.

        Args:
            state_traj: State trajectory array (N x 24)
            mpc_dt: Time step

        Returns:
            Dictionary of comprehensive trajectory metrics
        """
        if state_traj.shape[0] == 0:
            return {"error": "Empty trajectory"}

        try:
            # Extract key state components
            com_positions = state_traj[:, 0:3]  # x, y, z
            com_velocities = state_traj[:, 3:6]  # vx, vy, vz
            euler_angles = state_traj[:, 6:9]  # roll, pitch, yaw
            angular_velocities = state_traj[:, 9:12]  # wx, wy, wz
            joint_angles = state_traj[:, 12:24]  # 12 joint angles

            # Basic trajectory metrics
            metrics = {
                # Height analysis
                "max_com_height": float(np.max(com_positions[:, 2])),
                "min_com_height": float(np.min(com_positions[:, 2])),
                "initial_com_height": float(com_positions[0, 2]),
                "final_com_height": float(com_positions[-1, 2]),
                "height_gain": float(np.max(com_positions[:, 2]) - com_positions[0, 2]),
                
                # Orientation analysis
                "initial_pitch": float(euler_angles[0, 1]),
                "final_pitch": float(euler_angles[-1, 1]),
                "total_pitch_rotation": float(euler_angles[-1, 1] - euler_angles[0, 1]),
                "max_roll": float(np.max(np.abs(euler_angles[:, 0]))),
                "max_pitch": float(np.max(np.abs(euler_angles[:, 1]))),
                "max_yaw": float(np.max(
                    np.abs(euler_angles[:, 2] - euler_angles[0, 2]))),

                # Velocity analysis
                "max_com_velocity": float(np.max(
                    np.linalg.norm(com_velocities, axis=1))),
                "max_angular_vel": float(np.max(
                    np.linalg.norm(angular_velocities, axis=1))),
                "final_com_velocity": float(np.linalg.norm(com_velocities[-1, :])),
                
                # Displacement analysis
                "com_displacement_x": float(
                    com_positions[-1, 0] - com_positions[0, 0]),
                "com_displacement_y": float(
                    com_positions[-1, 1] - com_positions[0, 1]),
                "total_distance": float(np.sum(
                    np.linalg.norm(np.diff(com_positions, axis=0), axis=1))),

                # Timing
                "trajectory_duration": float(len(state_traj) * mpc_dt),
            }

            # Flight phase analysis
            initial_height = com_positions[0, 2]
            height_threshold = initial_height + 0.05  # 5cm above initial
            airborne_mask = (com_positions[:, 2] > height_threshold)

            if np.any(airborne_mask):
                flight_indices = np.where(airborne_mask)[0]
                metrics["flight_duration"] = float(len(flight_indices) * mpc_dt)
                metrics["flight_start_time"] = float(flight_indices[0] * mpc_dt)
                metrics["flight_peak_height"] = float(
                    np.max(com_positions[flight_indices, 2]))
            else:
                metrics["flight_duration"] = 0.0
                metrics["flight_start_time"] = 0.0
                metrics["flight_peak_height"] = metrics["max_com_height"]

            # Joint motion analysis
            joint_ranges = np.max(joint_angles, axis=0) - np.min(joint_angles, axis=0)
            metrics["max_joint_range"] = float(np.max(joint_ranges))
            metrics["avg_joint_range"] = float(np.mean(joint_ranges))

            # Smoothness metrics
            com_accelerations = np.diff(com_velocities, axis=0) / mpc_dt
            if com_accelerations.shape[0] > 0:
                metrics["max_acceleration"] = float(np.max(
                    np.linalg.norm(com_accelerations, axis=1)))
            else:
                metrics["max_acceleration"] = 0.0

            return metrics

        except Exception as e:
            return {"error": f"Trajectory analysis failed: {str(e)}"}
