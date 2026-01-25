"""Constraint generation system using LLM feedback."""

from typing import Any

import numpy as np

from . import prompts


class ConstraintGenerator:
    """
    Manages the system prompt and context for LLM constraint generation.
    Enhanced with robot-specific details and improved trajectory analysis.
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the constraint generator.

        Args:
            config: Optional robot configuration object. If provided, physical
                   facts are extracted from actual robot data for accuracy.
        """
        self.iteration_history: list[dict[str, Any]] = []
        self.config = config
        self.robot_details = self._get_robot_details()

    def _get_robot_details(self) -> dict[str, Any]:
        """Get robot-specific details from config or use sensible defaults."""
        # Default values (used if no config provided)
        details = {
            "mass": 15.0,
            "initial_height": 0.2117,
            "joint_limits_lower": [-0.8, -1.6, -2.6] * 4,
            "joint_limits_upper": [0.8, 1.6, -0.5] * 4,
        }

        if self.config is None:
            return details

        # Extract real values from config if available
        try:
            robot_data = getattr(self.config, "robot_data", None)
            experiment = getattr(self.config, "experiment", None)

            if robot_data and hasattr(robot_data, "mass"):
                details["mass"] = float(robot_data.mass)

            if experiment and hasattr(experiment, "initial_qpos"):
                initial_qpos = experiment.initial_qpos
                if len(initial_qpos) >= 3:
                    details["initial_height"] = float(initial_qpos[2])

            if robot_data and hasattr(robot_data, "joint_limits_lower"):
                details["joint_limits_lower"] = robot_data.joint_limits_lower.tolist()

            if robot_data and hasattr(robot_data, "joint_limits_upper"):
                details["joint_limits_upper"] = robot_data.joint_limits_upper.tolist()

        except Exception:
            # If anything fails, use defaults silently
            pass

        return details

    def get_system_prompt(self) -> str:
        """
        Get the system prompt that instructs the LLM on constraint generation.

        Returns:
            System prompt string with accurate robot details from config
        """
        # Pass actual robot details to the prompt generator
        base_prompt = prompts.get_system_prompt(
            mass=self.robot_details["mass"],
            initial_height=self.robot_details["initial_height"],
        )

        # Add robot-specific context
        robot_context = f"""

ROBOT PHYSICAL DETAILS:
- Mass: ~{self.robot_details["mass"]:.1f} kg
- Body: ~30cm x 20cm x 10cm
- Leg reach: ~30cm leg extension
- Joint limits: Hip: ±45°, Thigh: ±90°, Calf: ±150°
- Realistic jump height: ~0.5-0.8m
- Typical stance COM height: ~{self.robot_details["initial_height"]:.2f}m
- Foot spacing: Front/rear: ~30cm, Left/right: ~20cm

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
    ) -> dict[str, Any]:
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
                "initial_roll": float(euler_angles[0, 0]),
                "final_roll": float(euler_angles[-1, 0]),
                "total_roll_rotation": float(euler_angles[-1, 0] - euler_angles[0, 0]),
                "initial_pitch": float(euler_angles[0, 1]),
                "final_pitch": float(euler_angles[-1, 1]),
                "total_pitch_rotation": float(euler_angles[-1, 1] - euler_angles[0, 1]),
                "initial_yaw": float(euler_angles[0, 2]),
                "final_yaw": float(euler_angles[-1, 2]),
                "total_yaw_rotation": float(euler_angles[-1, 2] - euler_angles[0, 2]),
                "max_roll": float(np.max(np.abs(euler_angles[:, 0]))),
                "max_pitch": float(np.max(np.abs(euler_angles[:, 1]))),
                "max_yaw": float(
                    np.max(np.abs(euler_angles[:, 2] - euler_angles[0, 2]))
                ),
                # Velocity analysis
                "max_com_velocity": float(
                    np.max(np.linalg.norm(com_velocities, axis=1))
                ),
                "max_angular_vel": float(
                    np.max(np.linalg.norm(angular_velocities, axis=1))
                ),
                "final_com_velocity": float(np.linalg.norm(com_velocities[-1, :])),
                # Displacement analysis
                "com_displacement_x": float(com_positions[-1, 0] - com_positions[0, 0]),
                "com_displacement_y": float(com_positions[-1, 1] - com_positions[0, 1]),
                "total_distance": float(
                    np.sum(np.linalg.norm(np.diff(com_positions, axis=0), axis=1))
                ),
                # Timing
                "trajectory_duration": float(len(state_traj) * mpc_dt),
            }

            # Flight phase analysis
            initial_height = com_positions[0, 2]
            height_threshold = initial_height + 0.05  # 5cm above initial
            airborne_mask = com_positions[:, 2] > height_threshold

            if np.any(airborne_mask):
                flight_indices = np.where(airborne_mask)[0]
                metrics["flight_duration"] = float(len(flight_indices) * mpc_dt)
                metrics["flight_start_time"] = float(flight_indices[0] * mpc_dt)
                metrics["flight_peak_height"] = float(
                    np.max(com_positions[flight_indices, 2])
                )
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
                metrics["max_acceleration"] = float(
                    np.max(np.linalg.norm(com_accelerations, axis=1))
                )
            else:
                metrics["max_acceleration"] = 0.0

            return metrics

        except Exception as e:
            return {"error": f"Trajectory analysis failed: {str(e)}"}
