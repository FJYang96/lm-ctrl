"""Main feedback pipeline implementing Algorithm 1 from the PDF."""

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from mpc.dynamics.model import KinoDynamic_Model
from mpc.mpc_opti import QuadrupedMPCOpti
from utils import conversion
from utils.inv_dyn import compute_joint_torques
from utils.simulation import create_reference_trajectory, simulate_trajectory
from utils.visualization import render_and_save_planned_trajectory

from .constraint_generator import ConstraintGenerator
from .llm_client import LLMClient
from .llm_mpc import LLMTaskMPC
from .safe_executor import SafeConstraintExecutor

try:
    from gym_quadruped.quadruped_env import QuadrupedEnv
except ImportError:
    print("Warning: gym_quadruped not available. Simulation features may be limited.")
    QuadrupedEnv = None
import config


class FeedbackPipeline:
    """
    Main pipeline implementing the iterative LLM refinement loop (Algorithm 1).

    This implements:
    1. LLM constraint generation
    2. Trajectory optimization with generated constraints
    3. Simulation execution
    4. Feedback collection and iteration
    """

    def __init__(self, config_obj: Any = None):
        """
        Initialize the feedback pipeline.

        Args:
            config_obj: Configuration object (uses default config if None)
        """
        self.config = config_obj if config_obj is not None else config

        # Initialize components
        self.llm_client = LLMClient()
        self.constraint_generator = ConstraintGenerator()
        self.safe_executor = SafeConstraintExecutor()

        # Initialize kinodynamic model
        self.kindyn_model = KinoDynamic_Model(self.config)

        # Initialize LLM-specific MPC (replaces the fixed MPC)
        self.llm_mpc = LLMTaskMPC(self.kindyn_model, self.config)

        # Legacy MPC for fallback (in case LLM MPC fails)
        self.fallback_mpc = QuadrupedMPCOpti(
            model=self.kindyn_model, config=self.config, build=True
        )

        # Task-specific MPC tracking
        self.current_task_mpc: LLMTaskMPC | None = None
        self.llm_mpc_code: str = ""

        # LLM constraint tracking
        self.llm_constraints: list[Any] = []

        # MPC reference for constraint injection
        self.mpc: Any = None

        # Initialize simulation environment
        if QuadrupedEnv is not None:
            self.env = QuadrupedEnv(
                robot=self.config.robot,
                scene="flat",
                ground_friction_coeff=self.config.experiment.mu_ground,
                state_obs_names=QuadrupedEnv._DEFAULT_OBS + ("contact_forces:base",),
                sim_dt=self.config.experiment.sim_dt,
            )
        else:
            print("Warning: Simulation environment not available")
            self.env = None

        # Pipeline state
        self.iteration_results: list[dict[str, Any]] = []
        self.max_iterations = int(os.getenv("MAX_LLM_ITERATIONS", "5"))
        self.results_dir = Path(os.getenv("RESULTS_DIR", "results/llm_iterations"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_pipeline(self, command: str) -> dict[str, Any]:
        """
        Run the complete LLM feedback pipeline for a given command.

        Args:
            command: Natural language command (e.g., "do a backflip")

        Returns:
            Dictionary containing complete pipeline results
        """
        print(f"Starting LLM Feedback Pipeline for command: '{command}'")

        # Create results directory for this run
        timestamp = int(time.time())
        run_dir = self.results_dir / f"{command.replace(' ', '_')}_{timestamp}"
        run_dir.mkdir(exist_ok=True)

        # Initialize pipeline state
        self.iteration_results = []
        context = None
        best_result = None
        best_score = -float("inf")

        # Algorithm 1: Iterative Refinement Pipeline
        system_prompt = self.constraint_generator.get_system_prompt()
        initial_user_message = self.constraint_generator.get_user_prompt(command)

        for iteration in range(1, self.max_iterations + 1):
            print(f"\\n=== ITERATION {iteration} ===")

            try:
                # Step 3: Generate constraints using LLM with auto-retry
                print("Generating constraints with LLM...")
                constraint_code, function_name, attempt_log = (
                    self._generate_constraints_with_retry(
                        system_prompt, initial_user_message, context, command
                    )
                )

                # Step 4: Solve optimization problem with new constraints
                print("Solving trajectory optimization...")
                optimization_result = self._solve_trajectory_optimization(
                    constraint_code, function_name, iteration, run_dir
                )

                # Step 5: Execute trajectory in simulation
                print("Executing trajectory in simulation...")
                simulation_result = self._execute_simulation(
                    optimization_result, iteration, run_dir
                )

                # Collect iteration results
                iteration_result = {
                    "iteration": iteration,
                    "command": command,
                    "constraint_code": constraint_code,
                    "function_name": function_name,
                    "attempt_log": attempt_log,
                    "optimization": optimization_result,
                    "simulation": simulation_result,
                    "timestamp": time.time(),
                }

                self.iteration_results.append(iteration_result)

                # Save iteration results
                self._save_iteration_results(iteration_result, run_dir)

                # Step 6: Create feedback for next iteration
                if iteration < self.max_iterations:
                    print("Generating feedback for next iteration...")
                    context = self._create_feedback_context(
                        iteration,
                        optimization_result,
                        simulation_result,
                        constraint_code,
                    )

                # Track best result based on success score
                score = self._score_iteration(iteration_result)
                if score > best_score:
                    best_score = score
                    best_result = iteration_result

                print(f"Iteration {iteration} score: {score:.3f}")

                # Early stopping if we achieve excellent results
                # Raised threshold for task-specific scoring
                if score > 0.95:
                    print(
                        f"Achieved excellent results (score: {score:.3f}), stopping early."
                    )
                    break

            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                error_result = {
                    "iteration": iteration,
                    "command": command,
                    "error": str(e),
                    "constraint_code": constraint_code
                    if "constraint_code" in locals()
                    else None,
                    "function_name": function_name
                    if "function_name" in locals()
                    else None,
                    "attempt_log": attempt_log if "attempt_log" in locals() else [],
                    "timestamp": time.time(),
                }
                self.iteration_results.append(error_result)
                continue

        # Compile final results
        final_results = {
            "command": command,
            "total_iterations": len(self.iteration_results),
            "best_iteration": best_result,
            "best_score": best_score,
            "all_iterations": self.iteration_results,
            "results_directory": str(run_dir),
            "pipeline_success": best_score > 0.5,
        }

        # Save final summary
        with open(run_dir / "pipeline_summary.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            json_safe_results = self._make_json_safe(final_results)
            json.dump(json_safe_results, f, indent=2)

        print("\\n=== PIPELINE COMPLETE ===")
        print(f"Best score achieved: {best_score:.3f}")
        print(f"Results saved to: {run_dir}")

        return final_results

    def _generate_constraints_with_retry(
        self,
        system_prompt: str,
        user_message: str,
        context: str | None = None,
        command: str = "",
        max_attempts: int = 10,
    ) -> tuple[str, str, list[dict[str, Any]]]:
        """
        Generate constraints using the LLM with comprehensive auto-retry on failures.

        This implements the full repair loop with detailed error feedback to the LLM.

        Returns:
            Tuple of (final_code, function_name, attempt_log)
        """
        attempts: list[dict[str, Any]] = []

        for attempt in range(1, max_attempts + 1):
            print(f"🔄 Constraint generation attempt {attempt}/{max_attempts}")

            try:
                # Generate code from LLM
                if attempt == 1:
                    # First attempt uses original prompt
                    prompt = user_message
                    if context:
                        prompt = f"{context}\n\n{user_message}"
                else:
                    # Subsequent attempts use repair prompts with detailed error feedback
                    failed_code = attempts[-1]["code"]
                    error_msg = attempts[-1]["error"]
                    prompt = self.constraint_generator.create_repair_prompt(
                        command, failed_code, error_msg, attempt
                    )

                # Call LLM
                response = self.llm_client.generate_constraints(
                    system_prompt, prompt, None
                )

                # Extract code from response with improved extraction
                mpc_config_code = self.llm_client.extract_raw_code(response)

                if not mpc_config_code.strip():
                    attempts.append(
                        {
                            "attempt": attempt,
                            "code": mpc_config_code,
                            "error": "No code extracted from LLM response - check response format",
                            "success": False,
                        }
                    )
                    continue

                # Create fresh LLM MPC instance for this attempt
                task_mpc = LLMTaskMPC(self.kindyn_model, self.config)

                # Test the MPC configuration code with SafeExecutor
                success, error_msg = self.safe_executor.execute_mpc_configuration_code(
                    mpc_config_code, task_mpc
                )

                if not success:
                    # Log detailed failure reason
                    attempts.append(
                        {
                            "attempt": attempt,
                            "code": mpc_config_code,
                            "error": error_msg,
                            "success": False,
                            "failure_stage": "mpc_configuration",
                        }
                    )
                    continue

                # Store the configured MPC for later use
                self.current_task_mpc = task_mpc
                self.llm_mpc_code = mpc_config_code

                # Get configuration summary for logging
                config_summary = task_mpc.get_configuration_summary()
                task_name = config_summary.get("task_name", "unknown")

                # Success!
                attempts.append(
                    {
                        "attempt": attempt,
                        "code": mpc_config_code,
                        "error": "",
                        "success": True,
                        "task_name": task_name,
                        "failure_stage": "none",
                        "config_summary": config_summary,
                    }
                )

                print(f"✅ MPC configuration successful on attempt {attempt}")
                print(f"   Task: {task_name}")
                print(f"   Duration: {config_summary.get('duration', 0):.2f}s")
                print(f"   Constraints: {config_summary.get('num_constraints', 0)}")
                print(
                    f"   Contact phases: {len(config_summary.get('contact_phases', []))}"
                )
                return mpc_config_code, task_name, attempts

            except Exception as e:
                # Catch any unexpected errors and provide details
                import traceback

                error_details = (
                    f"Unexpected error: {str(e)}\nTraceback: {traceback.format_exc()}"
                )

                attempts.append(
                    {
                        "attempt": attempt,
                        "code": mpc_config_code
                        if "mpc_config_code" in locals()
                        else "",
                        "error": error_details,
                        "success": False,
                        "failure_stage": "unexpected_exception",
                    }
                )
                print(f"❌ Attempt {attempt} failed with unexpected error: {str(e)}")
                continue

        # All attempts failed - provide comprehensive failure summary
        print(f"❌ All {max_attempts} constraint generation attempts failed")

        # Analyze failure patterns
        failure_stages = [
            attempt.get("failure_stage", "unknown") for attempt in attempts
        ]
        common_errors = {}
        for attempt in attempts:
            error = attempt.get("error", "")[:100]  # First 100 chars
            common_errors[error] = common_errors.get(error, 0) + 1

        print("Failure analysis:")
        print(f"  Failure stages: {set(failure_stages)}")
        print(f"  Most common errors: {list(common_errors.keys())[:3]}")

        last_error = attempts[-1]["error"] if attempts else "No attempts recorded"
        raise ValueError(
            f"Failed to generate valid constraints after {max_attempts} attempts.\n"
            f"Last error: {last_error}\n"
            f"Common failure stages: {set(failure_stages)}"
        )

    def _solve_trajectory_optimization(
        self, mpc_config_code: str, task_name: str, iteration: int, run_dir: Path
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

    def _execute_simulation(
        self, optimization_result: dict[str, Any], iteration: int, run_dir: Path
    ) -> dict[str, Any]:
        """Execute the optimized trajectory in simulation."""

        if not optimization_result["success"]:
            return {
                "success": False,
                "error": "Cannot simulate - optimization failed",
                "tracking_error": float("inf"),
            }

        if self.env is None:
            return {
                "success": False,
                "error": "Cannot simulate - environment not available",
                "tracking_error": float("inf"),
            }

        try:
            state_traj = optimization_result["state_trajectory"]
            grf_traj = optimization_result["grf_trajectory"]
            joint_vel_traj = optimization_result["joint_vel_trajectory"]

            # Create input trajectory for rendering
            input_traj = np.concatenate([joint_vel_traj, grf_traj], axis=1)

            print("🎬 Creating planned trajectory visualization...")
            # Render planned trajectory
            planned_traj_images = render_and_save_planned_trajectory(
                state_traj, input_traj, self.env, f"_iter_{iteration}"
            )

            print("🔧 Computing joint torques via inverse dynamics...")
            # Compute joint torques using inverse dynamics
            joint_torques_traj = compute_joint_torques(
                self.kindyn_model,
                state_traj,
                grf_traj,
                self.config.mpc_config.contact_sequence,
                self.config.mpc_config.mpc_dt,
                joint_vel_traj,
            )

            print("🏃 Executing trajectory in physics simulation...")
            # Execute in simulation
            qpos_traj, qvel_traj, sim_grf_traj, sim_images = simulate_trajectory(
                self.env, joint_torques_traj, planned_traj_images
            )

            # Calculate tracking error
            tracking_error = self._calculate_tracking_error(
                state_traj, qpos_traj, qvel_traj
            )

            # Analyze simulation results
            simulation_analysis = self._analyze_simulation_quality(
                qpos_traj, qvel_traj, sim_grf_traj, tracking_error
            )

            # Save simulation video
            print("💾 Saving simulation video...")
            if sim_images:
                import imageio

                fps = 1 / self.config.experiment.sim_dt
                video_path = run_dir / f"simulation_iter_{iteration}.mp4"
                imageio.mimsave(str(video_path), sim_images, fps=fps)
                print(f"  Saved: {video_path}")

            # Also save planned trajectory video if available
            if planned_traj_images:
                import imageio

                fps = 1 / self.config.experiment.sim_dt
                planned_video_path = run_dir / f"planned_traj_iter_{iteration}.mp4"
                imageio.mimsave(str(planned_video_path), planned_traj_images, fps=fps)
                print(f"  Saved: {planned_video_path}")

            result = {
                "success": True,
                "tracking_error": tracking_error,
                "simulation_analysis": simulation_analysis,
                "simulated_qpos": qpos_traj,
                "simulated_qvel": qvel_traj,
                "simulated_grf": sim_grf_traj,
                "realistic": simulation_analysis.get("realistic", False),
            }

            return result

        except Exception as e:
            return {
                "success": False,
                "error": f"Simulation failed: {str(e)}",
                "tracking_error": float("inf"),
            }

    def _calculate_tracking_error(
        self, planned_state: np.ndarray, sim_qpos: np.ndarray, sim_qvel: np.ndarray
    ) -> float:
        """Calculate tracking error between planned and simulated trajectories."""
        try:
            # Convert simulation results to MPC state format
            sim_states = []
            for i in range(min(len(sim_qpos), planned_state.shape[0])):
                sim_state, _ = conversion.sim_to_mpc(sim_qpos[i], sim_qvel[i])
                sim_states.append(sim_state)

            if not sim_states:
                return float("inf")

            sim_state_traj = np.array(sim_states)

            # Calculate RMS error for comparable lengths
            min_length = min(planned_state.shape[0], sim_state_traj.shape[0])
            planned_truncated = planned_state[:min_length]
            sim_truncated = sim_state_traj[:min_length]

            # Focus on key state components (position, orientation)
            key_states = np.concatenate(
                [
                    planned_truncated[:, 0:3],  # COM position
                    planned_truncated[:, 6:9],  # Euler angles
                ],
                axis=1,
            )

            sim_key_states = np.concatenate(
                [
                    sim_truncated[:, 0:3],
                    sim_truncated[:, 6:9],
                ],
                axis=1,
            )

            error = np.sqrt(np.mean((key_states - sim_key_states) ** 2))
            return float(error)

        except Exception as e:
            print(f"Error calculating tracking error: {e}")
            return float("inf")

    def _analyze_simulation_quality(
        self,
        qpos_traj: np.ndarray,
        qvel_traj: np.ndarray,
        grf_traj: np.ndarray,
        tracking_error: float,
    ) -> dict[str, Any]:
        """Analyze the quality and realism of the simulation."""

        analysis: dict[str, Any] = {
            "realistic": True,
            "max_joint_velocity": 0.0,
            "max_grf": 0.0,
            "trajectory_length": len(qpos_traj),
            "issues": [],
        }

        try:
            # Check for realistic joint velocities
            if len(qvel_traj) > 0:
                joint_velocities = qvel_traj[:, 6:]  # Skip base velocities
                max_joint_vel = np.max(np.abs(joint_velocities))
                analysis["max_joint_velocity"] = float(max_joint_vel)

                if max_joint_vel > 20.0:  # Unrealistic joint velocity
                    analysis["realistic"] = False
                    analysis["issues"].append("Unrealistic joint velocities")

            # Check ground reaction forces
            if len(grf_traj) > 0:
                max_grf = np.max(np.abs(grf_traj))
                analysis["max_grf"] = float(max_grf)

                if max_grf > 1000.0:  # Unrealistic forces
                    analysis["realistic"] = False
                    analysis["issues"].append("Unrealistic ground reaction forces")

            # Check tracking error
            if tracking_error > 0.5:
                analysis["realistic"] = False
                analysis["issues"].append("Poor trajectory tracking")

            # Check for simulation failure
            if len(qpos_traj) < 10:  # Very short trajectory suggests failure
                analysis["realistic"] = False
                analysis["issues"].append("Simulation terminated early")

        except Exception as e:
            analysis["realistic"] = False
            analysis["issues"].append(f"Analysis failed: {str(e)}")

        return analysis

    def _create_feedback_context(
        self,
        iteration: int,
        optimization_result: dict[str, Any],
        simulation_result: dict[str, Any],
        constraint_code: str,
    ) -> str:
        """Create feedback context for the next LLM iteration."""

        # Extract trajectory data
        trajectory_data = optimization_result.get("trajectory_analysis", {})
        optimization_status = optimization_result.get("optimization_metrics", {})

        return self.constraint_generator.create_feedback_context(
            iteration,
            trajectory_data,
            optimization_status,
            simulation_result,
            constraint_code,
        )

    def _score_iteration(self, iteration_result: dict[str, Any]) -> float:
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
        task_score = self._score_task_specific_behavior(iteration_result)
        score += 0.20 * task_score

        return score

    def _score_task_specific_behavior(self, iteration_result: dict[str, Any]) -> float:
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
            total_pitch_rotation = abs(
                trajectory_analysis.get("total_pitch_rotation", 0)
            )
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
                    return (
                        min(1.0, height_gain / target_height)
                        if height_gain > 0.1
                        else 0
                    )

                elif any(
                    word in command
                    for word in ["left", "right", "forward", "backward", "side"]
                ):
                    # Directional jump: needs both height and displacement
                    height_score = (
                        min(1.0, height_gain / 0.2) if height_gain > 0.05 else 0
                    )
                    displacement = max(com_displacement_x, com_displacement_y)
                    displacement_score = (
                        min(1.0, displacement / 0.3) if displacement > 0.1 else 0
                    )
                    return 0.5 * height_score + 0.5 * displacement_score

                else:
                    # Generic jump: just needs height
                    return min(1.0, height_gain / 0.2) if height_gain > 0.05 else 0

            elif any(word in command for word in ["squat", "crouch", "lower"]):
                # Lowering motion: needs negative height change
                return min(1.0, abs(height_gain) / 0.1) if height_gain < -0.05 else 0

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

    def _save_iteration_results(
        self, iteration_result: dict[str, Any], run_dir: Path
    ) -> None:
        """Save detailed results for a single iteration."""
        iteration = iteration_result["iteration"]

        # Save iteration summary (JSON-safe version)
        iteration_file = run_dir / f"iteration_{iteration}.json"
        json_safe_result = self._make_json_safe(iteration_result)

        with open(iteration_file, "w") as f:
            json.dump(json_safe_result, f, indent=2)

        # Save constraint code separately
        if "constraint_code" in iteration_result:
            func_name = iteration_result.get("function_name", "unknown")
            code_file = run_dir / f"constraints_iter_{iteration}_{func_name}.py"
            with open(code_file, "w") as f:
                f.write(iteration_result["constraint_code"])

        # Save attempt log separately for detailed debugging
        if "attempt_log" in iteration_result and iteration_result["attempt_log"]:
            attempt_file = run_dir / f"attempts_iter_{iteration}.json"
            with open(attempt_file, "w") as f:
                json.dump(iteration_result["attempt_log"], f, indent=2)

    def _inject_llm_constraints_to_mpc(self) -> None:
        """
        Inject LLM constraints into the MPC problem using the proper MPC constraint interface.

        This adds the LLM-generated constraints as additional path constraints to the
        existing MPC constraint system, rather than directly to the Opti problem.
        """
        if not self.llm_constraints:
            print("Warning: No LLM constraints to inject")
            return

        constraint_func = self.llm_constraints[0]  # Use the current constraint

        try:
            print("🔧 Adding LLM constraint to MPC path constraints...")

            # Add the LLM constraint to the MPC's path constraint list
            # This ensures it's applied consistently with the existing constraint system
            if hasattr(self.config, "mpc_config") and hasattr(
                self.config.mpc_config, "path_constraints"
            ):
                # Add the LLM constraint to the path constraints list
                self.config.mpc_config.path_constraints.append(constraint_func)
                print("✅ Added LLM constraint to path constraints list")
                print(
                    f"   Total path constraints: {len(self.config.mpc_config.path_constraints)}"
                )
            else:
                # Fallback: inject directly into the Opti problem (less preferred)
                print("Warning: Using direct Opti injection as fallback")
                self._inject_llm_constraints_direct()

        except Exception as e:
            print(f"Error: Failed to inject LLM constraints: {e}")
            import traceback

            traceback.print_exc()

    def _inject_llm_constraints_direct(self) -> None:
        """
        Fallback method: Inject LLM constraints directly into the Opti problem.
        This is used when the preferred path constraint method fails.
        """
        if not self.llm_constraints:
            return

        constraint_func = self.llm_constraints[0]

        try:
            print(f"🔧 Direct injection for {self.mpc.horizon} time steps...")

            constraints_added = 0
            for k in range(self.mpc.horizon):
                try:
                    x_k = self.mpc.X[:, k]
                    u_k = self.mpc.U[:, k]
                    contact_k = self.mpc.P_contact[:, k]

                    # Apply the constraint function
                    constraint_result = constraint_func(
                        x_k, u_k, self.kindyn_model, self.config, contact_k
                    )

                    if (
                        not isinstance(constraint_result, tuple)
                        or len(constraint_result) != 3
                    ):
                        print(f"Warning: Invalid constraint result at time step {k}")
                        continue

                    constraint_expr, constraint_l, constraint_u = constraint_result

                    if constraint_expr is not None:
                        # Add lower and upper bound constraints
                        self.mpc.opti.subject_to(constraint_expr >= constraint_l)
                        self.mpc.opti.subject_to(constraint_expr <= constraint_u)
                        constraints_added += 1

                except Exception as step_error:
                    print(f"Warning: Failed to process time step {k}: {step_error}")
                    continue

            print(
                f"✅ Direct injection completed: {constraints_added} constraint sets added"
            )

        except Exception as e:
            print(f"Error: Direct constraint injection failed: {e}")
            import traceback

            traceback.print_exc()

    def _make_json_safe(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-JSON-serializable objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
