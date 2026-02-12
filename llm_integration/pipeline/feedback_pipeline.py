"""Main feedback pipeline implementing Algorithm 1 from the PDF."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from mpc.dynamics.model import KinoDynamic_Model
from mpc.mpc_opti import QuadrupedMPCOpti

from ..client import LLMClient
from ..constraint import ConstraintGenerator
from ..executor import SafeConstraintExecutor
from ..feedback import create_visual_feedback, evaluate_iteration, summarize_iteration
from ..logging_config import logger
from ..mpc import LLMTaskMPC
from .constraint_generation import generate_constraints_with_retry
from .feedback_context import create_feedback_context
from .optimization import solve_trajectory_optimization
from .scoring import score_iteration, score_task_specific_behavior
from .simulation import (
    analyze_simulation_quality,
    calculate_tracking_error,
    execute_simulation,
)
from .utils import (
    inject_llm_constraints_direct,
    inject_llm_constraints_to_mpc,
    make_json_safe,
    save_iteration_results,
)

try:
    from gym_quadruped.quadruped_env import QuadrupedEnv
except ImportError:
    logger.warning("gym_quadruped not available. Simulation features may be limited.")
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

    # Assign imported functions as methods
    _solve_trajectory_optimization = solve_trajectory_optimization
    _execute_simulation = execute_simulation
    _calculate_tracking_error = calculate_tracking_error
    _analyze_simulation_quality = analyze_simulation_quality
    _create_feedback_context = create_feedback_context
    _score_iteration = score_iteration
    _score_task_specific_behavior = score_task_specific_behavior
    _save_iteration_results = save_iteration_results
    _inject_llm_constraints_to_mpc = inject_llm_constraints_to_mpc
    _inject_llm_constraints_direct = inject_llm_constraints_direct
    _make_json_safe = make_json_safe
    _generate_constraints_with_retry = generate_constraints_with_retry

    def __init__(self, config_obj: Any = None, use_slack: bool = True):
        """
        Initialize the feedback pipeline.

        Args:
            config_obj: Configuration object (uses default config if None)
            use_slack: Whether to use slack formulation for robust optimization
        """
        self.config = config_obj if config_obj is not None else config
        self.use_slack = use_slack

        # Initialize components
        self.llm_client = LLMClient()
        self.constraint_generator = ConstraintGenerator(config=self.config)
        self.safe_executor = SafeConstraintExecutor()

        # Initialize kinodynamic model
        self.kindyn_model = KinoDynamic_Model(self.config)

        # Initialize LLM-specific MPC (replaces the fixed MPC)
        self.llm_mpc = LLMTaskMPC(
            self.kindyn_model, self.config, use_slack=self.use_slack
        )

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
            logger.warning("Simulation environment not available")
            self.env = None

        # Pipeline state
        self.iteration_results: list[dict[str, Any]] = []
        self.max_iterations = int(os.getenv("MAX_LLM_ITERATIONS", "5"))
        self.results_dir = Path(os.getenv("RESULTS_DIR", "results/llm_iterations"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Enhanced feedback tracking
        self.previous_iteration_analysis: dict[str, Any] | None = None
        self.current_joint_torques: np.ndarray | None = None
        self.current_images: list[str] = []

        # LLM-based iteration history
        self.iteration_summaries: list[dict[str, Any]] = []

        # Slack weights tracking (for feedback display)
        self.current_slack_weights: dict[str, float] = {}

    def run_pipeline(self, command: str) -> dict[str, Any]:
        """
        Run the complete LLM feedback pipeline for a given command.

        Args:
            command: Natural language command (e.g., "do a backflip")

        Returns:
            Dictionary containing complete pipeline results
        """
        logger.info(f"Pipeline started: '{command}'")

        # Create results directory for this run
        timestamp = int(time.time())
        run_dir = self.results_dir / f"{command.replace(' ', '_')}_{timestamp}"
        run_dir.mkdir(exist_ok=True)

        # Initialize pipeline state
        self.iteration_results = []
        self.iteration_summaries = []
        self.previous_iteration_analysis = None
        self.current_images = []
        context = None
        best_result = None
        best_score = -float("inf")

        # Algorithm 1: Iterative Refinement Pipeline
        system_prompt = self.constraint_generator.get_system_prompt()
        initial_user_message = self.constraint_generator.get_user_prompt(command)

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"--- Iteration {iteration} ---")

            try:
                # Step 3: Generate constraints using LLM with auto-retry
                constraint_code, function_name, attempt_log = (
                    self._generate_constraints_with_retry(
                        system_prompt,
                        initial_user_message,
                        context,
                        command,
                        images=self.current_images,
                    )
                )

                # Step 4: Solve optimization problem with new constraints
                optimization_result = self._solve_trajectory_optimization(
                    constraint_code, function_name, iteration, run_dir
                )

                # Step 5: Execute trajectory in simulation
                simulation_result = self._execute_simulation(
                    optimization_result, iteration, run_dir
                )

                # Step 6: Create enhanced feedback context with visual frames
                feedback_context = self._create_feedback_context(
                    iteration,
                    command,
                    optimization_result,
                    simulation_result,
                    constraint_code,
                    run_dir,
                )

                # Extract visual feedback for next iteration
                self.current_images = create_visual_feedback(run_dir, iteration)

                # Track previous iteration analysis for comparison
                self.previous_iteration_analysis = optimization_result.get(
                    "trajectory_analysis", {}
                )

                # Collect iteration results (including feedback for debugging)
                iteration_result = {
                    "iteration": iteration,
                    "command": command,
                    "constraint_code": constraint_code,
                    "function_name": function_name,
                    "attempt_log": attempt_log,
                    "optimization": optimization_result,
                    "simulation": simulation_result,
                    "feedback_context": feedback_context,  # Added for debugging
                    "timestamp": time.time(),
                }

                self.iteration_results.append(iteration_result)

                # Save iteration results
                self._save_iteration_results(iteration_result, run_dir)

                # Use feedback for next iteration
                if iteration < self.max_iterations:
                    context = feedback_context

                # Use LLM-based evaluation for scoring
                trajectory_analysis = optimization_result.get("trajectory_analysis", {})
                opt_success = optimization_result.get("success", False)

                if opt_success and trajectory_analysis:
                    # LLM evaluates the successful iteration
                    llm_eval = evaluate_iteration(
                        command=command,
                        trajectory_analysis=trajectory_analysis,
                        constraint_code=constraint_code,
                        images=self.current_images,
                    )
                    score = llm_eval.get("score", 0.5)
                    iteration_result["llm_evaluation"] = llm_eval

                    # Log detailed success evaluation
                    logger.info("=== LLM Evaluation (SUCCESS) ===")
                    logger.info(f"  Score: {score:.2f}")
                    logger.info("  Trajectory Metrics:")
                    logger.info(
                        f"    Pitch: {trajectory_analysis.get('total_pitch_rotation', 0):.2f} rad ({trajectory_analysis.get('total_pitch_rotation', 0) * 57.3:.0f}°)"
                    )
                    logger.info(
                        f"    Height gain: {trajectory_analysis.get('height_gain', 0):.3f}m"
                    )
                    logger.info(
                        f"    Yaw: {trajectory_analysis.get('max_yaw', 0):.2f} rad"
                    )
                    logger.info(
                        f"    Flight duration: {trajectory_analysis.get('flight_duration', 0):.2f}s"
                    )
                    logger.info("  Criteria:")
                    for criterion in llm_eval.get("criteria", []):
                        progress = criterion.get("progress", 0)
                        status = "✓" if progress >= 0.8 else "✗"
                        logger.info(
                            f"    {status} {criterion.get('name')}: {criterion.get('achieved')} (target: {criterion.get('target')}, {progress:.0%})"
                        )
                    if llm_eval.get("warnings"):
                        logger.info("  Warnings:")
                        for warning in llm_eval.get("warnings", []):
                            logger.info(f"    ⚠ {warning}")
                    summary = llm_eval.get(
                        "summary", f"Iteration {iteration} completed"
                    )
                    logger.info(f"  Summary: {summary}")
                else:
                    # Fallback scoring for failed optimization
                    score = self._score_iteration(iteration_result)
                    error_info = optimization_result.get("optimization_metrics", {})
                    trajectory_analysis = optimization_result.get(
                        "trajectory_analysis", {}
                    )

                    # Log detailed failure evaluation
                    logger.info("=== LLM Evaluation (FAILED) ===")
                    logger.info(f"  Score: {score:.2f}")
                    if error_info.get("error_message"):
                        logger.info(f"  Error: {error_info.get('error_message')}")
                    if error_info.get("solver_iterations"):
                        logger.info(
                            f"  Solver iterations: {error_info.get('solver_iterations')}"
                        )
                    if trajectory_analysis:
                        logger.info("  Last attempt metrics:")
                        logger.info(
                            f"    Pitch: {trajectory_analysis.get('total_pitch_rotation', 0):.2f} rad ({trajectory_analysis.get('total_pitch_rotation', 0) * 57.3:.0f}°)"
                        )
                        logger.info(
                            f"    Height gain: {trajectory_analysis.get('height_gain', 0):.3f}m"
                        )

                    # Get LLM summary for failed optimization (with video frames)
                    summary = summarize_iteration(
                        command=command,
                        constraint_code=constraint_code,
                        success=False,
                        error_info=error_info,
                        trajectory_analysis=trajectory_analysis,
                        images=self.current_images,
                    )
                    logger.info(f"  Summary: {summary}")

                # Add detailed iteration info to history
                iteration_history_entry = {
                    "iteration": iteration,
                    "success": opt_success,
                    "score": score,
                    "summary": summary,
                    "metrics": {
                        "pitch": trajectory_analysis.get("total_pitch_rotation", 0),
                        "height_gain": trajectory_analysis.get("height_gain", 0),
                        "yaw": trajectory_analysis.get("max_yaw", 0),
                        "flight_duration": trajectory_analysis.get(
                            "flight_duration", 0
                        ),
                    },
                    "criteria": llm_eval.get("criteria", [])
                    if opt_success and "llm_eval" in locals()
                    else [],
                    "warnings": llm_eval.get("warnings", [])
                    if opt_success and "llm_eval" in locals()
                    else [],
                    "error": error_info.get("error_message", "")
                    if not opt_success
                    else "",
                }
                self.iteration_summaries.append(iteration_history_entry)
                iteration_result["summary"] = summary

                if score > best_score:
                    best_score = score
                    best_result = iteration_result

                # Early stopping if we achieve excellent results
                if score > 0.95:
                    logger.info(f"Early stop: score {score:.2f}")
                    break

            except Exception as e:
                logger.error(f"Iteration {iteration} error: {e}")

                # Generate summary for failed iteration
                error_info = {"error_message": str(e)}
                code = constraint_code if "constraint_code" in locals() else ""
                summary = summarize_iteration(
                    command=command,
                    constraint_code=code,
                    success=False,
                    error_info=error_info,
                    images=self.current_images if self.current_images else None,
                )
                # Add detailed error info to history
                iteration_history_entry = {
                    "iteration": iteration,
                    "success": False,
                    "score": 0.0,
                    "summary": summary,
                    "metrics": {},
                    "criteria": [],
                    "warnings": [],
                    "error": str(e)[:300],
                }
                self.iteration_summaries.append(iteration_history_entry)

                error_result = {
                    "iteration": iteration,
                    "command": command,
                    "error": str(e),
                    "summary": summary,
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

        logger.info(f"Pipeline complete: best_score={best_score:.2f}")

        return final_results
