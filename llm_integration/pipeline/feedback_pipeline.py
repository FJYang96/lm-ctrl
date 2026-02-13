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
from ..feedback import (
    create_visual_feedback,
    evaluate_failed_iteration,
    evaluate_iteration,
    summarize_iteration,
)
from ..logging_config import logger
from ..mpc import LLMTaskMPC
from .constraint_generation import generate_constraints_with_retry
from .feedback_context import create_feedback_context
from .optimization import solve_trajectory_optimization
from .simulation import (
    execute_simulation,
)
from .utils import (
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
    _create_feedback_context = create_feedback_context
    _save_iteration_results = save_iteration_results
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

        # Legacy MPC for fallback (in case LLM MPC fails)
        self.fallback_mpc = QuadrupedMPCOpti(
            model=self.kindyn_model, config=self.config, build=True
        )

        # Task-specific MPC tracking
        self.current_task_mpc: LLMTaskMPC | None = None

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

        # Visual summary of current iteration's trajectory frames
        self.current_visual_summary: str = ""

        # Slack weights tracking (for feedback display)
        self.current_slack_weights: dict[str, float] = {}

        # Warm-start state for adaptive policy
        self.previous_warmstart: dict[str, Any] | None = None
        self.previous_objective: float = float("inf")
        self.previous_score: float = -float("inf")
        self.use_warmstart_next: bool = False  # False for iteration 1 (cold start)

        # Pivot/tweak tracking
        self.consecutive_no_improvement: int = 0
        self.recent_scores: list[float] = []  # rolling window for stagnation detection

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

        # Reset warm-start state
        self.previous_warmstart = None
        self.previous_objective = float("inf")
        self.previous_score = -float("inf")
        self.use_warmstart_next = False  # First iteration always cold-starts
        self.consecutive_no_improvement = 0
        self.recent_scores = []

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
                warmstart: dict[str, Any] | None = (
                    self.previous_warmstart if self.use_warmstart_next else None
                )
                optimization_result = self._solve_trajectory_optimization(
                    constraint_code,
                    function_name,
                    iteration,
                    run_dir,
                    warmstart=warmstart,
                )

                # Step 5: Execute trajectory in simulation
                simulation_result = self._execute_simulation(
                    optimization_result, iteration, run_dir
                )

                # Extract visual feedback for next iteration
                self.current_images = create_visual_feedback(run_dir, iteration)

                # Summarize this iteration's frames so we can attach to feedback
                self.current_visual_summary = self.llm_client.summarize_frames(
                    self.current_images, command
                )

                # Track previous iteration analysis for comparison
                self.previous_iteration_analysis = optimization_result.get(
                    "trajectory_analysis", {}
                )

                # === LLM-based evaluation for scoring ===
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
                    # LLM-based scoring for failed optimization
                    error_info = optimization_result.get("optimization_metrics", {})
                    trajectory_analysis = optimization_result.get(
                        "trajectory_analysis", {}
                    )

                    # Use LLM to score the failed iteration based on partial progress
                    llm_eval = evaluate_failed_iteration(
                        command=command,
                        trajectory_analysis=trajectory_analysis,
                        constraint_code=constraint_code,
                        error_info=error_info,
                        images=self.current_images,
                    )
                    score = llm_eval.get("score", 0.0)

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

                    summary = llm_eval.get("summary", "")
                    if not summary:
                        summary = "Failed to generate summary for feedback"
                        logger.warning("LLM evaluation did not return a summary")
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
                    "criteria": llm_eval.get("criteria", []),
                    "warnings": llm_eval.get("warnings", []),
                    "error": error_info.get("error_message", "")
                    if not opt_success
                    else "",
                }
                self.iteration_summaries.append(iteration_history_entry)

                # === Consecutive no-improvement counter + pivot signal ===
                is_new_best = score > best_score
                if is_new_best:
                    self.consecutive_no_improvement = 0
                    best_score = score
                else:
                    self.consecutive_no_improvement += 1

                # Track recent scores for stagnation detection
                self.recent_scores.append(score)

                # Stagnation: last 4 scores all within 0.1 of each other
                stagnated = False
                if len(self.recent_scores) >= 4:
                    window = self.recent_scores[-4:]
                    score_range = max(window) - min(window)
                    if score_range <= 0.1:
                        stagnated = True

                # Determine pivot signal
                if stagnated:
                    pivot_signal: str | None = "pivot"
                elif self.consecutive_no_improvement >= 3:
                    pivot_signal = "pivot"
                elif opt_success and score < 0.2:
                    pivot_signal = "pivot"  # converged but drastically wrong
                elif self.consecutive_no_improvement >= 1:
                    pivot_signal = "tweak"
                else:
                    pivot_signal = None

                logger.info(
                    f"Pivot logic: consecutive_no_improvement={self.consecutive_no_improvement}, "
                    f"stagnated={stagnated}, pivot_signal={pivot_signal}"
                )

                # === Warmstart policy based on pivot signal ===
                current_objective = optimization_result.get(
                    "optimization_metrics", {}
                ).get("objective_value", float("inf"))

                if pivot_signal == "pivot":
                    self.use_warmstart_next = False  # cold-start with new reference
                    self.recent_scores.clear()  # reset stagnation window after pivot
                else:
                    self.use_warmstart_next = True  # warmstart from previous

                # Log the decision
                if iteration == 1:
                    logger.info(
                        f"Warmstart policy: Score {score:.2f}, "
                        f"Obj {current_objective:.4f} → warm-start next"
                    )
                else:
                    decision = (
                        "warm-start next"
                        if self.use_warmstart_next
                        else "pivot → cold-start next"
                    )
                    logger.info(
                        f"Warmstart policy: Score {self.previous_score:.2f}→{score:.2f}, "
                        f"Obj {self.previous_objective:.4f}→{current_objective:.4f} → {decision}"
                    )

                # Always store warmstart data (even if we plan to cold-start next)
                ws_X = optimization_result.get("warmstart_X")
                ws_U = optimization_result.get("warmstart_U")
                if ws_X is not None and ws_U is not None:
                    self.previous_warmstart = {"X": ws_X, "U": ws_U}

                self.previous_objective = current_objective
                self.previous_score = score

                # Step 6: Create feedback context (AFTER scoring + pivot signal)
                feedback_context = self._create_feedback_context(
                    iteration,
                    command,
                    optimization_result,
                    simulation_result,
                    constraint_code,
                    run_dir,
                    pivot_signal=pivot_signal,
                )

                # Append visual summary of THIS iteration's trajectory frames
                if self.current_visual_summary:
                    feedback_context += (
                        f"\n\n--- VISUAL SUMMARY OF ITERATION {iteration} TRAJECTORY ---\n"
                        f"{self.current_visual_summary}"
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
                    "feedback_context": feedback_context,
                    "llm_evaluation": llm_eval,
                    "summary": summary,
                    "pivot_signal": pivot_signal,
                    "timestamp": time.time(),
                }

                self.iteration_results.append(iteration_result)

                # Update best result (deferred from pivot logic to after iteration_result exists)
                if is_new_best:
                    best_result = iteration_result

                # Save iteration results
                self._save_iteration_results(iteration_result, run_dir)

                # Use feedback for next iteration
                if iteration < self.max_iterations:
                    context = feedback_context

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
