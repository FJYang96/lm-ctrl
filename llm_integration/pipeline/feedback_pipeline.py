"""Main feedback pipeline implementing Algorithm 1 from the PDF."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from mpc.dynamics.model import KinoDynamic_Model
from mpc.mpc_opti import QuadrupedMPCOpti

from ..client import LLMClient
from ..constraint import ConstraintGenerator
from ..executor import SafeConstraintExecutor
from ..feedback import (
    create_visual_feedback,
    format_hardness_report,
)
from ..feedback.constraint_feedback import generate_constraint_feedback
from ..feedback.llm_evaluation import (
    evaluate_iteration_unified,
    generate_iteration_summary,
)
from ..feedback.reference_feedback import generate_reference_feedback
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
    4. Dual feedback (constraint + reference) and iteration
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
        self.current_images: list[str] = []

        # LLM-based iteration history (structured summaries)
        self.iteration_summaries: list[dict[str, Any]] = []

        # Visual summary of current iteration's trajectory frames
        self.current_visual_summary: str = ""

        # Slack weights tracking (for feedback display)
        self.current_slack_weights: dict[str, float] = {}

        # Score tracking
        self.previous_score: float = -float("inf")
        self.recent_scores: list[float] = []  # rolling window for pivot detection

    def _compute_pivot_signal(self, iteration: int) -> str | None:
        """Compute the pivot/tweak signal based on score history.

        Returns:
            "pivot" — fundamentally change approach
            "tweak" — make incremental improvements
            None — first iteration, no signal
        """
        scores = self.recent_scores

        if len(scores) < 2:
            # First iteration or not enough history
            return None

        # Monotonic decline for 2 consecutive iterations (3+ scores needed)
        if len(scores) >= 3:
            if scores[-1] < scores[-2] < scores[-3]:
                logger.info(
                    "Pivot trigger: monotonic decline for 2 consecutive iterations"
                )
                return "pivot"

        # Stagnation: last 5 scores within a tight 0.05 band
        if len(scores) >= 5:
            window = scores[-5:]
            score_range = max(window) - min(window)
            if score_range < 0.05:
                logger.info(
                    f"Pivot trigger: stagnation (last 5 scores within {score_range:.3f} band)"
                )
                return "pivot"

        # Default: tweak
        return "tweak"

    def _extract_constraint_violations(
        self, optimization_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract constraint violations from the MPC solver.

        Works for both successful and failed solves.
        """
        constraint_violations: dict[str, Any] = {}
        opt_success = optimization_result.get("success", False)

        if not self.current_task_mpc or not self.current_task_mpc.mpc:
            return constraint_violations

        try:
            # Get system constraint violations
            constraint_violations = (
                self.current_task_mpc.mpc.get_constraint_violations()
            )

            # Get LLM constraint violations
            try:
                if opt_success:
                    X_val = self.current_task_mpc.mpc.opti.value(
                        self.current_task_mpc.mpc.X
                    )
                    U_val = self.current_task_mpc.mpc.opti.value(
                        self.current_task_mpc.mpc.U
                    )
                else:
                    X_val = self.current_task_mpc.mpc.opti.debug.value(
                        self.current_task_mpc.mpc.X
                    )
                    U_val = self.current_task_mpc.mpc.opti.debug.value(
                        self.current_task_mpc.mpc.U
                    )
                llm_violations = self.current_task_mpc.evaluate_constraint_violations(
                    X_val, U_val
                )
                constraint_violations["llm_constraints"] = llm_violations.get(
                    "llm_constraints", []
                )
                constraint_violations["llm_summary"] = llm_violations.get("summary", [])
            except Exception as llm_e:
                constraint_violations["llm_constraints"] = [
                    f"Could not evaluate LLM constraints: {llm_e}"
                ]

        except Exception as e:
            constraint_violations = {"summary": [f"Could not analyze violations: {e}"]}

        return constraint_violations

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
        self.current_images = []
        context = None
        best_result = None
        best_score = -float("inf")

        # Reset score tracking
        self.previous_score = -float("inf")
        self.recent_scores = []

        # Algorithm 1: Iterative Refinement Pipeline
        system_prompt = self.constraint_generator.get_system_prompt()
        initial_user_message = self.constraint_generator.get_user_prompt(command)

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"--- Iteration {iteration} ---")

            try:
                # Step 1: Generate constraints via LLM with auto-retry
                constraint_code, function_name, attempt_log = (
                    self._generate_constraints_with_retry(
                        system_prompt,
                        initial_user_message,
                        context,
                        command,
                        images=self.current_images,
                    )
                )

                # Step 2: Solve optimization (no warmstart from previous solution)
                optimization_result = self._solve_trajectory_optimization(
                    constraint_code,
                    function_name,
                    iteration,
                    run_dir,
                )

                # Step 3: Execute simulation
                simulation_result = self._execute_simulation(
                    optimization_result, iteration, run_dir
                )

                # Step 4: Extract 20 video frames
                self.current_images = create_visual_feedback(run_dir, iteration)

                # Step 5: LLM summarizes frames
                self.current_visual_summary = self.llm_client.summarize_frames(
                    self.current_images, command
                )

                # === Step 6: Unified scoring ===
                trajectory_analysis = optimization_result.get("trajectory_analysis", {})
                opt_success = optimization_result.get("success", False)
                error_info = optimization_result.get("optimization_metrics", {})

                llm_eval = evaluate_iteration_unified(
                    command=command,
                    trajectory_analysis=trajectory_analysis,
                    constraint_code=constraint_code,
                    opt_success=opt_success,
                    error_info=error_info if not opt_success else None,
                    images=self.current_images,
                    visual_summary=self.current_visual_summary,
                )
                score = llm_eval.get("score", 0.0 if not opt_success else 0.5)

                # Log evaluation
                logger.info(
                    f"=== LLM Evaluation ({'SUCCESS' if opt_success else 'FAILED'}) ==="
                )
                logger.info(f"  Score: {score:.2f}")
                for criterion in llm_eval.get("criteria", []):
                    progress = criterion.get("progress", 0)
                    status = "+" if progress >= 0.8 else "-"
                    logger.info(
                        f"    {status} {criterion.get('name')}: {criterion.get('achieved')} "
                        f"(target: {criterion.get('target')}, {progress:.0%})"
                    )
                if llm_eval.get("warnings"):
                    for warning in llm_eval.get("warnings", []):
                        logger.info(f"    ! {warning}")
                summary = llm_eval.get("summary", f"Iteration {iteration} completed")
                logger.info(f"  Summary: {summary}")

                # Step 7: Append score to recent_scores
                self.recent_scores.append(score)

                # Step 8: Compute pivot/tweak signal
                pivot_signal = self._compute_pivot_signal(iteration)
                logger.info(
                    f"Pivot logic: scores={[f'{s:.2f}' for s in self.recent_scores[-5:]]}, "
                    f"pivot_signal={pivot_signal}"
                )

                # Step 9: Extract constraint violations
                constraint_violations = self._extract_constraint_violations(
                    optimization_result
                )

                # Step 10: Constraint feedback + Reference feedback in parallel
                hardness_report = optimization_result.get(
                    "optimization_metrics", {}
                ).get("hardness_report")
                mpc_dt = float(self.config.mpc_config.mpc_dt)
                current_slack_weights = getattr(self, "current_slack_weights", None)
                hardness_text = format_hardness_report(
                    hardness_report,
                    dt=mpc_dt,
                    current_slack_weights=current_slack_weights,
                )

                ref_trajectory_data = optimization_result.get("ref_trajectory_data")
                state_trajectory = optimization_result.get("state_trajectory")

                with ThreadPoolExecutor(max_workers=2) as executor:
                    constraint_future = executor.submit(
                        generate_constraint_feedback,
                        command=command,
                        constraint_code=constraint_code,
                        images=self.current_images,
                        visual_summary=self.current_visual_summary,
                        hardness_report=hardness_text,
                        constraint_violations=constraint_violations,
                        trajectory_analysis=trajectory_analysis,
                        opt_success=opt_success,
                        error_info=error_info if not opt_success else None,
                        pivot_signal=pivot_signal,
                    )
                    reference_future = executor.submit(
                        generate_reference_feedback,
                        command=command,
                        constraint_code=constraint_code,
                        images=self.current_images,
                        visual_summary=self.current_visual_summary,
                        ref_trajectory_data=ref_trajectory_data,
                        trajectory_analysis=trajectory_analysis,
                        state_trajectory=state_trajectory,
                        opt_success=opt_success,
                        pivot_signal=pivot_signal,
                        mpc_dt=mpc_dt,
                    )
                    constraint_fb = constraint_future.result()
                    ref_fb = reference_future.result()

                logger.info(
                    f"Dual feedback: constraint={len(constraint_fb)} chars, "
                    f"reference={len(ref_fb)} chars"
                )

                # Step 11: Iteration summary LLM call
                iter_summary = generate_iteration_summary(
                    command=command,
                    iteration=iteration,
                    score=score,
                    constraint_code=constraint_code,
                    constraint_feedback=constraint_fb,
                    reference_feedback=ref_fb,
                    trajectory_analysis=trajectory_analysis,
                    opt_success=opt_success,
                    simulation_result=simulation_result,
                    images=self.current_images,
                )
                self.iteration_summaries.append(iter_summary)

                # Step 12: Assemble feedback context (new format)
                feedback_context = self._create_feedback_context(
                    iteration,
                    command,
                    optimization_result,
                    simulation_result,
                    constraint_code,
                    run_dir,
                    pivot_signal=pivot_signal,
                    constraint_feedback=constraint_fb,
                    reference_feedback=ref_fb,
                    visual_summary=self.current_visual_summary,
                    score=score,
                )

                # Step 13: Track best result
                is_new_best = score > best_score
                if is_new_best:
                    best_score = score

                # Step 14: Save iteration results
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

                if is_new_best:
                    best_result = iteration_result

                self._save_iteration_results(iteration_result, run_dir)

                # Step 15: Set context for next iteration
                self.previous_score = score
                if iteration < self.max_iterations:
                    context = feedback_context

                # Step 16: Early stop if score > 0.95
                if score > 0.95:
                    logger.info(f"Early stop: score {score:.2f}")
                    break

            except Exception as e:
                logger.error(f"Iteration {iteration} error: {e}")

                # Generate LLM-driven summary for failed iteration
                code = constraint_code if "constraint_code" in locals() else ""
                try:
                    error_summary = generate_iteration_summary(
                        command=command,
                        iteration=iteration,
                        score=0.0,
                        constraint_code=code,
                        constraint_feedback="",
                        reference_feedback="",
                        trajectory_analysis={},
                        opt_success=False,
                        simulation_result={
                            "success": False,
                            "error": str(e),
                        },
                        images=self.current_images if self.current_images else None,
                    )
                except Exception as summary_err:
                    logger.error(f"Error summary generation failed: {summary_err}")
                    error_summary = {
                        "iteration": iteration,
                        "score": 0.0,
                        "success": False,
                        "constraint_approach": "Summary generation failed",
                        "reference_approach": "Summary generation failed",
                        "constraint_feedback_summary": "",
                        "reference_feedback_summary": "",
                        "simulation_summary": "",
                        "metrics_summary": str(e),
                    }
                self.iteration_summaries.append(error_summary)
                summary = error_summary.get("simulation_summary", str(e))
                self.recent_scores.append(0.0)

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
