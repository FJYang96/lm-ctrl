"""Main feedback pipeline implementing Algorithm 1 from the PDF."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from mpc.dynamics.model import KinoDynamic_Model
from mpc.mpc_opti import QuadrupedMPCOpti

from ..code_generation.prompts import (
    get_robot_details,
    get_system_prompt,
    get_user_prompt,
)
from ..executor import SafeConstraintExecutor
from ..feedback.llm_calls import (
    evaluate_iteration_unified,
    generate_iteration_summary,
    generate_unified_feedback,
)
from ..feedback.motion_analysis import compute_motion_quality_report
from ..feedback.reference_feedback import _compute_reference_metrics
from ..logging_config import logger
from ..mpc import LLMTaskMPC
from .constraint_generation import generate_constraints_with_retry
from .optimization import solve_trajectory_optimization
from .simulation import (
    execute_simulation,
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
    4. Unified feedback and iteration
    """

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
        self.robot_details = get_robot_details(self.config)
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
        self.max_iterations = int(os.getenv("MAX_LLM_ITERATIONS", "5"))
        self.results_dir = Path(os.getenv("RESULTS_DIR", "results/llm_iterations"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Enhanced feedback tracking
        self.current_motion_quality_report: str = ""

        # LLM-based iteration history (structured summaries)
        self.iteration_summaries: list[dict[str, Any]] = []

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

        # Stagnation: best score in last 5 hasn't improved by 0.05 over first
        if len(scores) >= 5:
            window = scores[-5:]
            improvement = max(window) - window[0]
            if improvement < 0.05:
                logger.info(
                    f"Pivot trigger: stagnation (best improvement {improvement:.3f} over 5 iterations)"
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
        opt_success = optimization_result["success"]
        assert self.current_task_mpc is not None
        mpc = self.current_task_mpc.mpc
        assert mpc is not None

        # Get system constraint violations
        constraint_violations = mpc.get_constraint_violations()

        # Get LLM constraint violations
        if opt_success:
            X_val = mpc.opti.value(mpc.X)
            U_val = mpc.opti.value(mpc.U)
        else:
            X_val = mpc.opti.debug.value(mpc.X)
            U_val = mpc.opti.debug.value(mpc.U)

        llm_violations = self.current_task_mpc.evaluate_constraint_violations(
            X_val, U_val
        )
        constraint_violations["llm_constraints"] = llm_violations["llm_constraints"]
        constraint_violations["llm_summary"] = llm_violations["summary"]

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
        self.iteration_summaries = []
        self.current_motion_quality_report = ""
        feedback_data: dict[str, Any] | None = None

        # Reset score tracking
        self.previous_score = -float("inf")
        self.recent_scores = []

        # Algorithm 1: Iterative Refinement Pipeline
        system_prompt = get_system_prompt(self.config)
        initial_user_message = get_user_prompt(command)

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"--- Iteration {iteration} ---")

            try:
                # Step 1: Generate constraints via LLM with auto-retry
                (constraint_code, function_name, attempt_log, feedback_context) = (
                    generate_constraints_with_retry(
                        self,
                        system_prompt,
                        initial_user_message,
                        feedback_data,
                        command,
                    )
                )

                # Save feedback context to disk (describes previous iteration)
                if feedback_context is not None:
                    ctx_file = run_dir / f"codegen_prompt_iter_{iteration}.txt"
                    with open(ctx_file, "w") as f:
                        f.write(feedback_context)

                # Step 2: Solve optimization (no warmstart from previous solution)
                optimization_result = solve_trajectory_optimization(
                    self,
                    constraint_code,
                    function_name,
                    iteration,
                    run_dir,
                )

                # Step 3: Execute simulation
                simulation_result = execute_simulation(
                    self, optimization_result, iteration, run_dir
                )

                # === Step 5: Extract constraint data (needed by scoring + feedback) ===
                trajectory_analysis = optimization_result["trajectory_analysis"]
                opt_success = optimization_result["success"]
                error_info = optimization_result["optimization_metrics"]

                constraint_violations = self._extract_constraint_violations(
                    optimization_result
                )

                hardness_report = optimization_result["optimization_metrics"][
                    "hardness_report"
                ]
                mpc_dt = float(self.config.mpc_config.mpc_dt)

                # Compute reference analysis (used by scoring, feedback, and summary)
                ref_analysis = _compute_reference_metrics(
                    optimization_result["ref_trajectory_data"],
                    optimization_result["state_trajectory"],
                    mpc_dt,
                )

                # Step 5b: Compute motion quality report (pure computation, no LLM)
                import numpy as np

                self.current_motion_quality_report = compute_motion_quality_report(
                    state_traj=optimization_result["state_trajectory"],
                    grf_traj=optimization_result["grf_trajectory"],
                    joint_vel_traj=optimization_result["joint_vel_trajectory"],
                    mpc_dt=mpc_dt,
                    contact_sequence=self.current_task_mpc.contact_sequence,  # type: ignore[union-attr]
                    kindyn_model=self.kindyn_model,
                    joint_limits_lower=np.array(
                        self.robot_details["joint_limits_lower"]
                    ),
                    joint_limits_upper=np.array(
                        self.robot_details["joint_limits_upper"]
                    ),
                    robot_mass=self.robot_details["mass"],
                    mu_friction=float(self.config.experiment.mu_ground),
                )

                # === Step 6: Unified scoring ===
                llm_eval = evaluate_iteration_unified(
                    command=command,
                    trajectory_analysis=trajectory_analysis,
                    constraint_code=constraint_code,
                    opt_success=opt_success,
                    error_info=error_info if not opt_success else None,
                    motion_quality_report=self.current_motion_quality_report,
                    hardness_report=hardness_report,
                    mpc_dt=mpc_dt,
                    current_slack_weights=self.current_slack_weights,
                    constraint_violations=constraint_violations,
                    reference_analysis=ref_analysis,
                    run_dir=run_dir,
                    iteration=iteration,
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

                # Step 8: Append score to recent_scores
                self.recent_scores.append(score)

                # Step 9: Compute pivot/tweak signal
                pivot_signal = self._compute_pivot_signal(iteration)
                logger.info(
                    f"Pivot logic: scores={[f'{s:.2f}' for s in self.recent_scores[-5:]]}, "
                    f"pivot_signal={pivot_signal}"
                )

                # Step 9: Unified feedback (single Claude call — receives motion quality report)
                unified_fb = generate_unified_feedback(
                    command=command,
                    constraint_code=constraint_code,
                    motion_quality_report=self.current_motion_quality_report,
                    hardness_report=hardness_report,
                    mpc_dt=mpc_dt,
                    current_slack_weights=self.current_slack_weights,
                    constraint_violations=constraint_violations,
                    trajectory_analysis=trajectory_analysis,
                    opt_success=opt_success,
                    error_info=error_info if not opt_success else None,
                    pivot_signal=pivot_signal,
                    reference_analysis=ref_analysis,
                    run_dir=run_dir,
                    iteration=iteration,
                    iteration_summaries=list(self.iteration_summaries),
                )

                logger.info(f"Unified feedback: {len(unified_fb)} chars")

                # Step 10: Iteration summary LLM call
                iter_summary = generate_iteration_summary(
                    command=command,
                    iteration=iteration,
                    score=score,
                    constraint_code=constraint_code,
                    feedback=unified_fb,
                    trajectory_analysis=trajectory_analysis,
                    opt_success=opt_success,
                    error_info=error_info if not opt_success else None,
                    simulation_result=simulation_result,
                    hardness_report=hardness_report,
                    mpc_dt=mpc_dt,
                    current_slack_weights=self.current_slack_weights,
                    reference_analysis=ref_analysis,
                    constraint_violations=constraint_violations,
                    run_dir=run_dir,
                )
                # Step 11: Collect feedback data for next iteration
                # (snapshot iteration_summaries BEFORE appending current summary,
                # so current iteration's summary only shows in future history)
                feedback_data = {
                    "iteration": iteration,
                    "command": command,
                    "optimization_result": optimization_result,
                    "simulation_result": simulation_result,
                    "constraint_code": constraint_code,
                    "run_dir": run_dir,
                    "iteration_summaries": list(self.iteration_summaries),
                    "mpc_dt": float(self.config.mpc_config.mpc_dt),
                    "current_slack_weights": self.current_slack_weights,
                    "pivot_signal": pivot_signal,
                    "feedback": unified_fb,
                    "score": score,
                    "motion_quality_report": self.current_motion_quality_report,
                    "constraint_violations": constraint_violations,
                }
                self.iteration_summaries.append(iter_summary)

                # Update score tracking
                self.previous_score = score

                # Step 16: Early stop if score > 0.95
                if score > 0.95:
                    logger.info(f"Early stop: score {score:.2f}")
                    break

            except Exception as e:
                logger.error(f"Iteration {iteration} error: {e}")

                error_summary = {
                    "iteration": iteration,
                    "score": 0.0,
                    "success": False,
                    "approach": f"Iteration failed: {e}",
                    "feedback_summary": "",
                    "simulation_summary": "",
                    "metrics_summary": str(e),
                }
                self.iteration_summaries.append(error_summary)
                self.recent_scores.append(0.0)

                # Build error-path feedback data so the next iteration
                # sees this failure in the iteration history.
                code = constraint_code if "constraint_code" in locals() else ""
                pivot_signal = self._compute_pivot_signal(iteration)
                feedback_data = {
                    "iteration": iteration,
                    "command": command,
                    "optimization_result": {
                        "success": False,
                        "trajectory_analysis": {},
                        "optimization_metrics": {"hardness_report": None},
                        "ref_trajectory_data": None,
                        "state_trajectory": None,
                    },
                    "simulation_result": {
                        "success": False,
                        "error": str(e),
                    },
                    "constraint_code": code,
                    "run_dir": run_dir,
                    "iteration_summaries": list(self.iteration_summaries),
                    "mpc_dt": float(self.config.mpc_config.mpc_dt),
                    "current_slack_weights": self.current_slack_weights,
                    "pivot_signal": pivot_signal,
                    "feedback": "",
                    "score": 0.0,
                }

                continue

        # Compile final results
        best_score = max(self.recent_scores) if self.recent_scores else 0.0
        final_results = {
            "command": command,
            "total_iterations": len(self.recent_scores),
            "best_score": best_score,
            "results_directory": str(run_dir),
            "pipeline_success": best_score > 0.5,
        }

        logger.info(f"Pipeline complete: best_score={best_score:.2f}")

        return final_results
