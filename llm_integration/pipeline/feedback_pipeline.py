"""Main feedback pipeline implementing iterative LLM refinement."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

import go2_config as config
from mpc.dynamics.model import KinoDynamic_Model

from ..code_generation.prompts import get_system_prompt, get_user_prompt
from ..executor import SafeConstraintExecutor
from ..feedback.motion_analysis import compute_motion_quality_report
from ..feedback.reference_feedback import _compute_reference_metrics
from ..feedback.scoring import evaluate_iteration_unified
from ..feedback.summary import generate_iteration_summary
from ..logging_config import logger
from ..mpc import LLMTaskMPC
from .constraint_generation import generate_constraints_with_retry
from .optimization import solve_trajectory_optimization
from .simulation import execute_simulation

try:
    from gym_quadruped.quadruped_env import QuadrupedEnv
except ImportError:
    QuadrupedEnv = None


class FeedbackPipeline:
    """Iterative LLM refinement loop for quadruped trajectory optimization."""

    def __init__(self, use_slack: bool = True):
        self.use_slack = use_slack
        self.safe_executor = SafeConstraintExecutor()
        self.kindyn_model = KinoDynamic_Model()
        self.current_task_mpc: LLMTaskMPC | None = None
        self.current_slack_weights: dict[str, float] = {}

        if QuadrupedEnv is not None:
            self.env = QuadrupedEnv(
                robot=config.robot, scene="flat",
                ground_friction_coeff=config.experiment.mu_ground,
                state_obs_names=QuadrupedEnv._DEFAULT_OBS + ("contact_forces:base",),
                sim_dt=config.experiment.sim_dt,
            )
        else:
            self.env = None

        self.max_iterations = int(os.getenv("MAX_LLM_ITERATIONS", "5"))
        self.results_dir = Path(os.getenv("RESULTS_DIR", "results/llm_iterations"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.iteration_summaries: list[dict[str, Any]] = []
        self.all_scores: list[float] = []
        self._last_mpc_dt: float = float(config.mpc_config.mpc_dt)

    def _extract_constraint_violations(self, opt_result: dict[str, Any]) -> dict[str, Any]:
        """Extract system + LLM constraint violations from the MPC solver."""
        from mpc.mpc_analysis import get_constraint_violations

        assert self.current_task_mpc is not None
        mpc = self.current_task_mpc.mpc
        assert mpc is not None
        violations = get_constraint_violations(mpc)
        if opt_result["success"]:
            X_val, U_val = mpc.opti.value(mpc.X), mpc.opti.value(mpc.U)
        else:
            X_val, U_val = mpc.opti.debug.value(mpc.X), mpc.opti.debug.value(mpc.U)
        llm = self.current_task_mpc.evaluate_constraint_violations(X_val, U_val)
        violations.update({
            "llm_constraints": llm["llm_constraints"], "llm_summary": llm["summary"],
            "by_constraint": llm["by_constraint"], "constraint_meta": llm["constraint_meta"],
        })
        return violations

    def run_pipeline(self, command: str) -> dict[str, Any]:
        """Run the complete LLM feedback pipeline for a given command."""
        logger.info(f"Pipeline started: '{command}'")
        run_dir = self.results_dir / f"{command.replace(' ', '_')}_{int(time.time())}"
        run_dir.mkdir(exist_ok=True)

        self.iteration_summaries = []
        self.all_scores = []
        feedback_data: dict[str, Any] | None = None
        system_prompt = get_system_prompt()
        user_message = get_user_prompt(command)

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"--- Iteration {iteration} ---")
            constraint_code = ""

            try:
                constraint_code, function_name, _, feedback_context = (
                    generate_constraints_with_retry(
                        self, system_prompt, user_message, feedback_data, command,
                    )
                )
                if feedback_context is not None:
                    (run_dir / f"codegen_prompt_iter_{iteration}.txt").write_text(
                        feedback_context
                    )

                optimization_result = solve_trajectory_optimization(
                    self, constraint_code, function_name, iteration, run_dir,
                )
                simulation_result = execute_simulation(
                    self, optimization_result, iteration, run_dir
                )

                trajectory_analysis = optimization_result["trajectory_analysis"]
                opt_success = optimization_result["success"]
                error_info = optimization_result["optimization_metrics"]
                constraint_violations = self._extract_constraint_violations(
                    optimization_result
                )
                hardness_report = error_info["hardness_report"]

                if self.current_task_mpc is not None:
                    mpc_dt = float(self.current_task_mpc.mpc_dt)
                    self._last_mpc_dt = mpc_dt
                else:
                    mpc_dt = self._last_mpc_dt

                ref_analysis = _compute_reference_metrics(
                    optimization_result["ref_trajectory_data"],
                    optimization_result["state_trajectory"], mpc_dt,
                )

                motion_quality_report = ""
                if optimization_result["state_trajectory"] is not None:
                    motion_quality_report = compute_motion_quality_report(
                        state_traj=optimization_result["state_trajectory"],
                        grf_traj=optimization_result["grf_trajectory"],
                        joint_vel_traj=optimization_result["joint_vel_trajectory"],
                        mpc_dt=mpc_dt,
                        contact_sequence=self.current_task_mpc.contact_sequence,  # type: ignore[union-attr]
                        kindyn_model=self.kindyn_model,
                        joint_limits_lower=np.array(config.urdf_joint_limits_lower),
                        joint_limits_upper=np.array(config.urdf_joint_limits_upper),
                        robot_mass=config.composite_mass,
                        mu_friction=float(config.experiment.mu_ground),
                    )
                    if not opt_success:
                        motion_quality_report = (
                            "!! SOLVER DID NOT CONVERGE — metrics from INFEASIBLE "
                            "iterate. !!\n\n" + motion_quality_report
                        )

                llm_eval = evaluate_iteration_unified(
                    command=command, trajectory_analysis=trajectory_analysis,
                    constraint_code=constraint_code, opt_success=opt_success,
                    error_info=error_info if not opt_success else None,
                    motion_quality_report=motion_quality_report,
                    hardness_report=hardness_report, mpc_dt=mpc_dt,
                    current_slack_weights=self.current_slack_weights,
                    constraint_violations=constraint_violations,
                    reference_analysis=ref_analysis,
                    run_dir=run_dir, iteration=iteration,
                )
                score = llm_eval.get("score", 0.0 if not opt_success else 0.5)
                logger.info(f"Score: {score:.2f} | {llm_eval.get('summary', '')}")
                self.all_scores.append(score)

                iter_summary = generate_iteration_summary(
                    command=command, iteration=iteration, score=score,
                    constraint_code=constraint_code,
                    trajectory_analysis=trajectory_analysis,
                    opt_success=opt_success,
                    error_info=error_info if not opt_success else None,
                    simulation_result=simulation_result,
                    hardness_report=hardness_report, mpc_dt=mpc_dt,
                    current_slack_weights=self.current_slack_weights,
                    reference_analysis=ref_analysis,
                    constraint_violations=constraint_violations,
                    motion_quality_report=motion_quality_report,
                    run_dir=run_dir,
                )
                iter_summary["constraint_code"] = constraint_code
                self.iteration_summaries.append(iter_summary)
                feedback_data = {
                    "iteration": iteration, "command": command,
                    "iteration_summaries": list(self.iteration_summaries),
                    "run_dir": run_dir, "mpc_dt": mpc_dt,
                }

                if score > 0.95:
                    logger.info(f"Early stop: score {score:.2f}")
                    break

            except Exception as e:
                logger.error(f"Iteration {iteration} error: {e}")
                self.iteration_summaries.append({
                    "iteration": iteration, "score": 0.0, "success": False,
                    "approach": f"Iteration failed: {e}", "solver": "failed",
                    "motion_quality": "", "metrics": "", "terminal": "",
                    "hardness": "", "violations": "", "reference": "",
                    "constraint_code": constraint_code,
                })
                self.all_scores.append(0.0)
                feedback_data = {
                    "iteration": iteration, "command": command,
                    "iteration_summaries": list(self.iteration_summaries),
                    "run_dir": run_dir,
                    "mpc_dt": float(
                        self.current_task_mpc.mpc_dt
                        if self.current_task_mpc is not None
                        else self._last_mpc_dt
                    ),
                }
                continue

        best_score = max(self.all_scores) if self.all_scores else 0.0
        logger.info(f"Pipeline complete: best_score={best_score:.2f}")
        return {
            "command": command, "total_iterations": len(self.all_scores),
            "best_score": best_score, "results_directory": str(run_dir),
            "pipeline_success": best_score > 0.5,
        }
