"""Utility functions for the feedback pipeline."""

import ast
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..logging_config import logger

if TYPE_CHECKING:
    from .feedback_pipeline import FeedbackPipeline


def save_iteration_results(
    self: "FeedbackPipeline", iteration_result: dict[str, Any], run_dir: Path
) -> None:
    """Save detailed results for a single iteration."""
    iteration = iteration_result["iteration"]

    # Save iteration summary (JSON-safe version)
    iteration_file = run_dir / f"iteration_{iteration}.json"
    json_safe_result = make_json_safe(self, iteration_result)

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

    # Save reference trajectory function code separately
    if "constraint_code" in iteration_result:
        ref_code = _extract_ref_trajectory_code(iteration_result["constraint_code"])
        if ref_code:
            ref_file = run_dir / f"ref_trajectory_iter_{iteration}.py"
            with open(ref_file, "w") as f:
                f.write(ref_code)

    # Save feedback context as readable text file for debugging
    if "feedback_context" in iteration_result and iteration_result["feedback_context"]:
        feedback_file = run_dir / f"feedback_iter_{iteration}.txt"
        with open(feedback_file, "w") as f:
            f.write(iteration_result["feedback_context"])


def _extract_ref_trajectory_code(full_code: str) -> str:
    """Extract the reference trajectory function from the full LLM code."""
    try:
        tree = ast.parse(full_code)
    except SyntaxError:
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and (
            "reference" in node.name or node.name.startswith("generate_")
        ):
            start = node.lineno - 1
            end = node.end_lineno
            lines = full_code.splitlines()
            return "\n".join(lines[start:end]) + "\n"
    return ""


def inject_llm_constraints_to_mpc(self: "FeedbackPipeline") -> None:
    """
    Inject LLM constraints into the MPC problem using the proper MPC constraint interface.

    This adds the LLM-generated constraints as additional path constraints to the
    existing MPC constraint system, rather than directly to the Opti problem.
    """
    if not self.llm_constraints:
        return

    constraint_func = self.llm_constraints[0]

    try:
        if hasattr(self.config, "mpc_config") and hasattr(
            self.config.mpc_config, "path_constraints"
        ):
            self.config.mpc_config.path_constraints.append(constraint_func)
        else:
            inject_llm_constraints_direct(self)

    except Exception as e:
        logger.error(f"Constraint injection failed: {e}")


def inject_llm_constraints_direct(self: "FeedbackPipeline") -> None:
    """
    Fallback method: Inject LLM constraints directly into the Opti problem.
    This is used when the preferred path constraint method fails.
    """
    if not self.llm_constraints:
        return

    constraint_func = self.llm_constraints[0]

    try:
        constraints_added = 0
        for k in range(self.mpc.horizon):
            try:
                x_k = self.mpc.X[:, k]
                u_k = self.mpc.U[:, k]
                contact_k = self.mpc.P_contact[:, k]

                constraint_result = constraint_func(
                    x_k, u_k, self.kindyn_model, self.config, contact_k
                )

                if (
                    not isinstance(constraint_result, tuple)
                    or len(constraint_result) != 3
                ):
                    continue

                constraint_expr, constraint_l, constraint_u = constraint_result

                if constraint_expr is not None:
                    self.mpc.opti.subject_to(constraint_expr >= constraint_l)
                    self.mpc.opti.subject_to(constraint_expr <= constraint_u)
                    constraints_added += 1

            except Exception:
                continue

    except Exception as e:
        logger.error(f"Direct injection failed: {e}")


def make_json_safe(self: "FeedbackPipeline", obj: Any) -> Any:
    """Convert numpy arrays and other non-JSON-serializable objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_safe(self, v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(self, item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    else:
        return obj
