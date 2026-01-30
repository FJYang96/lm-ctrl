"""Constraint wrapping and violation evaluation utilities."""

import inspect
from collections.abc import Callable
from typing import Any, cast

import numpy as np


def wrap_constraint_for_contact_phases(
    constraint_func: Callable[..., Any],
) -> Callable[..., Any]:
    """
    Wrap LLM-generated constraints to be contact-aware.

    Supports both 5-argument and 7-argument constraint signatures:
    - 5 args: (x_k, u_k, kindyn_model, config, contact_k)
    - 7 args: (x_k, u_k, kindyn_model, config, contact_k, k, horizon)

    The LLM is responsible for generating constraints that are appropriate
    for the task's contact sequence.
    """
    # Detect the number of parameters in the constraint function
    try:
        sig = inspect.signature(constraint_func)
        num_params = len(sig.parameters)
    except (ValueError, TypeError):
        num_params = 5  # Default to 5-arg signature

    def contact_aware_constraint(
        x_k: Any,
        u_k: Any,
        kindyn_model: Any,
        config: Any,
        contact_k: Any,
        k: int = 0,
        horizon: int = 1,
    ) -> tuple[Any, Any, Any]:
        # Call with appropriate number of arguments based on function signature
        if num_params >= 7:
            return cast(
                tuple[Any, Any, Any],
                constraint_func(x_k, u_k, kindyn_model, config, contact_k, k, horizon),
            )
        else:
            return cast(
                tuple[Any, Any, Any],
                constraint_func(x_k, u_k, kindyn_model, config, contact_k),
            )

    return contact_aware_constraint


def evaluate_constraint_violations(
    constraint_functions: list[Any],
    contact_sequence: np.ndarray | None,
    kindyn_model: Any,
    base_config: Any,
    X_debug: np.ndarray,
    U_debug: np.ndarray,
) -> dict[str, Any]:
    """
    Evaluate LLM-generated constraints against a (possibly failed) trajectory.

    This helps diagnose why optimization failed by showing exactly which
    LLM constraints are violated and at which timesteps.

    Args:
        constraint_functions: List of constraint functions
        contact_sequence: Contact sequence array (4 x horizon)
        kindyn_model: Robot kinodynamic model
        base_config: Configuration object
        X_debug: State trajectory (states_dim x horizon+1) from opti.debug.value(X)
        U_debug: Input trajectory (inputs_dim x horizon) from opti.debug.value(U)

    Returns:
        Dictionary with constraint violation details
    """
    violations: dict[str, Any] = {
        "llm_constraints": [],
        "by_constraint": {},
        "summary": [],
    }

    if not constraint_functions:
        violations["summary"].append("No LLM constraints to evaluate")
        return violations

    if contact_sequence is None:
        violations["summary"].append("No contact sequence configured")
        return violations

    horizon = contact_sequence.shape[1]

    # Track violations per constraint
    for i in range(len(constraint_functions)):
        violations["by_constraint"][i] = []

    # Evaluate each constraint at each timestep (skip k=0 as MPC does)
    for k in range(1, min(horizon, X_debug.shape[1])):
        x_k = X_debug[:, k]
        u_k = U_debug[:, k] if k < U_debug.shape[1] else U_debug[:, -1]
        contact_k = contact_sequence[:, k]

        for i, constraint_func in enumerate(constraint_functions):
            try:
                # Call the constraint function
                result = constraint_func(
                    x_k,
                    u_k,
                    kindyn_model,
                    base_config,
                    contact_k,
                    k,
                    horizon,
                )

                # Extract value, lower, upper bounds
                if isinstance(result, tuple) and len(result) == 3:
                    expr_value, lower, upper = result

                    # Convert CasADi types to float if needed
                    try:
                        if hasattr(expr_value, "full"):
                            expr_value = float(expr_value.full().flatten()[0])
                        elif hasattr(expr_value, "__float__"):
                            expr_value = float(expr_value)
                    except Exception:
                        pass  # Keep as-is if conversion fails

                    try:
                        if hasattr(lower, "__float__"):
                            lower = float(lower)
                        if hasattr(upper, "__float__"):
                            upper = float(upper)
                    except Exception:
                        pass

                    # Check for violations
                    if isinstance(expr_value, (int, float)) and isinstance(
                        lower, (int, float)
                    ):
                        if expr_value < lower:
                            violation_msg = (
                                f"Constraint {i} at k={k}: "
                                f"value={expr_value:.6f} < lower={lower:.6f}"
                            )
                            violations["llm_constraints"].append(violation_msg)
                            violations["by_constraint"][i].append(
                                {
                                    "k": k,
                                    "value": expr_value,
                                    "lower": lower,
                                    "type": "below_lower",
                                }
                            )

                    if isinstance(expr_value, (int, float)) and isinstance(
                        upper, (int, float)
                    ):
                        if expr_value > upper:
                            violation_msg = (
                                f"Constraint {i} at k={k}: "
                                f"value={expr_value:.6f} > upper={upper:.6f}"
                            )
                            violations["llm_constraints"].append(violation_msg)
                            violations["by_constraint"][i].append(
                                {
                                    "k": k,
                                    "value": expr_value,
                                    "upper": upper,
                                    "type": "above_upper",
                                }
                            )

            except Exception as e:
                # Record evaluation error but continue
                violations["llm_constraints"].append(
                    f"Constraint {i} at k={k}: evaluation error - {str(e)[:50]}"
                )

    # Generate summary
    for i, constraint_violations in violations["by_constraint"].items():
        if constraint_violations:
            violations["summary"].append(
                f"LLM constraint {i}: violated at {len(constraint_violations)} timesteps"
            )

    if not violations["summary"]:
        violations["summary"].append(
            "No LLM constraint violations detected in evaluated trajectory"
        )

    return violations
