"""Constraint wrapping and violation evaluation utilities."""

from __future__ import annotations

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
    sig = inspect.signature(constraint_func)
    num_params = len(sig.parameters)

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


def _to_float_array(val: Any) -> np.ndarray:
    """Convert a scalar, CasADi DM/MX, or numpy value to a flat float array."""
    # CasADi MX (symbolic) — evaluate constant expressions to DM first
    try:
        import casadi as cs

        if isinstance(val, cs.MX):
            val = cs.evalf(val)
    except ImportError:
        pass
    if hasattr(val, "full"):
        return np.asarray(val.full(), dtype=float).flatten()
    return np.atleast_1d(np.asarray(val, dtype=float)).flatten()


def evaluate_constraint_violations(
    constraint_functions: list[Any],
    contact_sequence: np.ndarray,
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

    horizon = contact_sequence.shape[1]

    # Track violations per constraint
    for i in range(len(constraint_functions)):
        violations["by_constraint"][i] = []

    # Evaluate each constraint at each timestep (skip k=0 as MPC does)
    for k in range(1, horizon):
        x_k = X_debug[:, k]
        u_k = U_debug[:, k]
        contact_k = contact_sequence[:, k]

        for i, constraint_func in enumerate(constraint_functions):
            # Call the constraint function
            expr_value, lower, upper = constraint_func(
                x_k,
                u_k,
                kindyn_model,
                base_config,
                contact_k,
                k,
                horizon,
            )

            # Convert to numpy arrays to handle both scalar and vector constraints
            expr_arr = _to_float_array(expr_value)
            lower_arr = _to_float_array(lower)
            upper_arr = _to_float_array(upper)

            # Check element-wise violations
            for j in range(len(expr_arr)):
                v = expr_arr[j]
                lb = lower_arr[j] if j < len(lower_arr) else lower_arr[0]
                ub = upper_arr[j] if j < len(upper_arr) else upper_arr[0]

                suffix = f"[{j}]" if len(expr_arr) > 1 else ""

                if v < lb:
                    violation_msg = (
                        f"Constraint {i}{suffix} at k={k}: "
                        f"value={v:.6f} < lower={lb:.6f}"
                    )
                    violations["llm_constraints"].append(violation_msg)
                    violations["by_constraint"][i].append(
                        {
                            "k": k,
                            "element": j,
                            "value": v,
                            "lower": lb,
                            "type": "below_lower",
                        }
                    )

                if v > ub:
                    violation_msg = (
                        f"Constraint {i}{suffix} at k={k}: "
                        f"value={v:.6f} > upper={ub:.6f}"
                    )
                    violations["llm_constraints"].append(violation_msg)
                    violations["by_constraint"][i].append(
                        {
                            "k": k,
                            "element": j,
                            "value": v,
                            "upper": ub,
                            "type": "above_upper",
                        }
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
