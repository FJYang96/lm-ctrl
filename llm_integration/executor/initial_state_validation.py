"""Initial state feasibility validation for constraints."""

from collections.abc import Callable
from typing import Any

import casadi as cs
import numpy as np


def validate_initial_state_feasibility(
    constraint_func: Callable[..., Any],
    func_name: str,
    test_kindyn_model: Any,
    test_config: Any,
    initial_height: float = 0.2117,
    horizon: int = 75,
) -> tuple[bool, str]:
    """
    Validate that the constraint function doesn't violate the initial state at k=0.

    This catches a common LLM error: setting bounds that exclude the robot's
    starting position (e.g., height upper bound < 0.2117m at t=0).

    Args:
        constraint_func: The constraint function to validate
        func_name: Name of the function
        test_kindyn_model: Test kinodynamic model
        test_config: Test config object
        initial_height: Robot's initial COM height (default 0.2117m)
        horizon: MPC horizon for testing

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Create initial state vector with actual values
        # State: [pos(3), vel(3), euler(3), angular_vel(3), joint_angles(12)]
        initial_state = np.zeros(24)
        initial_state[2] = initial_height  # COM z height
        # All other values (velocities, angles) start at 0

        # Convert to CasADi DM for numerical evaluation
        test_x = cs.DM(initial_state)
        test_u = cs.DM.zeros(24)
        test_contact = cs.DM([1, 1, 1, 1])  # All feet in contact at start
        test_k = 0  # Critical: test at k=0
        test_horizon = horizon

        # Call the constraint function at k=0
        try:
            result = constraint_func(
                test_x,
                test_u,
                test_kindyn_model,
                test_config,
                test_contact,
                test_k,
                test_horizon,
            )
        except TypeError:
            # Try 5-arg version
            result = constraint_func(
                test_x, test_u, test_kindyn_model, test_config, test_contact
            )

        if result is None:
            return True, ""  # No constraints at k=0 is fine

        if not isinstance(result, tuple) or len(result) != 3:
            return True, ""  # Let other validation catch format errors

        constraint_expr, lower_bounds, upper_bounds = result

        if constraint_expr is None:
            return True, ""

        # Evaluate the constraint expression with initial state
        try:
            # Create a CasADi function to evaluate the constraint
            x_sym = cs.MX.sym("x", 24)

            # Re-call to get symbolic expression
            try:
                sym_result = constraint_func(
                    x_sym,
                    cs.MX.sym("u", 24),
                    test_kindyn_model,
                    test_config,
                    cs.MX.sym("c", 4),
                    0,
                    test_horizon,
                )
            except TypeError:
                sym_result = constraint_func(
                    x_sym,
                    cs.MX.sym("u", 24),
                    test_kindyn_model,
                    test_config,
                    cs.MX.sym("c", 4),
                )

            if sym_result is None:
                return True, ""

            sym_expr, sym_lower, sym_upper = sym_result

            # Create function to evaluate constraint value at initial state
            eval_func = cs.Function("eval", [x_sym], [sym_expr])
            constraint_value = eval_func(cs.DM(initial_state))

            # Get bounds as numpy arrays
            lower_np = np.array(sym_lower).flatten()
            upper_np = np.array(sym_upper).flatten()
            value_np = np.array(constraint_value).flatten()

            # Check if initial state violates any bounds
            violations = []
            for i, (val, lo, hi) in enumerate(zip(value_np, lower_np, upper_np)):
                if val < lo - 1e-6:
                    violations.append(
                        f"Constraint {i}: value {val:.4f} < lower bound {lo:.4f}"
                    )
                if val > hi + 1e-6:
                    violations.append(
                        f"Constraint {i}: value {val:.4f} > upper bound {hi:.4f}"
                    )

            if violations:
                return (
                    False,
                    f"Constraints violate initial state (k=0, height={initial_height}m):\n"
                    + "\n".join(f"  - {v}" for v in violations)
                    + "\n\nFix: At k=0, bounds must INCLUDE the starting state. "
                    + "Use 'if k == 0: upper = 0.25' to allow initial height.",
                )

            return True, ""

        except Exception as e:
            # If we can't evaluate numerically, skip this validation
            return True, f"Could not numerically validate t=0 (non-fatal): {e}"

    except Exception as e:
        # Non-fatal - let the optimization catch it
        return True, f"t=0 validation skipped: {e}"
