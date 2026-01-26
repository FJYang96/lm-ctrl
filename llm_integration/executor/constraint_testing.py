"""Constraint function testing utilities."""

from typing import Any

import casadi as cs


class MockKindyn:
    """Mock kinodynamics model for testing constraint functions."""

    def forward_kinematics_FL_fun(self, H: cs.MX, joint_pos: cs.MX) -> cs.MX:
        return cs.MX.eye(4)


class MockConfig:
    """Mock config for testing constraint functions."""

    def __init__(self) -> None:
        self.mpc_config = type("obj", (object,), {"mpc_dt": 0.02})()
        self.experiment = type("obj", (object,), {"mu_ground": 0.8})()
        self.robot_data = type("obj", (object,), {"mass": 12.0})()


def test_constraint_function_execution(
    constraint_func: Any, func_name: str
) -> tuple[bool, str]:
    """
    Test a constraint function with mock inputs to catch early errors.

    Args:
        constraint_func: The constraint function to test
        func_name: Name of the function for error messages

    Returns:
        Tuple of (success, error_message)
    """
    try:
        test_x = cs.MX.sym("test_x", 24)
        test_u = cs.MX.sym("test_u", 24)
        test_contact = cs.MX.sym("test_contact", 4)
        test_k = 10  # Sample timestep
        test_horizon = 75  # Sample horizon

        # Try new 7-argument signature first, fall back to old 5-argument
        try:
            result = constraint_func(
                test_x,
                test_u,
                MockKindyn(),
                MockConfig(),
                test_contact,
                test_k,
                test_horizon,
            )
        except TypeError:
            result = constraint_func(
                test_x, test_u, MockKindyn(), MockConfig(), test_contact
            )

        # Validate return format
        if not isinstance(result, tuple) or len(result) != 3:
            return (
                False,
                f"Function '{func_name}' must return tuple of 3 elements, got {type(result)} with {len(result) if hasattr(result, '__len__') else 'unknown'} elements",
            )

        expr, lower, upper = result

        # Validate that returned values are CasADi expressions
        if expr is not None:
            if not hasattr(expr, "size"):
                return (
                    False,
                    f"First return value must be a CasADi expression with .size() method, got {type(expr)}",
                )

            # Check bounds have correct dimensions
            success, error = validate_constraint_dimensions(
                expr, lower, upper, func_name
            )
            if not success:
                return False, error

        return True, ""

    except Exception as test_error:
        return False, f"Function test call failed: {test_error}"


def validate_constraint_dimensions(
    expr: Any, lower: Any, upper: Any, func_name: str
) -> tuple[bool, str]:
    """
    Validate that constraint expression and bounds have matching dimensions.

    Args:
        expr: Constraint expression
        lower: Lower bound
        upper: Upper bound
        func_name: Function name for error messages

    Returns:
        Tuple of (success, error_message)
    """
    try:
        expr_size = expr.size()
        if len(expr_size) > 1:
            expr_length = expr_size[0]
        else:
            expr_length = 1

        # Check lower bound
        if hasattr(lower, "size"):
            lower_size = lower.size()
            lower_length = lower_size[0] if len(lower_size) > 1 else 1
            if lower_length != expr_length:
                return (
                    False,
                    f"Lower bound dimension ({lower_length}) must match constraint dimension ({expr_length})",
                )

        # Check upper bound
        if hasattr(upper, "size"):
            upper_size = upper.size()
            upper_length = upper_size[0] if len(upper_size) > 1 else 1
            if upper_length != expr_length:
                return (
                    False,
                    f"Upper bound dimension ({upper_length}) must match constraint dimension ({expr_length})",
                )

        return True, ""

    except Exception as dim_error:
        return False, f"Could not verify dimensions: {dim_error}"
