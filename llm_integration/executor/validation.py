"""Validation functions for constraint code execution."""

import traceback
from collections.abc import Callable
from typing import Any

import casadi as cs

from .globals import create_restricted_globals
from .initial_state_validation import validate_initial_state_feasibility

# Common constraint function names to look for
CONSTRAINT_FUNCTION_NAMES = [
    "task_specific_constraints",
    "constraints",
    "get_constraints",
    "apply_constraints",
    "constraint_function",
    "main_constraints",
    "constraint",
]

# Re-export for backwards compatibility
__all__ = [
    "CONSTRAINT_FUNCTION_NAMES",
    "find_constraint_entry_point",
    "validate_constraint_compatibility",
    "validate_initial_state_feasibility",
    "test_constraint_function",
]


def find_constraint_entry_point(
    locals_dict: dict[str, Any],
    globals_dict: dict[str, Any],
    preferred_name: str | None = None,
) -> tuple[Callable[..., Any] | None, str, str]:
    """
    Find the main constraint function entry point in the executed code.

    Args:
        locals_dict: Local namespace from exec
        globals_dict: Global namespace from exec
        preferred_name: Preferred function name if known

    Returns:
        Tuple of (function, function_name, error_message)
    """
    # Combine locals and globals for searching
    all_symbols = {**globals_dict, **locals_dict}

    # Remove built-in symbols to focus on user-defined functions
    restricted_globals = create_restricted_globals()
    user_symbols = {
        k: v
        for k, v in all_symbols.items()
        if callable(v) and not k.startswith("_") and k not in restricted_globals
    }

    if not user_symbols:
        return None, "", "No callable functions found in generated code"

    # If preferred name is specified and exists, use it
    if preferred_name and preferred_name in user_symbols:
        return user_symbols[preferred_name], preferred_name, ""

    # Look for common constraint function names
    for name in CONSTRAINT_FUNCTION_NAMES:
        if name in user_symbols:
            return user_symbols[name], name, ""

    # If no common names found, use the first user-defined function
    func_name = list(user_symbols.keys())[0]
    return user_symbols[func_name], func_name, ""


def validate_constraint_compatibility(
    constraint_func: Callable[..., Any],
    func_name: str,
    test_kindyn_model: Any,
    test_config: Any,
) -> tuple[bool, str]:
    """
    Validate that the constraint function is compatible with MPC interface.

    Args:
        constraint_func: The constraint function to validate
        func_name: Name of the function
        test_kindyn_model: Test kinodynamic model
        test_config: Test config object

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Create test inputs as expected by MPC
        test_x = cs.MX.sym("test_x", 24)  # State vector
        test_u = cs.MX.sym("test_u", 24)  # Input vector
        test_contact = cs.MX.sym("test_contact", 4)  # Contact sequence
        test_k = 10  # Sample timestep
        test_horizon = 75  # Sample horizon

        # Try to call the function with MPC interface (try 7-arg first, then 5-arg)
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
            result = constraint_func(
                test_x, test_u, test_kindyn_model, test_config, test_contact
            )

        # Validate return format (should be tuple of 3 elements)
        if not isinstance(result, tuple) or len(result) != 3:
            return (
                False,
                f"Function '{func_name}' must return tuple of (constraint_expr, lower_bounds, upper_bounds), got {type(result)} with {len(result) if hasattr(result, '__len__') else 'unknown'} elements",
            )

        constraint_expr, lower_bounds, upper_bounds = result

        # Check if expressions are valid CasADi objects
        if constraint_expr is not None:
            try:
                if hasattr(constraint_expr, "size"):
                    constraint_expr.size()
                else:
                    # Try to convert to CasADi if needed
                    cs.MX(constraint_expr)
            except Exception as e:
                return (
                    False,
                    f"Constraint expression from '{func_name}' is not a valid CasADi object: {e}",
                )

        return True, ""

    except TypeError as e:
        if "takes" in str(e) and "positional argument" in str(e):
            return (
                False,
                f"Function '{func_name}' has wrong number of arguments. Expected: (x_k, u_k, kindyn_model, config, contact_k)",
            )
        else:
            return False, f"Function '{func_name}' signature error: {e}"
    except Exception as e:
        return False, f"Function '{func_name}' validation failed: {e}"


def test_constraint_function(
    constraint_func: Callable[..., Any],
    test_x: Any,
    test_u: Any,
    test_kindyn_model: Any,
    test_config: Any,
    test_contact: Any,
) -> tuple[bool, str]:
    """
    Test that the constraint function works with sample inputs.

    Args:
        constraint_func: The constraint function to test
        test_x: Sample state vector (CasADi MX)
        test_u: Sample input vector (CasADi MX)
        test_kindyn_model: Sample kinodynamic model
        test_config: Sample config object
        test_contact: Sample contact sequence (CasADi MX)

    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Get the restricted globals to ensure function execution has access
        restricted_globals = create_restricted_globals()

        # Set the function's globals to our restricted namespace
        if hasattr(constraint_func, "__globals__"):
            # Update the function's globals with our restricted environment
            constraint_func.__globals__.update(restricted_globals)

        # Call the constraint function
        result = constraint_func(
            test_x, test_u, test_kindyn_model, test_config, test_contact
        )

        # Validate return format (should be tuple of 3 elements)
        if not isinstance(result, tuple) or len(result) != 3:
            return (
                False,
                "Constraint function must return tuple of (expr, lower_bounds, upper_bounds)",
            )

        expr, lower, upper = result

        # Basic validation that we got CasADi expressions/arrays
        if expr is None:
            return False, "Constraint expression is None"

        # Check if expressions are CasADi objects or can be converted
        try:
            # Try to get the size of the expression to validate it's a proper CasADi object
            if hasattr(expr, "size"):
                expr.size()  # Just validate that size can be computed
            elif hasattr(expr, "shape"):
                _ = expr.shape  # Just validate that shape can be accessed
            else:
                # Convert to CasADi if needed
                expr = cs.MX(expr)
                expr.size()  # Just validate that size can be computed
        except Exception as e:
            return False, f"Constraint expression is not a valid CasADi object: {e}"

        return True, ""

    except Exception as e:
        error_msg = (
            f"Constraint function test failed: {str(e)}\n{traceback.format_exc()}"
        )
        return False, error_msg
