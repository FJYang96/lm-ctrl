"""Safe execution environment for LLM-generated constraint code."""

import ast
import traceback
from typing import Any, Callable, Dict, Optional, Tuple

import casadi as cs


class SafeConstraintExecutor:
    """
    Safely execute LLM-generated constraint code with restricted imports and execution.
    Supports flexible function signatures and automatic constraint entry point detection.
    """

    # Allowed imports for constraint generation
    ALLOWED_IMPORTS = {
        "casadi": [
            "cs",
            "MX",
            "SX",
            "vertcat",
            "mtimes",
            "fabs",
            "sin",
            "cos",
            "sqrt",
            "exp",
            "sum1",
            "fmax",
            "fmin",
            "inf",
            "horzcat",
            "DM",
            "transpose",
            "norm_2",
            "atan2",
            "tan",
            "asin",
            "acos",
            "tanh",
            "sinh",
            "cosh",
        ],
        "numpy": [
            "np",
            "array",
            "zeros",
            "ones",
            "eye",
            "pi",
            "sin",
            "cos",
            "sqrt",
            "inf",
            "concatenate",
            "stack",
            "linalg",
            "maximum",
            "minimum",
        ],
        "math": [
            "pi",
            "sin",
            "cos",
            "sqrt",
            "exp",
            "log",
            "fabs",
            "atan2",
            "tan",
            "asin",
            "acos",
            "tanh",
            "sinh",
            "cosh",
            "radians",
            "degrees",
        ],
        "typing": ["Any", "Tuple", "List", "Union", "Optional"],
        "liecasadi": ["SO3", "SE3"],  # Allow LieCasadi for rotation matrices
    }

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

    def __init__(self) -> None:
        """Initialize the safe executor."""
        self.constraint_functions: Dict[str, Callable[..., Any]] = {}
        self.last_executed_code: Optional[str] = None
        self.detected_entry_point: Optional[str] = None

    def validate_code_safety(self, code: str) -> tuple[bool, str]:
        """
        Validate that the code only contains allowed operations.

        Args:
            code: Python code string to validate

        Returns:
            Tuple of (is_safe, error_message)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Check for dangerous operations
        for node in ast.walk(tree):
            # Block dangerous built-ins
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["exec", "eval", "open", "__import__"]:
                        return False, f"Dangerous function call: {node.func.id}"

            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.ALLOWED_IMPORTS:
                        return False, f"Unauthorized import: {alias.name}"

            if isinstance(node, ast.ImportFrom):
                if node.module and node.module not in self.ALLOWED_IMPORTS:
                    return False, f"Unauthorized import from: {node.module}"

                # Check specific imports from allowed modules
                if node.names and node.module:
                    for alias in node.names:
                        if alias.name not in self.ALLOWED_IMPORTS[node.module]:
                            return (
                                False,
                                f"Unauthorized import: {alias.name} from {node.module}",
                            )

        return True, ""

    def execute_constraint_code(
        self, code: str, preferred_function_name: Optional[str] = None
    ) -> Tuple[bool, Optional[Callable[..., Any]], str, str]:
        """
        Safely execute constraint code and extract the constraint function.

        Args:
            code: Python code defining constraints
            preferred_function_name: Preferred function name (optional)

        Returns:
            Tuple of (success, constraint_function, error_message, detected_function_name)
        """
        # Store the executed code
        self.last_executed_code = code

        # Validate code safety first
        is_safe, safety_error = self.validate_code_safety(code)
        if not is_safe:
            return False, None, f"Code safety validation failed: {safety_error}", ""

        # Create a restricted execution environment
        restricted_globals = self._create_restricted_globals()
        restricted_locals: Dict[str, Any] = {}

        try:
            # Execute the code in restricted environment
            exec(code, restricted_globals, restricted_locals)

            # Find the constraint entry point
            constraint_func, func_name, error = self._find_constraint_entry_point(
                restricted_locals, restricted_globals, preferred_function_name
            )

            if constraint_func is None:
                return False, None, error, ""

            # Validate that it's actually a function
            if not callable(constraint_func):
                return False, None, f"'{func_name}' is not callable", func_name

            # Ensure the function has access to the restricted globals
            if hasattr(constraint_func, "__globals__"):
                # Update the function's globals to include our restricted environment
                constraint_func.__globals__.update(restricted_globals)

            # Store the function and detected entry point
            self.constraint_functions[func_name] = constraint_func
            self.detected_entry_point = func_name

            return True, constraint_func, "", func_name

        except Exception as e:
            error_msg = f"Execution error: {str(e)}\n{traceback.format_exc()}"
            return False, None, error_msg, ""

    def _create_restricted_globals(self) -> Dict[str, Any]:
        """Create a restricted global namespace for code execution."""
        restricted_globals = {
            "__builtins__": {
                # Basic Python built-ins needed for constraint functions
                "abs": abs,
                "max": max,
                "min": min,
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "print": print,  # For debugging
                "float": float,
                "int": int,
                "str": str,
                "list": list,
                "tuple": tuple,
                "dict": dict,
                "hasattr": hasattr,
                "getattr": getattr,
                "isinstance": isinstance,
                "type": type,
            }
        }

        # Import allowed modules
        try:
            import math
            from typing import Any, List, Optional, Tuple, Union

            import casadi as cs
            import numpy as np
            from liecasadi import SO3

            # Add main module references
            restricted_globals["cs"] = cs
            restricted_globals["np"] = np
            restricted_globals["math"] = math  # type: ignore
            restricted_globals["SO3"] = SO3
            restricted_globals["Any"] = Any  # type: ignore
            restricted_globals["Tuple"] = Tuple  # type: ignore
            restricted_globals["List"] = List  # type: ignore
            restricted_globals["Union"] = Union  # type: ignore
            restricted_globals["Optional"] = Optional  # type: ignore

            # Make liecasadi module available too for imports within functions
            import liecasadi

            restricted_globals["liecasadi"] = liecasadi

            # Add commonly used CasADi functions and constants directly for convenience
            restricted_globals["vertcat"] = cs.vertcat
            restricted_globals["horzcat"] = cs.horzcat
            restricted_globals["mtimes"] = cs.mtimes
            restricted_globals["fabs"] = cs.fabs
            restricted_globals["fmax"] = cs.fmax
            restricted_globals["fmin"] = cs.fmin
            restricted_globals["sum1"] = cs.sum1
            restricted_globals["sqrt"] = cs.sqrt
            restricted_globals["sin"] = cs.sin
            restricted_globals["cos"] = cs.cos
            restricted_globals["exp"] = cs.exp
            restricted_globals["atan2"] = cs.atan2
            restricted_globals["tan"] = cs.tan
            restricted_globals["inf"] = cs.inf
            restricted_globals["MX"] = cs.MX
            restricted_globals["SX"] = cs.SX
            restricted_globals["DM"] = cs.DM

            # Add eye function for matrix creation
            restricted_globals["eye"] = cs.MX.eye

            # Add NumPy constants
            restricted_globals["pi"] = np.pi

        except ImportError as e:
            print(f"Warning: Could not import required modules: {e}")

        return restricted_globals

    def _find_constraint_entry_point(
        self,
        locals_dict: Dict[str, Any],
        globals_dict: Dict[str, Any],
        preferred_name: Optional[str] = None,
    ) -> Tuple[Optional[Callable[..., Any]], str, str]:
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
        user_symbols = {
            k: v
            for k, v in all_symbols.items()
            if callable(v)
            and not k.startswith("_")
            and k not in self._create_restricted_globals()
        }

        if not user_symbols:
            return None, "", "No callable functions found in generated code"

        # If preferred name is specified and exists, use it
        if preferred_name and preferred_name in user_symbols:
            return user_symbols[preferred_name], preferred_name, ""

        # Look for common constraint function names
        for name in self.CONSTRAINT_FUNCTION_NAMES:
            if name in user_symbols:
                return user_symbols[name], name, ""

        # If no common names found, use the first user-defined function
        func_name = list(user_symbols.keys())[0]
        return user_symbols[func_name], func_name, ""

    def validate_constraint_compatibility(
        self,
        constraint_func: Callable[..., Any],
        func_name: str,
        test_kindyn_model: Any,
        test_config: Any,
    ) -> Tuple[bool, str]:
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

            # Try to call the function with MPC interface
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
        self,
        constraint_func: Callable[..., Any],
        test_x: Any,
        test_u: Any,
        test_kindyn_model: Any,
        test_config: Any,
        test_contact: Any,
    ) -> Tuple[bool, str]:
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
            restricted_globals = self._create_restricted_globals()

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
