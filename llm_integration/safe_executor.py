"""Safe execution environment for LLM-generated constraint code."""

import ast
import inspect
import traceback
from collections.abc import Callable
from typing import Any

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
        self.constraint_functions: dict[str, Callable[..., Any]] = {}
        self.last_executed_code: str | None = None
        self.detected_entry_point: str | None = None

    def validate_code_safety(self, code: str) -> tuple[bool, str]:
        """
        Validate that the code only contains allowed operations.

        Args:
            code: Python code string to validate

        Returns:
            Tuple of (is_safe, error_message)
        """
        # First check basic syntax
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Check for dangerous operations
        dangerous_functions = [
            "exec",
            "eval",
            "open",
            "__import__",
            "compile",
            "globals",
            "locals",
        ]
        dangerous_attributes = ["__class__", "__mro__", "__bases__", "__globals__"]

        for node in ast.walk(tree):
            # Block dangerous built-ins
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in dangerous_functions:
                        return False, f"Dangerous function call: {node.func.id}"

            # Block dangerous attribute access
            if isinstance(node, ast.Attribute):
                if node.attr in dangerous_attributes:
                    return False, f"Dangerous attribute access: {node.attr}"

            # Allow imports but validate they're safe
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.ALLOWED_IMPORTS:
                        return (
                            False,
                            f"Unauthorized import: {alias.name}. Allowed: {list(self.ALLOWED_IMPORTS.keys())}",
                        )

            if isinstance(node, ast.ImportFrom):
                if node.module and node.module not in self.ALLOWED_IMPORTS:
                    return (
                        False,
                        f"Unauthorized import from: {node.module}. Allowed: {list(self.ALLOWED_IMPORTS.keys())}",
                    )

                # Check specific imports from allowed modules
                if node.names and node.module:
                    for alias in node.names:
                        if alias.name not in self.ALLOWED_IMPORTS[node.module]:
                            return (
                                False,
                                f"Unauthorized import: {alias.name} from {node.module}. Allowed from {node.module}: {self.ALLOWED_IMPORTS[node.module]}",
                            )

        # Check for proper function definition
        function_defs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if len(function_defs) == 0:
            return (
                False,
                "No function definition found. Must define a constraint function.",
            )

        # Validate function signatures (prefer 7 arguments, accept 5 for backward compatibility)
        for func_def in function_defs:
            num_args = len(func_def.args.args)
            if num_args not in (5, 7):
                return (
                    False,
                    f"Function '{func_def.name}' must have 7 arguments: (x_k, u_k, kindyn_model, config, contact_k, k, horizon). Got {num_args} arguments.",
                )

        return True, ""

    def execute_mpc_configuration_code(
        self, code: str, llm_mpc: Any
    ) -> tuple[bool, str]:
        """
        Safely execute LLM-generated MPC configuration code with dynamic imports.

        Args:
            code: Python code that configures the LLM MPC instance
            llm_mpc: LLMTaskMPC instance to be configured

        Returns:
            Tuple of (success, error_message)
        """
        # Validate code safety first (now allows imports)
        is_safe, safety_error = self.validate_code_safety(code)
        if not is_safe:
            return False, f"Code validation failed: {safety_error}"

        try:
            # The LLM MPC will handle dynamic import processing internally
            success, error = llm_mpc.configure_from_llm(code)

            if not success:
                return False, f"MPC configuration failed: {error}"

            # Validate that MPC was properly configured
            config_summary = llm_mpc.get_configuration_summary()

            if not config_summary["is_configured"]:
                return False, "MPC configuration incomplete"

            if config_summary["num_constraints"] == 0:
                return False, "No constraints added to MPC"

            if config_summary["horizon"] == 0:
                return False, "No valid contact sequence specified"

            return True, ""

        except Exception as e:
            return False, f"MPC configuration execution failed: {str(e)}"

    def execute_constraint_code(
        self, code: str, preferred_function_name: str | None = None
    ) -> tuple[bool, Callable[..., Any] | None, str, str]:
        """
        Safely execute constraint code and extract the constraint function.
        (Legacy method for compatibility)

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
            return False, None, f"Code validation failed: {safety_error}", ""

        # Create a restricted execution environment
        restricted_globals = self._create_restricted_globals()
        restricted_locals: dict[str, Any] = {}

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

            # Validate function signature (accept both 5 and 7 parameters)
            try:
                sig = inspect.signature(constraint_func)
                if len(sig.parameters) not in (5, 7):
                    return (
                        False,
                        None,
                        f"Function '{func_name}' must have 5 or 7 parameters, got {len(sig.parameters)}",
                        func_name,
                    )
            except Exception as e:
                return (
                    False,
                    None,
                    f"Could not inspect function signature: {e}",
                    func_name,
                )

            # Test basic call with dummy arguments to catch early errors
            try:
                test_x = cs.MX.sym("test_x", 24)
                test_u = cs.MX.sym("test_u", 24)
                test_contact = cs.MX.sym("test_contact", 4)
                test_k = 10  # Sample timestep
                test_horizon = 75  # Sample horizon

                # Mock objects for testing
                class MockKindyn:
                    def forward_kinematics_FL_fun(
                        self, H: cs.MX, joint_pos: cs.MX
                    ) -> cs.MX:
                        return cs.MX.eye(4)

                class MockConfig:
                    def __init__(self) -> None:
                        self.mpc_config = type("obj", (object,), {"mpc_dt": 0.02})()
                        self.experiment = type("obj", (object,), {"mu_ground": 0.8})()
                        self.robot_data = type("obj", (object,), {"mass": 12.0})()

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
                        None,
                        f"Function '{func_name}' must return tuple of 3 elements, got {type(result)} with {len(result) if hasattr(result, '__len__') else 'unknown'} elements",
                        func_name,
                    )

                expr, lower, upper = result

                # Validate that returned values are CasADi expressions
                if expr is not None:
                    if not hasattr(expr, "size"):
                        return (
                            False,
                            None,
                            f"First return value must be a CasADi expression with .size() method, got {type(expr)}",
                            func_name,
                        )

                    # Check bounds have correct dimensions
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
                                    None,
                                    f"Lower bound dimension ({lower_length}) must match constraint dimension ({expr_length})",
                                    func_name,
                                )

                        # Check upper bound
                        if hasattr(upper, "size"):
                            upper_size = upper.size()
                            upper_length = upper_size[0] if len(upper_size) > 1 else 1
                            if upper_length != expr_length:
                                return (
                                    False,
                                    None,
                                    f"Upper bound dimension ({upper_length}) must match constraint dimension ({expr_length})",
                                    func_name,
                                )

                    except Exception as dim_error:
                        return (
                            False,
                            None,
                            f"Could not verify dimensions: {dim_error}",
                            func_name,
                        )

            except Exception as test_error:
                return (
                    False,
                    None,
                    f"Function test call failed: {test_error}",
                    func_name,
                )

            # Ensure the function has access to the restricted globals
            if hasattr(constraint_func, "__globals__"):
                # Update the function's globals to include our restricted environment
                constraint_func.__globals__.update(restricted_globals)

            # Store the function and detected entry point
            self.constraint_functions[func_name] = constraint_func
            self.detected_entry_point = func_name

            return True, constraint_func, "", func_name

        except SyntaxError as e:
            return False, None, f"Syntax error in code: {e}", ""
        except NameError as e:
            return False, None, f"Undefined variable: {e}", ""
        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            # Add more specific error details if available
            if "vertcat" in str(e):
                error_msg += "\nTip: Use vertcat() to stack multiple constraints"
            elif "MX" in str(e):
                error_msg += "\nTip: Ensure all expressions are CasADi MX objects"
            elif "dimension" in str(e).lower():
                error_msg += "\nTip: Check that bounds match constraint dimensions"
            return False, None, error_msg, ""

    def _create_restricted_globals(
        self, additional_imports: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Create a restricted global namespace for code execution.

        Args:
            additional_imports: Optional list of additional imports to include
        """
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
                "__import__": __import__,  # Needed for dynamic imports
            }
        }

        # Import allowed modules
        try:
            import math

            # from typing import Any, List, Optional, Tuple, Union
            import casadi as cs
            import numpy as np
            from liecasadi import SO3

            # Add main module references
            restricted_globals["cs"] = cs
            restricted_globals["np"] = np
            restricted_globals["math"] = math  # type: ignore[assignment]
            restricted_globals["SO3"] = SO3

            # Make liecasadi module available too
            import liecasadi

            restricted_globals["liecasadi"] = liecasadi

            # Add ALL commonly used CasADi functions directly
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
            restricted_globals["tan"] = cs.tan
            restricted_globals["exp"] = cs.exp
            restricted_globals["log"] = cs.log
            restricted_globals["atan2"] = cs.atan2
            restricted_globals["asin"] = cs.asin
            restricted_globals["acos"] = cs.acos
            restricted_globals["tanh"] = cs.tanh
            restricted_globals["sinh"] = cs.sinh
            restricted_globals["cosh"] = cs.cosh
            restricted_globals["norm_2"] = cs.norm_2
            restricted_globals["transpose"] = cs.transpose
            restricted_globals["inv"] = cs.inv
            restricted_globals["dot"] = cs.dot
            restricted_globals["cross"] = cs.cross
            restricted_globals["repmat"] = cs.repmat
            restricted_globals["reshape"] = cs.reshape
            restricted_globals["if_else"] = cs.if_else
            restricted_globals["logic_and"] = cs.logic_and
            restricted_globals["logic_or"] = cs.logic_or

            # Constants
            restricted_globals["inf"] = cs.inf
            restricted_globals["pi"] = np.pi

            # CasADi types
            restricted_globals["MX"] = cs.MX
            restricted_globals["SX"] = cs.SX
            restricted_globals["DM"] = cs.DM

            # Matrix creation functions
            restricted_globals["eye"] = cs.MX.eye
            restricted_globals["zeros"] = cs.MX.zeros
            restricted_globals["ones"] = cs.MX.ones

            # Process additional imports from LLM code
            if additional_imports:
                self._process_dynamic_imports(restricted_globals, additional_imports)

        except ImportError as e:
            print(f"Warning: Could not import required modules: {e}")

        return restricted_globals

    def _process_dynamic_imports(
        self, globals_dict: dict[str, Any], import_requests: list[str]
    ) -> None:
        """
        Process dynamic import requests from LLM code and add them to globals.

        Args:
            globals_dict: Dictionary to add imports to
            import_requests: List of import statements to process
        """
        for import_request in import_requests:
            try:
                # Parse the import statement
                tree = ast.parse(import_request)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name in self.ALLOWED_IMPORTS:
                                # Import the module
                                module = __import__(alias.name)
                                alias_name = (
                                    alias.asname if alias.asname else alias.name
                                )
                                globals_dict[alias_name] = module

                    elif isinstance(node, ast.ImportFrom):
                        if node.module and node.module in self.ALLOWED_IMPORTS:
                            module = __import__(
                                node.module,
                                fromlist=[alias.name for alias in node.names],
                            )
                            for alias in node.names:
                                if alias.name in self.ALLOWED_IMPORTS[node.module]:
                                    attr = getattr(module, alias.name)
                                    alias_name = (
                                        alias.asname if alias.asname else alias.name
                                    )
                                    globals_dict[alias_name] = attr

            except Exception as e:
                print(f"Warning: Could not process import '{import_request}': {e}")

    def extract_imports_from_code(self, code: str) -> list[str]:
        """
        Extract import statements from LLM code for dynamic processing.

        Args:
            code: Python code to analyze

        Returns:
            List of import statements found in the code
        """
        imports = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Reconstruct the import statement
                    import_stmt = ast.get_source_segment(code, node)
                    if import_stmt:
                        imports.append(import_stmt)
        except Exception:
            pass  # If parsing fails, return empty list

        return imports

    def _find_constraint_entry_point(
        self,
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

    def validate_initial_state_feasibility(
        self,
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
        import numpy as np

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

    def test_constraint_function(
        self,
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
