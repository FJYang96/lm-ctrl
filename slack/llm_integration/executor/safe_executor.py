"""Safe execution environment for LLM-generated constraint code."""

import ast
import inspect
from collections.abc import Callable
from typing import Any

from .constraint_testing import test_constraint_function_execution
from .globals import (
    ALLOWED_IMPORTS,
    create_restricted_globals,
    extract_imports_from_code,
    process_dynamic_imports,
)
from .validation import (
    CONSTRAINT_FUNCTION_NAMES,
    find_constraint_entry_point,
    test_constraint_function,
    validate_constraint_compatibility,
    validate_initial_state_feasibility,
)


class SafeConstraintExecutor:
    """
    Safely execute LLM-generated constraint code with restricted imports and execution.
    Supports flexible function signatures and automatic constraint entry point detection.
    """

    # Allowed imports for constraint generation (re-exported for compatibility)
    ALLOWED_IMPORTS = ALLOWED_IMPORTS

    # Common constraint function names to look for (re-exported for compatibility)
    CONSTRAINT_FUNCTION_NAMES = CONSTRAINT_FUNCTION_NAMES

    # Assign imported functions directly as methods (no wrapper needed)
    extract_imports_from_code = staticmethod(extract_imports_from_code)
    _find_constraint_entry_point = staticmethod(find_constraint_entry_point)
    validate_constraint_compatibility = staticmethod(validate_constraint_compatibility)
    validate_initial_state_feasibility = staticmethod(
        validate_initial_state_feasibility
    )
    test_constraint_function = staticmethod(test_constraint_function)

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
            success, test_error = test_constraint_function_execution(
                constraint_func, func_name
            )
            if not success:
                return False, None, test_error, func_name

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
        return create_restricted_globals(self.ALLOWED_IMPORTS, additional_imports)

    def _process_dynamic_imports(
        self, globals_dict: dict[str, Any], import_requests: list[str]
    ) -> None:
        """
        Process dynamic import requests from LLM code and add them to globals.

        Args:
            globals_dict: Dictionary to add imports to
            import_requests: List of import statements to process
        """
        process_dynamic_imports(globals_dict, import_requests, self.ALLOWED_IMPORTS)
