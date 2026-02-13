"""Safe execution environment for LLM-generated constraint code."""

import ast
from collections.abc import Callable
from typing import Any

from .globals import (
    ALLOWED_IMPORTS,
    create_restricted_globals,
    extract_imports_from_code,
)
from .validation import (
    find_constraint_entry_point,
)


class SafeConstraintExecutor:
    """
    Safely execute LLM-generated constraint code with restricted imports and execution.
    Supports flexible function signatures and automatic constraint entry point detection.
    """

    # Allowed imports for constraint generation (re-exported for compatibility)
    ALLOWED_IMPORTS = ALLOWED_IMPORTS

    # Assign imported functions directly as methods (no wrapper needed)
    extract_imports_from_code = staticmethod(extract_imports_from_code)
    _find_constraint_entry_point = staticmethod(find_constraint_entry_point)

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

        # Validate function signatures
        for func_def in function_defs:
            num_args = len(func_def.args.args)
            # Reference trajectory functions have a different signature (2-8 args)
            is_ref_func = "reference" in func_def.name or func_def.name.startswith(
                "generate_"
            )
            if is_ref_func:
                if not (2 <= num_args <= 8):
                    return (
                        False,
                        f"Reference function '{func_def.name}' must have 2-8 arguments. Got {num_args}.",
                    )
            else:
                # Constraint functions: prefer 7 arguments, accept 5 for backward compat
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

    def _create_restricted_globals(
        self, additional_imports: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Create a restricted global namespace for code execution.

        Args:
            additional_imports: Optional list of additional imports to include
        """
        return create_restricted_globals(self.ALLOWED_IMPORTS, additional_imports)
