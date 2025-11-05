"""
Safe code execution parser for LLM-generated constraint functions
"""

import ast
import inspect
import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import casadi as cs
import numpy as np


class SafeCodeParser:
    """
    Safely parse and execute LLM-generated constraint functions with security restrictions
    """

    # Allowed imports and built-in functions
    ALLOWED_IMPORTS = {
        "casadi": "cs",
        "numpy": "np",
        "math": "math",
    }

    ALLOWED_BUILTINS = {
        "abs",
        "max",
        "min",
        "sum",
        "len",
        "range",
        "enumerate",
        "zip",
        "float",
        "int",
        "bool",
        "str",
        "list",
        "tuple",
        "dict",
        "True",
        "False",
        "None",
    }

    # Dangerous functions/modules to block
    BLOCKED_FUNCTIONS = {
        "exec",
        "eval",
        "compile",
        "open",
        "file",
        "input",
        "raw_input",
        "__import__",
        "reload",
        "delattr",
        "setattr",
        "getattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "help",
    }

    BLOCKED_MODULES = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "urllib",
        "requests",
        "pickle",
        "cPickle",
        "marshal",
        "shelve",
        "dbm",
    }

    def __init__(self) -> None:
        """Initialize the safe code parser"""
        self.globals_dict = self._create_safe_globals()

    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create a restricted global namespace for code execution"""
        safe_globals = {"__builtins__": {}}

        # Add allowed built-ins
        for name in self.ALLOWED_BUILTINS:
            if hasattr(__builtins__, name):
                safe_globals["__builtins__"][name] = getattr(__builtins__, name)

        # Add allowed imports
        try:
            import math

            import casadi as cs
            import numpy as np

            safe_globals["cs"] = cs
            safe_globals["np"] = np
            safe_globals["math"] = math

            # Add state/input indexing constants
            safe_globals["MP_X_BASE_POS"] = slice(0, 3)
            safe_globals["MP_X_BASE_VEL"] = slice(3, 6)
            safe_globals["MP_X_BASE_EUL"] = slice(6, 9)
            safe_globals["MP_X_BASE_ANG"] = slice(9, 12)
            safe_globals["MP_X_Q"] = slice(12, 24)
            safe_globals["MP_U_QD"] = slice(0, 12)
            safe_globals["MP_U_CONTACT_F"] = slice(12, 24)

        except ImportError as e:
            warnings.warn(f"Could not import required modules: {e}", stacklevel=2)

        return safe_globals

    def _validate_ast(self, node: ast.AST) -> bool:
        """
        Validate AST for dangerous operations

        Args:
            node: AST node to validate

        Returns:
            True if safe, False if dangerous
        """
        for child in ast.walk(node):
            # Check for dangerous imports
            if isinstance(child, ast.Import):
                for alias in child.names:
                    if alias.name in self.BLOCKED_MODULES:
                        raise ValueError(f"Blocked import: {alias.name}")

            elif isinstance(child, ast.ImportFrom):
                if child.module in self.BLOCKED_MODULES:
                    raise ValueError(f"Blocked import from: {child.module}")

            # Check for dangerous function calls
            elif isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    if child.func.id in self.BLOCKED_FUNCTIONS:
                        raise ValueError(f"Blocked function call: {child.func.id}")

            # Check for attribute access to dangerous modules
            elif isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Name):
                    if child.value.id in self.BLOCKED_MODULES:
                        raise ValueError(f"Blocked module access: {child.value.id}")

        return True

    def parse_constraint_function(self, code: str) -> Optional[Callable]:
        """
        Parse and validate LLM-generated constraint function code

        Args:
            code: Python function code as string

        Returns:
            Compiled function if safe, None if invalid/unsafe

        Raises:
            ValueError: If code contains dangerous operations
            SyntaxError: If code has syntax errors
        """
        try:
            # Parse the code into AST
            tree = ast.parse(code)

            # Validate for security issues
            self._validate_ast(tree)

            # Compile the code
            compiled_code = compile(tree, "<llm_generated>", "exec")

            # Execute in safe namespace
            local_namespace = {}
            exec(compiled_code, self.globals_dict, local_namespace)

            # Find the constraint function
            constraint_func = None
            for name, obj in local_namespace.items():
                if callable(obj) and not name.startswith("_"):
                    constraint_func = obj
                    break

            if constraint_func is None:
                raise ValueError("No callable function found in generated code")

            # Validate function signature
            sig = inspect.signature(constraint_func)
            expected_params = [
                "x_k",
                "u_k",
                "kinodynamic_model",
                "config",
                "contact_k",
                "k",
                "horizon",
            ]

            if list(sig.parameters.keys()) != expected_params:
                raise ValueError(
                    f"Function signature mismatch. Expected: {expected_params}, Got: {list(sig.parameters.keys())}"
                )

            return constraint_func

        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in generated code: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing constraint function: {e}")

    def test_constraint_function(
        self, constraint_func: Callable, kinodynamic_model: Any, config: Any
    ) -> bool:
        """
        Test constraint function with dummy inputs to ensure it works

        Args:
            constraint_func: Parsed constraint function
            kinodynamic_model: Robot model instance
            config: Configuration object

        Returns:
            True if function executes without error
        """
        try:
            # Create dummy symbolic variables
            x_k = cs.SX.sym("x_test", 24)
            u_k = cs.SX.sym("u_test", 24)
            contact_k = cs.SX.sym("contact_test", 4)

            # Test function call
            result = constraint_func(
                x_k, u_k, kinodynamic_model, config, contact_k, k=5, horizon=50
            )

            # Validate return format
            if result is None:
                return True  # None is valid (no constraint at this time step)

            if not isinstance(result, tuple) or len(result) != 3:
                raise ValueError(
                    "Function must return tuple (expr, lower, upper) or None"
                )

            expr, lower, upper = result

            # Check that expression is CasADi symbolic
            if not isinstance(expr, (cs.SX, cs.MX, cs.DM)):
                raise ValueError("Constraint expression must be CasADi symbolic type")

            # Check bounds are numeric or CasADi
            for bound in [lower, upper]:
                if not isinstance(bound, (int, float, cs.SX, cs.MX, cs.DM)):
                    raise ValueError(
                        "Constraint bounds must be numeric or CasADi types"
                    )

            return True

        except Exception as e:
            raise RuntimeError(f"Function test failed: {e}")

    def extract_function_from_response(self, llm_response: str) -> str:
        """
        Extract Python function code from LLM response

        Args:
            llm_response: Full LLM response text

        Returns:
            Extracted function code
        """
        # Look for code blocks
        lines = llm_response.split("\n")
        in_code_block = False
        code_lines = []

        for line in lines:
            if line.strip().startswith("```python"):
                in_code_block = True
                continue
            elif line.strip() == "```" and in_code_block:
                break
            elif in_code_block:
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines)

        # Fallback: look for function definition
        func_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                func_start = i
                break

        if func_start is not None:
            # Extract from function definition to end
            return "\n".join(lines[func_start:])

        # Last resort: return entire response
        return llm_response


class ConstraintValidator:
    """
    Validate generated constraints for physical feasibility and numerical stability
    """

    @staticmethod
    def validate_bounds(lower: float, upper: float) -> bool:
        """Check that constraint bounds are reasonable"""
        if lower > upper:
            return False
        if abs(upper - lower) < 1e-12:  # Too tight
            return False
        if abs(upper) > 1e6 or abs(lower) > 1e6:  # Too large
            return False
        return True

    @staticmethod
    def validate_expression_complexity(expr: cs.SX) -> bool:
        """Check that expression isn't too complex for solver"""
        # Simple heuristic: count operations
        expr_str = str(expr)
        if len(expr_str) > 1000:  # Arbitrary complexity limit
            return False
        return True

    @staticmethod
    def check_physical_feasibility(
        constraint_type: str, bounds: Tuple[float, float]
    ) -> bool:
        """Check if constraints are physically reasonable for quadruped"""
        lower, upper = bounds

        if constraint_type == "height":
            return 0.0 <= upper <= 2.0  # Reasonable height range
        elif constraint_type == "velocity":
            return -10.0 <= lower and upper <= 10.0  # Reasonable velocity range
        elif constraint_type == "angle":
            return -4 * np.pi <= lower and upper <= 4 * np.pi  # Multiple rotations OK
        elif constraint_type == "force":
            return -1000.0 <= lower and upper <= 1000.0  # Reasonable force range

        return True  # Unknown type, assume OK
