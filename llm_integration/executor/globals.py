"""Restricted globals and import processing for safe code execution."""

import ast
from typing import Any

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


def create_restricted_globals(
    allowed_imports: dict[str, list[str]] | None = None,
    additional_imports: list[str] | None = None,
) -> dict[str, Any]:
    """
    Create a restricted global namespace for code execution.

    Args:
        allowed_imports: Dictionary of allowed imports (defaults to ALLOWED_IMPORTS)
        additional_imports: Optional list of additional imports to include
    """
    if allowed_imports is None:
        allowed_imports = ALLOWED_IMPORTS

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
            process_dynamic_imports(
                restricted_globals, additional_imports, allowed_imports
            )

    except ImportError as e:
        print(f"Warning: Could not import required modules: {e}")

    return restricted_globals


def process_dynamic_imports(
    globals_dict: dict[str, Any],
    import_requests: list[str],
    allowed_imports: dict[str, list[str]] | None = None,
) -> None:
    """
    Process dynamic import requests from LLM code and add them to globals.

    Args:
        globals_dict: Dictionary to add imports to
        import_requests: List of import statements to process
        allowed_imports: Dictionary of allowed imports (defaults to ALLOWED_IMPORTS)
    """
    if allowed_imports is None:
        allowed_imports = ALLOWED_IMPORTS

    for import_request in import_requests:
        try:
            # Parse the import statement
            tree = ast.parse(import_request)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in allowed_imports:
                            # Import the module
                            module = __import__(alias.name)
                            alias_name = alias.asname if alias.asname else alias.name
                            globals_dict[alias_name] = module

                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module in allowed_imports:
                        module = __import__(
                            node.module,
                            fromlist=[alias.name for alias in node.names],
                        )
                        for alias in node.names:
                            if alias.name in allowed_imports[node.module]:
                                attr = getattr(module, alias.name)
                                alias_name = (
                                    alias.asname if alias.asname else alias.name
                                )
                                globals_dict[alias_name] = attr

        except Exception as e:
            print(f"Warning: Could not process import '{import_request}': {e}")


def extract_imports_from_code(code: str) -> list[str]:
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
