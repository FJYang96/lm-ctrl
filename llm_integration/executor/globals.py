"""Restricted globals and import processing for safe code execution."""

from __future__ import annotations

import ast
import logging
from typing import Any

import go2_config

# Use module-level logger
logger = logging.getLogger("llm_integration.executor.globals")


# ── State / input index constants (exposed to LLM code) ──
# Derived from go2_config structural constants
_N_JOINTS = go2_config.N_JOINTS  # 12
_N_LEGS = go2_config.N_LEGS  # 4

STATES_DIM = go2_config.STATES_DIM  # 30
INPUTS_DIM = go2_config.INPUTS_DIM  # 24

# State vector slices
IDX_POS = slice(0, 3)  # COM position [x, y, z]
IDX_VEL = slice(3, 6)  # COM velocity [vx, vy, vz]
IDX_EULER = slice(6, 9)  # Euler angles [roll, pitch, yaw]
IDX_ANG_VEL = slice(9, 12)  # Angular velocity [wx, wy, wz]
IDX_JOINTS = slice(12, 12 + _N_JOINTS)  # Joint angles (12 joints)
IDX_INTEGRALS = slice(12 + _N_JOINTS, STATES_DIM)  # Integral states (padding/zeros)

# Scalar pitch index (row 7 in state vector)
IDX_PITCH = 7

# Input vector slices
IDX_U_JOINT_VEL = slice(0, _N_JOINTS)  # Joint velocities
IDX_U_GRF = slice(_N_JOINTS, INPUTS_DIM)  # Ground reaction forces (4 legs × 3)

# GRF z-component indices (one per foot: FL, FR, RL, RR)
IDX_GRF_Z = [_N_JOINTS + f_idx * 3 + 2 for f_idx in range(_N_LEGS)]


# Allowed imports for constraint generation
ALLOWED_IMPORTS = {
    "casadi": [
        "cs",
        "MX",
        "SX",
        "DM",
        "vertcat",
        "horzcat",
        "mtimes",
        "dot",
        "cross",
        "norm_2",
        "sum1",
        "transpose",
        "inv",
        "reshape",
        "repmat",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan2",
        "sqrt",
        "exp",
        "log",
        "fabs",
        "fmax",
        "fmin",
        "tanh",
        "sinh",
        "cosh",
        "if_else",
        "logic_and",
        "logic_or",
        "inf",
        "eye",
        "zeros",
        "ones",
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


class _RestrictedNumpy:
    """Proxy that blocks dangerous numpy submodules."""

    _BLOCKED = frozenset({"ctypeslib", "testing", "distutils", "f2py", "core"})

    def __init__(self, mod: Any) -> None:
        object.__setattr__(self, "_mod", mod)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"Access to private attribute '{name}' is not allowed")
        if name in self._BLOCKED:
            raise AttributeError(f"Access to np.{name} is not allowed")
        return getattr(object.__getattribute__(self, "_mod"), name)


class _RestrictedCasadi:
    """Proxy that blocks CasADi code-generation and file I/O facilities."""

    _BLOCKED = frozenset(
        {
            "CodeGenerator",
            "cse",
            "external",
            "load_library",
            "import_plugin",
        }
    )

    def __init__(self, mod: Any) -> None:
        object.__setattr__(self, "_mod", mod)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"Access to private attribute '{name}' is not allowed")
        if name in self._BLOCKED:
            raise AttributeError(f"Access to cs.{name} is not allowed")
        return getattr(object.__getattribute__(self, "_mod"), name)


class _RestrictedLiecasadi:
    """Proxy that only exposes SO3 and SE3 from liecasadi."""

    _ALLOWED = frozenset({"SO3", "SE3"})

    def __init__(self, mod: Any) -> None:
        object.__setattr__(self, "_mod", mod)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"Access to private attribute '{name}' is not allowed")
        if name not in self._ALLOWED:
            raise AttributeError(f"Access to liecasadi.{name} is not allowed")
        return getattr(object.__getattribute__(self, "_mod"), name)


def _safe_type(obj: Any) -> type:
    """type() for checking types only — no metaclass construction."""
    return type(obj)


def _safe_getattr(obj: Any, name: str, *default: Any) -> Any:
    """getattr wrapper that blocks access to dunder attributes."""
    if isinstance(name, str) and name.startswith("__") and name.endswith("__"):
        raise AttributeError(f"Access to dunder attribute '{name}' is not allowed")
    return getattr(obj, name, *default)


def _safe_hasattr(obj: Any, name: str) -> bool:
    """hasattr wrapper that blocks probing of dunder attributes."""
    if isinstance(name, str) and name.startswith("__") and name.endswith("__"):
        return False
    return hasattr(obj, name)


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

    restricted_globals: dict[str, Any] = {
        "__builtins__": {
            # Basic Python built-ins needed for constraint functions
            "abs": abs,
            "round": round,
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
            "bool": bool,
            "sum": sum,
            "hasattr": _safe_hasattr,
            "getattr": _safe_getattr,
            "isinstance": isinstance,
            "type": _safe_type,
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
        restricted_globals["cs"] = _RestrictedCasadi(cs)
        restricted_globals["np"] = _RestrictedNumpy(np)
        restricted_globals["math"] = math
        restricted_globals["SO3"] = SO3

        # Make liecasadi module available too
        import liecasadi

        restricted_globals["liecasadi"] = _RestrictedLiecasadi(liecasadi)

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

        # State/input index constants
        restricted_globals["STATES_DIM"] = STATES_DIM
        restricted_globals["INPUTS_DIM"] = INPUTS_DIM
        restricted_globals["IDX_POS"] = IDX_POS
        restricted_globals["IDX_VEL"] = IDX_VEL
        restricted_globals["IDX_EULER"] = IDX_EULER
        restricted_globals["IDX_ANG_VEL"] = IDX_ANG_VEL
        restricted_globals["IDX_JOINTS"] = IDX_JOINTS
        restricted_globals["IDX_INTEGRALS"] = IDX_INTEGRALS
        restricted_globals["IDX_PITCH"] = IDX_PITCH
        restricted_globals["IDX_U_JOINT_VEL"] = IDX_U_JOINT_VEL
        restricted_globals["IDX_U_GRF"] = IDX_U_GRF
        restricted_globals["IDX_GRF_Z"] = IDX_GRF_Z

        # Process additional imports from LLM code
        if additional_imports:
            process_dynamic_imports(
                restricted_globals, additional_imports, allowed_imports
            )

        # CasADi matrix creation functions — set AFTER process_dynamic_imports
        # so numpy zeros/ones/eye don't overwrite the CasADi MX versions.
        restricted_globals["eye"] = cs.MX.eye
        restricted_globals["zeros"] = cs.MX.zeros
        restricted_globals["ones"] = cs.MX.ones

    except ImportError as e:
        logger.warning(f"Could not import required modules: {e}")

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
                            alias_name = alias.asname if alias.asname else alias.name
                            # Always wrap allowed modules in their restricted proxy
                            # to prevent bypassing restrictions via aliasing
                            # (e.g. "import casadi" instead of "import casadi as cs")
                            module = __import__(alias.name)
                            if alias.name == "numpy":
                                globals_dict[alias_name] = _RestrictedNumpy(module)
                            elif alias.name == "casadi":
                                globals_dict[alias_name] = _RestrictedCasadi(module)
                            elif alias.name == "liecasadi":
                                globals_dict[alias_name] = _RestrictedLiecasadi(module)
                            else:
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
            logger.warning(f"Could not process import '{import_request}': {e}")


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
