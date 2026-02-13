"""Validation functions for constraint code execution."""

from collections.abc import Callable
from typing import Any

from .globals import create_restricted_globals

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

__all__ = [
    "CONSTRAINT_FUNCTION_NAMES",
    "find_constraint_entry_point",
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
