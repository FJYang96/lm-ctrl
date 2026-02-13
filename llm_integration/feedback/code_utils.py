"""Utility functions for processing constraint code in feedback."""

import ast


def strip_ref_trajectory_code(full_code: str) -> str:
    """Remove the reference trajectory function from code, returning the rest.

    This strips out functions matching the reference trajectory naming convention
    (containing 'reference' or starting with 'generate_') to save tokens in feedback.
    """
    try:
        tree = ast.parse(full_code)
    except SyntaxError:
        return full_code

    lines = full_code.splitlines()
    # Collect line ranges to remove (0-indexed)
    ranges_to_remove: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and (
            "reference" in node.name or node.name.startswith("generate_")
        ):
            start = node.lineno - 1  # 0-indexed
            end = node.end_lineno if node.end_lineno is not None else node.lineno
            ranges_to_remove.append((start, end))

    if not ranges_to_remove:
        return full_code

    # Build result excluding removed ranges
    remove_set: set[int] = set()
    for start, end in ranges_to_remove:
        remove_set.update(range(start, end))

    kept_lines = [line for i, line in enumerate(lines) if i not in remove_set]
    return "\n".join(kept_lines)
