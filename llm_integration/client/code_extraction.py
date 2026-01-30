"""Code extraction utilities for LLM responses."""


def extract_code_blocks(response: str) -> list[str]:
    """
    Extract Python code blocks from LLM response.

    Args:
        response: Raw LLM response text

    Returns:
        List of Python code blocks found in the response
    """
    code_blocks = []
    lines = response.split("\n")
    in_code_block = False
    current_block: list[str] = []

    for line in lines:
        if line.strip().startswith("```python") or line.strip() == "```python":
            in_code_block = True
            current_block = []
        elif line.strip() == "```" and in_code_block:
            in_code_block = False
            if current_block:
                code_blocks.append("\n".join(current_block))
            current_block = []
        elif in_code_block:
            current_block.append(line)

    return code_blocks


def extract_raw_code(response: str) -> str:
    """
    Enhanced code extraction from LLM response with better heuristics for MPC configuration.

    This function extracts complete MPC configuration code including:
    - mpc.set_* calls (task_name, duration, time_step, contact_sequence)
    - Variable assignments (phases, contact_seq, etc.)
    - Function definitions (constraint functions)
    - mpc.add_constraint() calls

    Args:
        response: Raw LLM response text

    Returns:
        Extracted Python code
    """
    # First try to extract from code blocks
    code_blocks = extract_code_blocks(response)
    if code_blocks:
        # Return the longest code block (likely the complete MPC configuration)
        longest_block = max(code_blocks, key=len)
        return longest_block

    # If no code blocks, use enhanced raw extraction for MPC configuration
    cleaned = response.strip()
    lines = cleaned.split("\n")

    # MPC configuration patterns to look for
    mpc_patterns = [
        "mpc.set_",
        "mpc.add_",
        "mpc._create_",
        "phases =",
        "phases=",
        "contact_seq",
        "contact_sequence",
    ]

    # Find the start of MPC configuration code
    start_idx = None
    end_idx = len(lines)

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Look for MPC configuration calls first (these come before def)
        if any(pattern in stripped for pattern in mpc_patterns):
            start_idx = i
            break

        # Also look for function definitions as fallback
        if stripped.startswith("def "):
            start_idx = i
            break

    # If we found MPC config, look backwards for any preceding variable assignments
    if start_idx is not None:
        # Check if there are relevant variable assignments just before
        while start_idx > 0:
            prev_line = lines[start_idx - 1].strip()
            # Include comment lines and variable assignments that might be part of config
            if (
                prev_line.startswith("#")
                or "=" in prev_line
                or prev_line.startswith("def ")
                or any(pattern in prev_line for pattern in mpc_patterns)
            ):
                # But skip obvious non-code explanatory text
                if any(
                    prev_line.lower().startswith(skip)
                    for skip in ["here", "the ", "this ", "note:", "i'll", "let me"]
                ):
                    break
                start_idx -= 1
            else:
                break

    # If we found a starting point, extract from there
    if start_idx is not None:
        code_lines = lines[start_idx:end_idx]

        # Clean up common suffixes
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()  # Remove empty lines at end

        # Remove common explanatory text at the end
        while code_lines:
            last_line = code_lines[-1].strip().lower()
            if (
                last_line.startswith("#")
                and ("explanation" in last_line or "note" in last_line)
            ) or any(
                phrase in last_line
                for phrase in [
                    "this function",
                    "this code",
                    "this will",
                    "this should",
                ]
            ):
                code_lines.pop()
            else:
                break

        return "\n".join(code_lines)

    # Fallback: extract any Python-like content that looks like MPC configuration
    python_lines = []
    in_code_section = False

    for line in lines:
        stripped = line.strip()

        # Skip obvious non-code lines
        if any(
            stripped.lower().startswith(skip)
            for skip in [
                "here",
                "the ",
                "this ",
                "note:",
                "i'll",
                "let me",
                "```",
            ]
        ):
            if in_code_section and stripped.startswith("```"):
                break  # End of code block
            continue

        # Check if line looks like Python/MPC configuration code
        is_code_line = (
            stripped.startswith("def ")
            or stripped.startswith("return ")
            or stripped.startswith("if ")
            or stripped.startswith("for ")
            or stripped.startswith("while ")
            or stripped.startswith("mpc.")
            or "=" in stripped
            or stripped.startswith("    ")  # Indented line
            or stripped.startswith("#")  # Comments within code
        )

        if is_code_line:
            in_code_section = True
            python_lines.append(line)
        elif in_code_section and not stripped:
            python_lines.append(line)  # Keep empty lines within code
        elif in_code_section and stripped:
            # Check if this might be a continuation of code
            # (e.g., after a blank line in function body)
            if stripped.startswith("mpc.") or stripped.startswith("def "):
                python_lines.append(line)
            else:
                # Non-Python line after code started - might be end of code
                break

    return "\n".join(python_lines) if python_lines else cleaned
