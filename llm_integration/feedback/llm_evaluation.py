"""LLM clients and shared helpers for evaluation calls.

Provides:
- call_llm(): Claude-based text evaluation for scoring/feedback/summary
- format_violations(), format_error_info(): Shared formatting helpers
- extract_json_from_response(): JSON extraction from LLM responses
"""

from __future__ import annotations

import os
from typing import Any

import anthropic
from dotenv import load_dotenv

from .format_metrics import format_trajectory_metrics_text  # noqa: F401 (re-export)

# ---------------------------------------------------------------------------
# Claude client (scoring, feedback, summary)
# ---------------------------------------------------------------------------

_claude_client: anthropic.Anthropic | None = None
_claude_model: str = ""


def _get_claude_client() -> anthropic.Anthropic:
    """Get or create the global Anthropic client."""
    global _claude_client, _claude_model
    if _claude_client is None:
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        _claude_model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
        if not api_key or api_key == "your_api_key_here":
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
        _claude_client = anthropic.Anthropic(api_key=api_key)
    return _claude_client


def call_llm(system_prompt: str, user_message: str) -> str:
    """Make a text-only Claude call."""
    client = _get_claude_client()

    collected: list[str] = []
    with client.messages.stream(
        model=_claude_model,
        max_tokens=40000,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            collected.append(text)

    return "".join(collected)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def extract_json_from_response(response: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = response.strip()

    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]

        if text.endswith("```"):
            text = text[:-3]
        elif "```" in text:
            text = text[: text.rfind("```")]

    return text.strip()


def _compress_timesteps(ks: list[int]) -> str:
    """Turn a sorted list of timestep indices into compressed ranges.

    Example: [15, 16, 17, 48, 49, 50] -> "k=15-17, 48-50"
    """
    if not ks:
        return ""
    sorted_ks = sorted(ks)
    ranges: list[str] = []
    start = prev = sorted_ks[0]
    for k in sorted_ks[1:]:
        if k == prev + 1:
            prev = k
        else:
            ranges.append(f"{start}-{prev}" if prev > start else str(start))
            start = prev = k
    ranges.append(f"{start}-{prev}" if prev > start else str(start))
    return "k=" + ", ".join(ranges)


def format_violations(constraint_violations: dict[str, Any] | None) -> str:
    """Format constraint violations dict into condensed per-element summaries for LLM prompts."""
    if not constraint_violations:
        return "None"

    lines: list[str] = []

    # System constraint info (non-LLM keys)
    for key, val in constraint_violations.items():
        if key in ("llm_constraints", "by_constraint", "constraint_meta", "summary",
                    "llm_summary"):
            continue
        if isinstance(val, list):
            for item in val:
                lines.append(f"  {key}: {item}")
        else:
            lines.append(f"  {key}: {val}")

    # LLM constraint per-element summaries
    by_constraint = constraint_violations.get("by_constraint", {})
    meta = constraint_violations.get("constraint_meta", {})

    if by_constraint:
        if lines:
            lines.append("")
        lines.append("LLM Constraints:")

    for i, violations_list in sorted(by_constraint.items(), key=lambda x: int(x[0])):
        info = meta.get(i) or meta.get(int(i)) or meta.get(str(i))
        if info:
            name = info["name"]
            n_out = info["n_outputs"]
            lines.append(f"  {name} [{n_out} outputs]:")
        else:
            lines.append(f"  constraint_{i}:")
            n_out = 0

        # Group violations by element j
        by_element: dict[int, list[dict]] = {}
        for v in violations_list:
            j = v.get("element", 0)
            by_element.setdefault(j, []).append(v)

        # Determine all element indices (use n_out if known, else only violated ones)
        all_elements = set(range(n_out)) if n_out else set(by_element.keys())
        all_elements |= set(by_element.keys())

        for j in sorted(all_elements):
            elem_violations = by_element.get(j, [])
            if not elem_violations:
                lines.append(f"    [{j}]: 0 violations")
                continue

            count = len(elem_violations)
            # Find worst violation (max |value - bound|)
            worst = max(
                elem_violations,
                key=lambda v: abs(v["value"] - v.get("lower", v.get("upper", v["value"]))),
            )
            worst_k = worst["k"]
            worst_val = worst["value"]
            if worst.get("type") == "below_lower":
                deviation = abs(worst_val - worst["lower"])
                bound_str = f"lower={worst['lower']:.3f}"
            else:
                deviation = abs(worst_val - worst["upper"])
                bound_str = f"upper={worst['upper']:.3f}"

            timesteps = _compress_timesteps([v["k"] for v in elem_violations])
            lines.append(
                f"    [{j}]: {count} violations, max deviation {deviation:.3f} "
                f"at k={worst_k} (value={worst_val:.3f}, {bound_str}), {timesteps}"
            )

    # LLM summary (aggregate)
    llm_summary = constraint_violations.get("llm_summary") or constraint_violations.get("summary")
    if llm_summary and isinstance(llm_summary, list):
        if lines:
            lines.append("")
        for item in llm_summary:
            lines.append(f"  {item}")

    return "\n".join(lines) if lines else "None"


def format_error_info(error_info: dict[str, Any] | None) -> str:
    """Format error information for LLM prompts."""
    if not error_info:
        return ""
    lines = []
    if error_info.get("error_message"):
        lines.append(f"Error: {error_info['error_message']}")
    if error_info.get("solver_iterations"):
        lines.append(
            f"Solver stopped after {error_info['solver_iterations']} iterations"
        )
    if error_info.get("constraint_violations"):
        lines.append(f"Violations: {str(error_info['constraint_violations'])}")
    return "\n".join(lines) if lines else ""
