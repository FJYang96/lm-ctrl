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
    """Get or create the global Anthropic client for eval calls."""
    global _claude_client, _claude_model
    if _claude_client is None:
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        _claude_model = os.getenv("LLM_EVAL_MODEL", "claude-sonnet-4-5-20250929")
        if not api_key or api_key == "your_api_key_here":
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
        _claude_client = anthropic.Anthropic(api_key=api_key)
    return _claude_client


def call_llm(system_prompt: str, user_message: str) -> str:
    """Make a text-only Claude call for scoring/feedback/summary."""
    client = _get_claude_client()

    collected: list[str] = []
    with client.messages.stream(
        model=_claude_model,
        max_tokens=16384,
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


def format_violations(constraint_violations: dict[str, Any] | None) -> str:
    """Format constraint violations dict into text for LLM prompts."""
    if not constraint_violations:
        return "None"
    violation_lines = []
    for key, val in constraint_violations.items():
        if isinstance(val, list):
            for item in val:
                violation_lines.append(f"  {key}: {item}")
        else:
            violation_lines.append(f"  {key}: {val}")
    return "\n".join(violation_lines) if violation_lines else "None"


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
