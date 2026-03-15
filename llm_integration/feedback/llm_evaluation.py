"""LLM client and shared helpers for evaluation calls.

Provides:
- call_llm(): Lazily-initialized Anthropic client for making LLM calls
- format_violations(), format_error_info(): Shared formatting helpers
- extract_json_from_response(): JSON extraction from LLM responses
"""

from __future__ import annotations

import os
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

from .format_metrics import format_trajectory_metrics_text  # noqa: F401 (re-export)

# ---------------------------------------------------------------------------
# Shared LLM client (lazily initialized, no class)
# ---------------------------------------------------------------------------

_client: Anthropic | None = None
_model: str = ""
_max_tokens: int = 40000


def _get_client() -> Anthropic:
    """Get or create the global Anthropic client."""
    global _client, _model
    if _client is None:
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        _model = os.getenv("LLM_EVAL_MODEL", "claude-opus-4-5-20251101")
        if not api_key or api_key == "your_api_key_here":
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
        _client = Anthropic(api_key=api_key)
    return _client


def call_llm(
    system_prompt: str,
    user_message: str,
    images: list[str] | None = None,
) -> str:
    """Make an LLM call with optional images."""
    client = _get_client()
    content: list[dict[str, Any]] = []

    if images:
        content.append({"type": "text", "text": "TRAJECTORY FRAMES:"})
        for img_base64 in images:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_base64,
                    },
                }
            )

    content.append({"type": "text", "text": user_message})

    response_text = ""
    with client.messages.stream(
        model=_model,
        max_tokens=_max_tokens,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        for text in stream.text_stream:
            response_text += text

    return response_text


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
