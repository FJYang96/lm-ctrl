"""Codegen LLM call for constraint generation.

Iteration 1: no feedback context.
Iteration 2+: samples best + 2 score-weighted past iterations,
sends their code + detailed performance summaries as context.
"""

from __future__ import annotations

import random
from typing import Any

from ..feedback.llm_evaluation import call_llm


def _select_iterations(summaries: list[dict[str, Any]], n: int = 3) -> list[int]:
    """Select iterations to show: always best + (n-1) score-weighted random samples.

    Returns sorted indices (no duplicates). If <= n total, returns all.
    """
    total = len(summaries)
    if total <= n:
        return list(range(total))

    best_idx = max(range(total), key=lambda i: summaries[i].get("score", 0))
    remaining = [i for i in range(total) if i != best_idx]
    weights = [max(summaries[i].get("score", 0.0), 0.05) for i in remaining]

    sampled = random.choices(
        remaining, weights=weights, k=min(n - 1, len(remaining))
    )
    # Deduplicate (random.choices can repeat)
    sampled_set = set(sampled)
    while len(sampled_set) < min(n - 1, len(remaining)):
        extra = random.choices(remaining, weights=weights, k=1)[0]
        sampled_set.add(extra)

    return sorted({best_idx} | sampled_set)


def _format_summary_fields(entry: dict[str, Any]) -> list[str]:
    """Format an iteration summary's fields for display."""
    fields = [
        ("Approach", "approach"), ("Solver", "solver"),
        ("Motion Quality", "motion_quality"), ("Metrics", "metrics"),
        ("Terminal", "terminal"), ("Hardness", "hardness"),
        ("Violations", "violations"), ("Reference", "reference"),
    ]
    lines: list[str] = []
    for label, key in fields:
        val = entry.get(key, "")
        if not val:
            continue
        val_lines = str(val).split("\n")
        lines.append(f"    {label:16s} {val_lines[0]}")
        for extra in val_lines[1:]:
            lines.append(f"    {'':16s} {extra}")
    return lines


def generate_constraints(
    system_prompt: str,
    user_message: str,
    *,
    iteration: int | None = None,
    command: str | None = None,
    run_dir: Any = None,
    iteration_summaries: list[dict[str, Any]] | None = None,
    mpc_dt: float | None = None,
) -> tuple[str, str | None]:
    """Generate optimization constraints using Claude.

    Iteration 1: calls LLM directly with user_message.
    Iteration 2+: builds context from sampled past iterations, then calls LLM.

    Returns (llm_response, feedback_context). feedback_context is None on iter 1.
    """
    if iteration is None:
        return call_llm(system_prompt, user_message), None

    assert iteration_summaries is not None
    summaries = iteration_summaries
    lines: list[str] = []

    # Task command
    lines.append("=" * 60)
    lines.append("                      TASK COMMAND")
    lines.append("=" * 60)
    lines.append(command if command else "No task command")

    # Terminology
    lines.append("")
    lines.append("--- TERMINOLOGY ---")
    lines.append(
        "SOLVER CONVERGED = optimizer found feasible solution (may not match task goal). "
        "SOLVER FAILED = no feasible solution found. "
        "Score (0.0-1.0) = LLM judgment of task achievement. "
        "Failed solve scores capped at 0.40."
    )

    # All iteration scores (quick overview)
    best_idx = max(range(len(summaries)), key=lambda i: summaries[i].get("score", 0))
    lines.append("")
    lines.append("=" * 60)
    lines.append("                 ALL ITERATION SCORES")
    lines.append("=" * 60)
    score_parts: list[str] = []
    for i, s in enumerate(summaries):
        status = "OK" if s.get("success") else "FAIL"
        tag = " [BEST]" if i == best_idx else ""
        score_parts.append(f"Iter {s.get('iteration', i+1)}: {s.get('score', 0):.2f} [{status}]{tag}")
    # Show 4 per line
    for row_start in range(0, len(score_parts), 4):
        lines.append("  " + " | ".join(score_parts[row_start:row_start + 4]))

    # Sampled iterations (best + 2 random, with code + summary)
    selected = _select_iterations(summaries)
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"        SAMPLED ITERATIONS ({len(selected)} detailed)")
    lines.append("=" * 60)

    for idx in selected:
        entry = summaries[idx]
        status = "SOLVER CONVERGED" if entry.get("success") else "SOLVER FAILED"
        best_tag = " [BEST]" if idx == best_idx else ""
        lines.append("")
        lines.append(
            f"  Iter {entry.get('iteration', idx+1)} [{status}] "
            f"Score: {entry.get('score', 0):.2f}{best_tag}"
        )
        lines.extend(_format_summary_fields(entry))

        code = entry.get("constraint_code", "")
        if code:
            lines.append(f"    --- Code (Iter {entry.get('iteration', idx+1)}) ---")
            for code_line in code.split("\n"):
                lines.append(f"    | {code_line}")
            lines.append("    --- End Code ---")

    lines.append("")
    lines.append("=" * 60)
    lines.append(
        "Generate improved constraints and reference trajectory. "
        "You have sampled iterations above with their code, scores, and detailed "
        "performance summaries. Learn from ALL of them — high-scoring iterations "
        "show what works, low-scoring iterations show what to avoid and why. "
        "Don't repeat approaches that scored poorly. Build on what scored well. "
        "You decide whether to tweak a good approach or pivot to something new."
    )
    lines.append("Return ONLY Python code.")
    lines.append("=" * 60)

    feedback_ctx = "\n".join(lines)
    full_message = f"{feedback_ctx}\n\n{user_message}"
    return call_llm(system_prompt, full_message), feedback_ctx
