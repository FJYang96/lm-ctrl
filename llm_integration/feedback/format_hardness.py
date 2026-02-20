"""Format constraint hardness analysis from slack formulation for LLM feedback."""

from typing import Any


def _identify_time_ranges(
    timesteps: list[int], dt: float = 0.02
) -> list[tuple[float, float]]:
    """
    Convert a list of timesteps into time ranges (in seconds).

    Args:
        timesteps: List of timestep indices
        dt: Time step size in seconds

    Returns:
        List of (start_time, end_time) tuples
    """
    if not timesteps:
        return []

    sorted_steps = sorted(timesteps)
    ranges = []
    range_start = sorted_steps[0]
    range_end = sorted_steps[0]

    for k in sorted_steps[1:]:
        if k == range_end + 1:
            # Continue current range
            range_end = k
        else:
            # End current range, start new one
            ranges.append((range_start * dt, (range_end + 1) * dt))
            range_start = k
            range_end = k

    # Add final range
    ranges.append((range_start * dt, (range_end + 1) * dt))
    return ranges


def _format_time_ranges(ranges: list[tuple[float, float]]) -> str:
    """Format time ranges into a readable string."""
    if not ranges:
        return "none"

    parts = []
    for start, end in ranges[:3]:  # Limit to first 3 ranges to avoid clutter
        if end - start <= 0.02:
            parts.append(f"{start:.2f}s")
        else:
            parts.append(f"{start:.2f}-{end:.2f}s")

    result = ", ".join(parts)
    if len(ranges) > 3:
        result += f" (+{len(ranges) - 3} more)"
    return result


def _find_worst_timesteps(
    slack_by_timestep: dict[int, float], top_n: int = 3, dt: float = 0.02
) -> list[tuple[float, float]]:
    """
    Find the timesteps with highest slack values.

    Returns:
        List of (time_in_seconds, slack_value) for worst timesteps
    """
    if not slack_by_timestep:
        return []

    sorted_items = sorted(slack_by_timestep.items(), key=lambda x: -x[1])
    return [(k * dt, v) for k, v in sorted_items[:top_n] if v > 1e-6]


def format_hardness_report(
    hardness_report: dict[str, dict[str, Any]] | None,
    dt: float = 0.02,
    current_slack_weights: dict[str, float] | None = None,
) -> str:
    """
    Format constraint hardness analysis from slack formulation.

    The hardness report shows which constraints are most difficult to satisfy,
    helping the LLM understand which bounds to relax and WHEN.

    Args:
        hardness_report: Dictionary mapping constraint names to hardness metrics
        dt: Time step size (default 0.02s)
        current_slack_weights: Current slack weights set by LLM (for display)

    Returns:
        Formatted string for inclusion in LLM feedback
    """
    if not hardness_report:
        return ""

    lines = [
        "",
        "=" * 80,
        "CONSTRAINT HARDNESS ANALYSIS (from Slack Formulation)",
        "=" * 80,
        "",
    ]

    # Show current slack weights if LLM has customized any
    if current_slack_weights:
        lines.append("YOUR CURRENT SLACK WEIGHTS (from previous iteration):")
        for name, weight in current_slack_weights.items():
            lines.append(f"  {name}: {weight:.0e}")
        lines.append("")
        lines.append("You can modify these with mpc.set_slack_weights({{...}})")
        lines.append("")

    lines.extend(
        [
            "This shows which constraints are HARDEST to satisfy and WHEN.",
            "High slack = constraint bounds are violated at those timesteps.",
            "",
        ]
    )

    # Separate LLM-controllable constraints from system constraints
    llm_constraints = {}
    system_constraints = {}

    for name, metrics in hardness_report.items():
        if "contact_aware" in name.lower() or "llm" in name.lower():
            llm_constraints[name] = metrics
        else:
            system_constraints[name] = metrics

    # Format LLM constraints (most important - LLM can fix these)
    if llm_constraints:
        lines.append(">>> YOUR CONSTRAINTS (you can fix these directly) <<<")
        lines.append("")
        for name, metrics in llm_constraints.items():
            lines.extend(_format_constraint_detail(name, metrics, dt, is_llm=True))
        lines.append("")

    # Format system constraints (for context)
    if system_constraints:
        lines.append(
            ">>> SYSTEM CONSTRAINTS (for context - adjust your motion/contact to avoid) <<<"
        )
        lines.append("")
        for name, metrics in system_constraints.items():
            max_slack = metrics.get("max_slack_Linf", 0)
            # Only show details for problematic system constraints
            if max_slack > 0.01:
                lines.extend(_format_constraint_detail(name, metrics, dt, is_llm=False))
            else:
                display_name = name.replace("_constraints", "")
                lines.append(f"  {display_name}: max slack = {max_slack:.4f}")
        lines.append("")

    return "\n".join(lines)


def _format_constraint_detail(
    name: str, metrics: dict[str, Any], dt: float, is_llm: bool = False
) -> list[str]:
    """Format detailed information for a single constraint."""
    lines = []

    max_slack = metrics.get("max_slack_Linf", 0)
    total_slack = metrics.get("total_slack_L1", 0)
    active_timesteps = metrics.get("active_timesteps", [])
    slack_by_timestep = metrics.get("slack_by_timestep", {})

    display_name = name.replace("_constraints", "")
    lines.append(f"  {display_name}")
    lines.append(f"      Max slack: {max_slack:.4f} | Total slack: {total_slack:.4f}")

    # Show when violations occur
    if active_timesteps:
        time_ranges = _identify_time_ranges(active_timesteps, dt)
        time_str = _format_time_ranges(time_ranges)
        lines.append(
            f"      Violated at: {time_str} ({len(active_timesteps)} timesteps)"
        )

        # Show worst timesteps
        worst = _find_worst_timesteps(slack_by_timestep, top_n=3, dt=dt)
        if worst:
            worst_str = ", ".join([f"t={t:.2f}s (slack={v:.3f})" for t, v in worst])
            lines.append(f"      Worst at: {worst_str}")

    lines.append("")
    return lines
