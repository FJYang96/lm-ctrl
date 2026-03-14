"""LLM-based evaluation for iteration analysis.

Replaces manual string matching with LLM-based semantic analysis
for scoring, progress tracking, and diagnostic warnings.
"""

from __future__ import annotations

import json
import os
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

from ..logging_config import logger
from .format_metrics import format_trajectory_metrics_text


def _extract_json_from_response(response: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = response.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        # Find the end of the first line (```json or ```)
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]

        # Remove trailing ```
        if text.endswith("```"):
            text = text[:-3]
        elif "```" in text:
            # Handle case where ``` is followed by more text
            text = text[: text.rfind("```")]

    return text.strip()


class LLMEvaluator:
    """Evaluates iterations using LLM for semantic understanding."""

    def __init__(self) -> None:
        """Initialize the evaluator."""
        load_dotenv()

        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = os.getenv("LLM_EVAL_MODEL", "claude-opus-4-5-20251101")
        self.max_tokens = 40000  # Increased to handle full JSON response

        if not self.api_key or self.api_key == "your_api_key_here":
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")

        self.client = Anthropic(api_key=self.api_key)

    def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        images: list[str] | None = None,
    ) -> str:
        """Make an LLM call with optional images."""
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

        # Use streaming to avoid timeout after long optimization runs
        response_text = ""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        ) as stream:
            for text in stream.text_stream:
                response_text += text

        return response_text

    def evaluate_iteration_unified(
        self,
        command: str,
        trajectory_analysis: dict[str, Any],
        constraint_code: str,
        opt_success: bool,
        error_info: dict[str, Any] | None = None,
        images: list[str] | None = None,
        visual_summary: str = "",
        hardness_text: str = "",
        constraint_violations: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Unified evaluation for both successful and failed iterations.

        Returns dict with: score, criteria, warnings, summary
        """
        system_prompt = """You are an expert robotics engineer evaluating a quadruped robot trajectory optimization result.

Analyze the constraint code, trajectory metrics, solver status, and visual summary to evaluate task completion.

Return a JSON object with this structure:
{
    "score": <float 0.0-1.0>,
    "criteria": [
        {
            "name": "<criterion name>",
            "target": "<specific numerical target>",
            "achieved": "<exact numerical result>",
            "progress": <float 0.0-1.0>
        }
    ],
    "warnings": ["<specific technical warning>"],
    "summary": "<detailed 4-5 sentence analysis>"
}

=== SCORING GUIDELINES ===
Evaluate the trajectory holistically, considering ALL of the following factors:
  - Primary goal progress: How much of the commanded task was achieved?
  - Axis purity: Was motion on the intended axis only? Was there unwanted off-axis rotation?
  - Landing/terminal state: Stable final configuration? Low final velocity?
  - Solver health: Clean convergence vs timeout/infeasibility?

Weigh these factors according to their importance for the specific task. Use your expert judgment
to produce a precise score — do NOT round to convenient numbers like 0.70 or 0.75.
If the solver FAILED, cap the score at 0.85 maximum.

=== SUMMARY REQUIREMENTS (4-5 detailed sentences) ===
1. CONSTRAINT STRATEGY: Describe exactly what constraints were used with specific numerical bounds
2. NUMERICAL RESULTS: State exact outcomes for rotation (rad AND degrees), height (meters), flight duration (seconds)
3. ROOT CAUSE ANALYSIS: Explain WHY this result occurred
4. SPECIFIC RECOMMENDATIONS: Provide exact parameter changes or alternative approaches

Return ONLY valid JSON, no markdown, no extra text."""

        metrics_text = (
            self._format_metrics(trajectory_analysis, opt_success)
            if trajectory_analysis
            else "No trajectory data available"
        )

        solver_status = (
            "CONVERGED (success)" if opt_success else "FAILED (did not converge)"
        )

        error_text = ""
        if error_info:
            error_text = self._format_error_info(error_info)

        hardness_section = ""
        if hardness_text:
            hardness_section = hardness_text

        violations_section = ""
        if constraint_violations:
            violation_lines = []
            for key, val in constraint_violations.items():
                if isinstance(val, list):
                    for item in val:
                        violation_lines.append(f"  {key}: {item}")
                else:
                    violation_lines.append(f"  {key}: {val}")
            if violation_lines:
                violations_section = "\n".join(violation_lines)

        user_message = f"""<task>{command}</task>
<solver status="{solver_status}">{error_text}</solver>
<metrics>
{metrics_text}
</metrics>
<visual_summary>
{visual_summary if visual_summary else "Not available"}
</visual_summary>
<constraint_code>
{constraint_code}
</constraint_code>
<hardness>
{hardness_section if hardness_section else "Not available"}
</hardness>
<violations>
{violations_section if violations_section else "None"}
</violations>

Evaluate how well this trajectory achieves the commanded task."""

        try:
            response = self._call_llm(system_prompt, user_message, images)
            json_text = _extract_json_from_response(response)
            result: dict[str, Any] = json.loads(json_text)
            # Cap failed iteration scores at 0.85
            if not opt_success and result.get("score", 0) > 0.85:
                result["score"] = 0.85
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Unified evaluation failed: invalid JSON - {e}")
            return self._default_evaluation()
        except Exception as e:
            logger.error(
                f"Unified LLM evaluation failed: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return self._default_evaluation()

    def generate_iteration_summary(
        self,
        command: str,
        iteration: int,
        score: float,
        constraint_code: str,
        feedback: str,
        trajectory_analysis: dict[str, Any],
        opt_success: bool,
        simulation_result: dict[str, Any] | None = None,
        visual_summary: str = "",
    ) -> dict[str, Any]:
        """
        Generate a structured iteration summary for the history log.

        Returns a structured dict with multi-line fields (prose paragraphs).
        """
        system_prompt = """You are summarizing a trajectory optimization iteration for a history log.
A code-generation LLM will read this summary to understand what was tried, what happened, and
what to do differently next time. Be DETAILED and SPECIFIC — vague summaries are useless.

You receive:
- Full constraint code (read it to describe the approach)
- Trajectory metrics (position, velocity, orientation, timing, GRF, actuator)
- Unified feedback from a specialized LLM call covering both constraints and reference trajectory
  (already contains analysis of hardness data, violations, reference RMSE, and plausibility)
- Visual summary of the trajectory

Return a JSON object with this structure:
{
    "iteration": <int>,
    "score": <float>,
    "success": <bool>,
    "approach": "<DETAILED prose paragraph>",
    "feedback_summary": "<DETAILED prose paragraph>",
    "simulation_summary": "<DETAILED prose paragraph>",
    "metrics_summary": "<prose paragraph with all key metrics>"
}

=== FIELD REQUIREMENTS — ALL FIELDS MUST BE PROSE PARAGRAPHS ===

Write each field as a detailed paragraph of flowing sentences. No markdown headers, no bullet
lists, no asterisks, no code blocks. Just sentences forming coherent paragraphs.

approach (one detailed paragraph):
  Describe the full constraint and reference strategy together. Include the contact sequence with
  exact phase names and durations, each constraint with its variable, bounds, and timing, specific
  numerical values, and the reference trajectory design (target rotation in rad and degrees, height
  profile, velocity profile, how it was built). Explain how constraints and reference work together.

feedback_summary (one detailed paragraph):
  Summarize the unified feedback. Which constraints worked and which failed (cite slack values),
  specific bound values that need changing and why, root cause of any failure, RMSE values for
  height/pitch/vz, physics plausibility issues, phase timing problems, and prioritized
  recommendations with concrete numbers.

simulation_summary (one detailed paragraph):
  Whether rendering succeeded or failed, what the robot actually did (visible motion from video
  frames), landing quality, and any ground penetration or instability.

metrics_summary (one detailed paragraph):
  All key metrics as sentences: pitch rotation (rad AND degrees), yaw drift, roll, height gain,
  max height, final height, flight duration, total duration, solver status (converged/failed,
  iteration count), rendering success/failure.

Return ONLY valid JSON, no markdown, no extra text."""

        metrics_text = (
            self._format_metrics(trajectory_analysis, opt_success)
            if trajectory_analysis
            else "No trajectory data"
        )

        sim_text = ""
        if simulation_result:
            sim_success = simulation_result.get("success", False)
            sim_text = f"Rendering: {'success' if sim_success else 'failed'}"
            if simulation_result.get("error"):
                sim_text += f"\nRendering error: {str(simulation_result['error'])}"

        solver_status = "converged" if opt_success else "failed"

        user_message = f"""<task>{command}</task>
<iteration number="{iteration}" score="{score:.2f}" solver="{solver_status}"/>
<metrics>
{metrics_text}
</metrics>
<simulation>{sim_text}</simulation>
<constraint_code>
{constraint_code}
</constraint_code>
<feedback>
{feedback if feedback else "None"}
</feedback>
<visual_summary>
{visual_summary if visual_summary else "Not available"}
</visual_summary>"""

        try:
            response = self._call_llm(system_prompt, user_message, None)
            json_text = _extract_json_from_response(response)
            result: dict[str, Any] = json.loads(json_text)
            # Ensure required fields
            result.setdefault("iteration", iteration)
            result.setdefault("score", score)
            result.setdefault("success", opt_success)
            return result
        except Exception as e:
            logger.error(f"Iteration summary generation failed: {e}")
            return {
                "iteration": iteration,
                "score": score,
                "success": opt_success,
                "approach": "Summary generation failed",
                "feedback_summary": "",
                "simulation_summary": "",
                "metrics_summary": metrics_text,
            }

    def _format_metrics(self, ta: dict[str, Any], opt_success: bool = True) -> str:
        """Format trajectory metrics for the LLM (shared comprehensive formatter)."""
        return format_trajectory_metrics_text(ta, opt_success)

    def _format_error_info(self, error_info: dict[str, Any]) -> str:
        """Format error information for the LLM."""
        lines = []
        if error_info.get("error_message"):
            lines.append(f"Error: {error_info['error_message']}")
        if error_info.get("solver_iterations"):
            lines.append(
                f"Solver stopped after {error_info['solver_iterations']} iterations"
            )
        if error_info.get("constraint_violations"):
            lines.append(f"Violations: {str(error_info['constraint_violations'])}")
        return "\n".join(lines) if lines else "No detailed error information"

    def _default_evaluation(self) -> dict[str, Any]:
        """Return default evaluation when LLM fails."""
        return {
            "score": 0.0,
            "criteria": [],
            "warnings": ["Could not parse LLM evaluation"],
            "summary": "Evaluation failed - using default score of 0.0(Worst possible score)",
        }


# Global evaluator instance
_evaluator: LLMEvaluator | None = None


def get_evaluator() -> LLMEvaluator:
    """Get or create the global LLM evaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = LLMEvaluator()
    return _evaluator


def evaluate_iteration_unified(
    command: str,
    trajectory_analysis: dict[str, Any],
    constraint_code: str,
    opt_success: bool,
    error_info: dict[str, Any] | None = None,
    images: list[str] | None = None,
    visual_summary: str = "",
    hardness_text: str = "",
    constraint_violations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Unified evaluation for both success and failure. Returns: score, criteria, warnings, summary."""
    return get_evaluator().evaluate_iteration_unified(
        command,
        trajectory_analysis,
        constraint_code,
        opt_success,
        error_info,
        images,
        visual_summary,
        hardness_text,
        constraint_violations,
    )


def generate_iteration_summary(
    command: str,
    iteration: int,
    score: float,
    constraint_code: str,
    feedback: str,
    trajectory_analysis: dict[str, Any],
    opt_success: bool,
    simulation_result: dict[str, Any] | None = None,
    visual_summary: str = "",
) -> dict[str, Any]:
    """Generate a structured iteration summary for history."""
    return get_evaluator().generate_iteration_summary(
        command,
        iteration,
        score,
        constraint_code,
        feedback,
        trajectory_analysis,
        opt_success,
        simulation_result,
        visual_summary,
    )
