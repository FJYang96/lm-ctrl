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
Compute the score as a WEIGHTED AVERAGE of these factors (each 0.0-1.0):
  - Primary goal progress (60%): Fraction of the commanded task achieved
  - Axis purity (15%): Was motion on the intended axis only? Penalize unwanted off-axis rotation
  - Landing/terminal state (15%): Stable final configuration? Low final velocity?
  - Solver health (10%): Clean convergence vs timeout/infeasibility

score = 0.6 * primary_progress + 0.15 * axis_purity + 0.15 * landing_quality + 0.10 * solver_health

IMPORTANT: Use the formula to compute a precise score. Do NOT round to convenient numbers like 0.70 or 0.75.
If the solver FAILED, cap the score at 0.85 maximum.

=== SUMMARY REQUIREMENTS (4-5 detailed sentences) ===
1. CONSTRAINT STRATEGY: Describe exactly what constraints were used with specific numerical bounds
2. NUMERICAL RESULTS: State exact outcomes for rotation (rad AND degrees), height (meters), flight duration (seconds)
3. ROOT CAUSE ANALYSIS: Explain WHY this result occurred
4. SPECIFIC RECOMMENDATIONS: Provide exact parameter changes or alternative approaches

Return ONLY valid JSON, no markdown, no extra text."""

        metrics_text = (
            self._format_metrics(trajectory_analysis)
            if trajectory_analysis
            else "No trajectory data available"
        )

        solver_status = (
            "CONVERGED (success)" if opt_success else "FAILED (did not converge)"
        )

        error_text = ""
        if error_info:
            error_text = f"\nERROR INFO:\n{self._format_error_info(error_info)}"

        visual_text = ""
        if visual_summary:
            visual_text = f"\nVISUAL SUMMARY:\n{visual_summary}"

        user_message = f"""COMMAND: {command}

SOLVER STATUS: {solver_status}
{error_text}

TRAJECTORY METRICS:
{metrics_text}
{visual_text}

CONSTRAINT CODE USED:
```python
{constraint_code}
```

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
        constraint_feedback: str,
        reference_feedback: str,
        trajectory_analysis: dict[str, Any],
        opt_success: bool,
        simulation_result: dict[str, Any] | None = None,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a structured iteration summary for the history log.

        Returns a structured dict with multi-line fields.
        """
        system_prompt = """You are summarizing a trajectory optimization iteration for a history log.
A code-generation LLM will read this summary to understand what was tried, what happened, and
what to do differently next time. Be DETAILED and SPECIFIC — vague summaries are useless.

Return a JSON object with this structure:
{
    "iteration": <int>,
    "score": <float>,
    "success": <bool>,
    "constraint_approach": "<DETAILED multi-line description>",
    "reference_approach": "<DETAILED multi-line description>",
    "constraint_feedback_summary": "<DETAILED 4-6 sentence summary>",
    "reference_feedback_summary": "<DETAILED 4-6 sentence summary>",
    "simulation_summary": "<DETAILED 3-5 sentence summary>",
    "metrics_summary": "<all key metrics with exact numbers>"
}

=== FIELD REQUIREMENTS ===

constraint_approach (DETAILED — 5+ lines):
  - Contact sequence: exact phase names, durations, and patterns
  - Each constraint function: what variable is constrained, what bounds, what timing
  - Specific numerical values for all bounds and parameters
  - Phase-awareness: how constraints change across stance/flight/landing

reference_approach (DETAILED — 5+ lines):
  - Target rotation (rad and degrees), height profile, velocity profile
  - Phase breakdown: what happens in each phase (stance push, flight, landing)
  - Specific numerical targets: peak height, takeoff velocity, angular velocity
  - How the reference was built (min_jerk, ballistic, manual interpolation)

constraint_feedback_summary (4-6 sentences):
  - Which constraints worked and which failed
  - Specific bound values that need changing and why
  - Root cause of any solver failure or constraint violation
  - Exact recommendations from the constraint feedback

reference_feedback_summary (4-6 sentences):
  - RMSE between reference and actual trajectory (height, pitch, velocity)
  - Physics plausibility issues found
  - Phase timing alignment problems
  - Specific parameter changes recommended

simulation_summary (3-5 sentences):
  - Whether simulation succeeded or failed and why
  - Tracking error magnitude and what it means
  - What the robot actually DID (visible motion from video frames)
  - Landing quality and any ground penetration or instability

metrics_summary (compact but COMPLETE):
  - Pitch rotation (rad AND degrees), yaw drift, roll
  - Height gain, max height, final height
  - Flight duration, total duration
  - Solver status (converged/failed, iteration count)
  - Tracking error, simulation success/failure

Return ONLY valid JSON, no markdown, no extra text."""

        metrics_text = (
            self._format_metrics(trajectory_analysis)
            if trajectory_analysis
            else "No trajectory data"
        )

        sim_text = ""
        if simulation_result:
            sim_success = simulation_result.get("success", False)
            tracking_err = simulation_result.get("tracking_error", "N/A")
            sim_text = f"Simulation: {'success' if sim_success else 'failed'}, tracking_error={tracking_err}"
            if simulation_result.get("error"):
                sim_text += f"\nSimulation error: {str(simulation_result['error'])}"

        user_message = f"""COMMAND: {command}
ITERATION: {iteration}
SCORE: {score:.2f}
SOLVER: {"converged" if opt_success else "failed"}

TRAJECTORY METRICS:
{metrics_text}

{sim_text}

FULL CONSTRAINT CODE:
```python
{constraint_code}
```

CONSTRAINT FEEDBACK (full):
{constraint_feedback if constraint_feedback else "None"}

REFERENCE FEEDBACK (full):
{reference_feedback if reference_feedback else "None"}"""

        try:
            response = self._call_llm(system_prompt, user_message, images)
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
                "constraint_approach": "Summary generation failed",
                "reference_approach": "Summary generation failed",
                "constraint_feedback_summary": "",
                "reference_feedback_summary": "",
                "simulation_summary": "",
                "metrics_summary": metrics_text,
            }

    def _format_metrics(self, ta: dict[str, Any]) -> str:
        """Format trajectory metrics for the LLM."""
        total_pitch = ta.get("total_pitch_rotation", 0)
        max_yaw = ta.get("max_yaw", 0)
        total_roll = ta.get("total_roll_rotation", 0)

        return f"""Height: initial={ta.get("initial_com_height", 0):.3f}m, max={ta.get("max_com_height", 0):.3f}m, gain={ta.get("height_gain", 0):.3f}m
Pitch rotation: {total_pitch:.2f} rad ({abs(total_pitch) * 57.3:.0f} degrees)
Yaw rotation: {max_yaw:.2f} rad ({abs(max_yaw) * 57.3:.0f} degrees)
Roll rotation: {total_roll:.2f} rad ({abs(total_roll) * 57.3:.0f} degrees)
Max angular velocity: {ta.get("max_angular_vel", 0):.2f} rad/s
Flight duration: {ta.get("flight_duration", 0):.2f}s
Final COM velocity: {ta.get("final_com_velocity", 0):.2f} m/s"""

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
    )


def generate_iteration_summary(
    command: str,
    iteration: int,
    score: float,
    constraint_code: str,
    constraint_feedback: str,
    reference_feedback: str,
    trajectory_analysis: dict[str, Any],
    opt_success: bool,
    simulation_result: dict[str, Any] | None = None,
    images: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a structured iteration summary for history."""
    return get_evaluator().generate_iteration_summary(
        command,
        iteration,
        score,
        constraint_code,
        constraint_feedback,
        reference_feedback,
        trajectory_analysis,
        opt_success,
        simulation_result,
        images,
    )
