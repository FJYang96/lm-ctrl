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

    def evaluate_successful_iteration(
        self,
        command: str,
        trajectory_analysis: dict[str, Any],
        constraint_code: str,
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a successful optimization iteration using LLM.

        Returns dict with: score, criteria, warnings, summary
        """
        system_prompt = """You are an expert robotics engineer evaluating a quadruped robot trajectory optimization result.

Analyze the constraint code and trajectory metrics to evaluate task completion.

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

=== EVALUATION REQUIREMENTS ===
Based on the command, determine what criteria are relevant and evaluate them.
Identify any warnings about the trajectory or constraint approach.

=== SUMMARY REQUIREMENTS (4-5 detailed sentences) ===
1. CONSTRAINT STRATEGY: Describe exactly what constraints were used with specific numerical bounds and parameters

2. NUMERICAL RESULTS: State exact outcomes for rotation (rad AND degrees), height (meters), flight duration (seconds), final velocity

3. ROOT CAUSE ANALYSIS: Explain WHY this result occurred - identify loopholes, bound issues, or constraint flaws

4. SPECIFIC RECOMMENDATIONS: Provide exact parameter changes or alternative approaches to try

=== SCORING GUIDELINES ===
- 0.0-0.2: Complete failure - no meaningful motion toward goal
- 0.2-0.4: Wrong motion - moved but in wrong direction/axis
- 0.4-0.6: Partial progress - some correct motion but far from complete
- 0.6-0.8: Good progress - majority of task complete, minor issues
- 0.8-0.95: Nearly complete - task almost done, small adjustments needed
- 0.95-1.0: Success - task fully completed

Return ONLY valid JSON, no markdown, no extra text."""

        metrics_text = self._format_metrics(trajectory_analysis)

        user_message = f"""COMMAND: {command}

TRAJECTORY METRICS:
{metrics_text}

CONSTRAINT CODE USED:
```python
{constraint_code[:1500]}
```

Evaluate how well this trajectory achieves the commanded task."""

        try:
            response = self._call_llm(system_prompt, user_message, images)

            # Extract JSON from markdown code blocks if present
            json_text = _extract_json_from_response(response)
            result: dict[str, Any] = json.loads(json_text)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Evaluation failed: invalid JSON - {e}")
            return self._default_evaluation()
        except Exception as e:
            logger.error(
                f"LLM evaluation failed: {type(e).__name__}: {e}", exc_info=True
            )
            return self._default_evaluation()

    def evaluate_failed_iteration(
        self,
        command: str,
        trajectory_analysis: dict[str, Any],
        constraint_code: str,
        error_info: dict[str, Any],
        images: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate a FAILED optimization iteration using LLM.

        The solver did not converge, but partial progress may have been made.
        Score based on how close the solver's last attempt was to the goal.

        Returns dict with: score, criteria, warnings, summary
        """
        system_prompt = """You are an expert robotics engineer evaluating a quadruped robot trajectory optimization that FAILED (solver did not converge or timed out).

Even though the solver failed, there may be significant partial progress. Score based on what the solver was ATTEMPTING and how close it got.

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

=== EVALUATION REQUIREMENTS ===
Based on the command, determine what criteria are relevant.
Look at the trajectory metrics from the solver's LAST ATTEMPT (debug values).
If the solver was making good progress (e.g., 445° rotation for a backflip), score it highly even though it failed.
If the solver made no progress (e.g., 3° rotation), score it very low.

=== SCORING GUIDELINES ===
- 0.0-0.1: No meaningful motion at all, or no trajectory data
- 0.1-0.3: Some motion but in wrong direction/axis or minimal progress
- 0.3-0.5: Moderate progress toward goal (e.g., 30-50% of target rotation/height)
- 0.5-0.7: Significant progress (e.g., 50-80% of target achieved but solver couldn't converge)
- 0.7-0.85: Nearly achieved the goal but solver ran out of iterations or hit minor infeasibility
- 0.85-1.0: Reserved for successful iterations only — failures should not exceed 0.85

=== CRITICAL ===
A timeout after achieving 80%+ of the goal is MUCH better than a quick failure with 0% progress.
Do NOT score all failures as 0.0 — differentiate based on partial progress.

Return ONLY valid JSON, no markdown, no extra text."""

        metrics_text = (
            self._format_metrics(trajectory_analysis)
            if trajectory_analysis
            else "No trajectory data available"
        )
        error_text = self._format_error_info(error_info)

        user_message = f"""COMMAND: {command}

SOLVER STATUS: FAILED
{error_text}

TRAJECTORY METRICS (from solver's last attempt / debug values):
{metrics_text}

CONSTRAINT CODE USED:
```python
{constraint_code[:1500]}
```

Score how close the solver's last attempt was to achieving the commanded task, even though it failed."""

        try:
            response = self._call_llm(system_prompt, user_message, images)
            json_text = _extract_json_from_response(response)
            result: dict[str, Any] = json.loads(json_text)
            # Cap failed iteration scores at 0.85
            if result.get("score", 0) > 0.85:
                result["score"] = 0.85
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed iteration evaluation: invalid JSON - {e}")
            return self._default_evaluation_failed()
        except Exception as e:
            logger.error(
                f"LLM failed evaluation error: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return self._default_evaluation_failed()

    def _default_evaluation_failed(self) -> dict[str, Any]:
        """Return default evaluation for failed iterations when LLM fails."""
        return {
            "score": 0.0,
            "criteria": [],
            "warnings": ["Could not parse LLM evaluation for failed iteration"],
            "summary": "Evaluation failed - using default score of 0.0",
        }

    def summarize_failed_iteration(
        self,
        command: str,
        constraint_code: str,
        error_info: dict[str, Any],
        trajectory_analysis: dict[str, Any] | None = None,
        images: list[str] | None = None,
    ) -> str:
        """Generate a detailed summary for a failed iteration.

        Args:
            command: The task command
            constraint_code: The constraint code that failed
            error_info: Error information from the solver
            trajectory_analysis: Metrics from the debug trajectory (if available)
            images: Video frames from the debug trajectory (if available)
        """
        system_prompt = """You are analyzing a FAILED robot trajectory optimization attempt. Generate a detailed 4-5 sentence technical analysis.

If video frames are provided, USE THEM to understand what the solver was attempting before it gave up.

=== REQUIRED CONTENT ===

1. CONSTRAINT APPROACH: Describe exactly what constraints were used
   - Constraint type (Gaussian, linear, terminal, phase-based)
   - Exact numerical bounds
   - Parameters and timing
   - Contact phase handling

2. PARTIAL PROGRESS: What did the solver achieve before failing?
   - Look at the video frames to see what motion was attempted
   - Rotation achieved (if any trajectory data)
   - Height reached
   - Was the solver on the right track? How can it make a better trajectory that matches the command in fewer iterations?

3. FAILURE DIAGNOSIS: Identify the specific technical failure mode
   - Infeasibility at initial state (k=0)
   - CasADi symbolic math errors (if/else with MX variables)
   - Conflicting/mutually exclusive bounds
   - Bounds too tight for feasible trajectory
   - Solver divergence or numerical issues
   - OR: Solver was making good progress but ran out of iterations

4. ROOT CAUSE ANALYSIS: Explain WHY this approach failed
   - Code bug vs physics impossibility
   - Unrealistic constraint values
   - Contact sequence mismatch
   - Solver needed more iterations (if significant progress was made)

5. SPECIFIC FIX RECOMMENDATION: What exact changes would fix this?
   - Specific parameter values to try
   - Alternative constraint approaches
   - Bounds adjustments needed

DON'T JUST SAY THE SOLVER NEEDED MORE ITERATIONS; FIX THE CONSTRAINT CODE SO THAT THE SOLVER CAN CONVERGE IN FEWER ITERATIONS.
ANALYZE THE CONSTRAINT CODE AND THE TRAJECTORY METRICS TO FIND THE ROOT CAUSE OF THE FAILURE AND FIX IT.

Return ONLY the analysis text, no quotes or markdown."""

        error_text = self._format_error_info(error_info)
        metrics_text = (
            self._format_metrics_brief(trajectory_analysis)
            if trajectory_analysis
            else "No trajectory data"
        )

        user_message = f"""COMMAND: {command}

CONSTRAINT CODE:
```python
{constraint_code[:200000]}
```

TRAJECTORY METRICS (from solver's last attempt):
{metrics_text}

ERROR:
{error_text}"""

        response = self._call_llm(system_prompt, user_message, images)
        return response.strip().strip('"').strip("'")

    def summarize_successful_iteration(
        self,
        command: str,
        constraint_code: str,
        trajectory_analysis: dict[str, Any],
        score: float,
    ) -> str:
        """Generate a detailed summary for a successful iteration."""
        system_prompt = """You are analyzing a SUCCESSFUL (solver converged) robot trajectory optimization. Generate a detailed 4-5 sentence technical analysis.

Note: "Successful" means the solver converged, NOT that the task was completed. The robot may have found a solution that doesn't achieve the goal.

=== REQUIRED CONTENT ===

1. CONSTRAINT STRATEGY ANALYSIS:
   - Constraint type (Gaussian, linear, terminal, phase-based)
   - Exact numerical bounds used
   - Parameters and timing
   - Contact phase handling

2. NUMERICAL RESULTS (use exact numbers from metrics):
   - Pitch rotation in rad AND degrees (what % of target?)
   - Yaw rotation in rad AND degrees (intended or not?)
   - Roll rotation in rad AND degrees (intended or not?)
   - Height gain in meters
   - Flight duration in seconds
   - Final velocity

3. SUCCESS/FAILURE ANALYSIS:
   If task NOT achieved, identify WHY:
   - Optimizer loophole (found way to satisfy constraints without doing task)
   - Wrong rotation axis
   - Bounds too loose (allowed zero/minimal motion)
   - Missing constraint

   If task achieved:
   - What made this approach work?
   - Key insight or technique

4. SPECIFIC IMPROVEMENT RECOMMENDATIONS:
   For incomplete tasks, provide exact changes:
   - Specific parameter values to modify
   - Constraints to add/remove
   - Bound adjustments (especially if bounds include zero when they shouldn't)

=== CRITICAL PATTERNS TO IDENTIFY ===

LOOPHOLE DETECTION:
- Pitch rotation ≈ 0 but height gain > 0.2m → robot jumped without rotating
- Yaw/roll rotation > pitch rotation for backflip → wrong axis
- Bounds that include zero allow optimizer to avoid rotation

BOUND ANALYSIS:
- Bounds like [-15, 5] ALLOW zero → optimizer will choose zero
- Bounds like [-15, -5] FORCE negative values → rotation required

Return ONLY the analysis text, no quotes or markdown."""

        metrics_text = self._format_metrics_brief(trajectory_analysis)

        user_message = f"""COMMAND: {command}

CONSTRAINT CODE:
```python
{constraint_code[:1500]}
```

METRICS: {metrics_text}
Score: {score:.2f}"""

        try:
            response = self._call_llm(system_prompt, user_message)
            return response.strip().strip('"').strip("'")
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Optimization succeeded with score {score:.2f}"

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

    def _format_metrics_brief(self, ta: dict[str, Any]) -> str:
        """Format key metrics briefly."""
        return (
            f"Pitch: {ta.get('total_pitch_rotation', 0):.2f} rad, "
            f"Yaw: {ta.get('max_yaw', 0):.2f} rad, "
            f"Height gain: {ta.get('height_gain', 0):.3f}m"
        )

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
            lines.append(
                f"Violations: {str(error_info['constraint_violations'])[:500]}"
            )
        return "\n".join(lines) if lines else "No detailed error information"

    def _default_evaluation(self) -> dict[str, Any]:
        """Return default evaluation when LLM fails."""
        return {
            "score": 0.5,
            "criteria": [],
            "warnings": ["Could not parse LLM evaluation"],
            "summary": "Evaluation failed - using default score",
        }


# Global evaluator instance
_evaluator: LLMEvaluator | None = None


def get_evaluator() -> LLMEvaluator:
    """Get or create the global LLM evaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = LLMEvaluator()
    return _evaluator


def evaluate_iteration(
    command: str,
    trajectory_analysis: dict[str, Any],
    constraint_code: str,
    images: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a successful iteration. Returns: score, criteria, warnings, summary."""
    return get_evaluator().evaluate_successful_iteration(
        command, trajectory_analysis, constraint_code, images
    )


def evaluate_failed_iteration(
    command: str,
    trajectory_analysis: dict[str, Any],
    constraint_code: str,
    error_info: dict[str, Any],
    images: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a failed iteration. Returns: score, criteria, warnings, summary."""
    return get_evaluator().evaluate_failed_iteration(
        command, trajectory_analysis, constraint_code, error_info, images
    )


def summarize_iteration(
    command: str,
    constraint_code: str,
    success: bool,
    error_info: dict[str, Any] | None = None,
    trajectory_analysis: dict[str, Any] | None = None,
    score: float = 0.0,
    images: list[str] | None = None,
) -> str:
    """Generate an iteration summary for history.

    Args:
        images: Video frames (used for failed iterations to show what solver attempted)
    """
    evaluator = get_evaluator()
    if success and trajectory_analysis:
        return evaluator.summarize_successful_iteration(
            command, constraint_code, trajectory_analysis, score
        )
    else:
        return evaluator.summarize_failed_iteration(
            command,
            constraint_code,
            error_info or {},
            trajectory_analysis or {},
            images,
        )
