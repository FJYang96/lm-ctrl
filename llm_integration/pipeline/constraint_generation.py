"""Constraint generation with LLM retry logic."""

from typing import TYPE_CHECKING, Any

from ..code_generation import extract_raw_code, generate_constraints
from ..code_generation.prompts import create_repair_prompt
from ..logging_config import logger

if TYPE_CHECKING:
    from .feedback_pipeline import FeedbackPipeline


def generate_constraints_with_retry(
    self: "FeedbackPipeline",
    system_prompt: str,
    user_message: str,
    feedback_data: dict[str, Any] | None = None,
    command: str = "",
    max_attempts: int = 10,
) -> tuple[str, str, list[dict[str, Any]], str | None]:
    """
    Generate constraints using the LLM with comprehensive auto-retry on failures.

    This implements the full repair loop with detailed error feedback to the LLM.

    Args:
        system_prompt: System prompt for the LLM.
        user_message: Initial user prompt (e.g. "Generate MPC for: backflip").
        feedback_data: Structured feedback from the previous iteration (None on
            iteration 1).  Passed as keyword args to ``generate_constraints``
            which builds the context and calls the LLM in one shot.
        command: Original natural-language command (for repair prompts).
        max_attempts: Maximum LLM repair attempts.

    Returns:
        Tuple of (final_code, function_name, attempt_log, feedback_context).
        ``feedback_context`` is the assembled context string (None on iteration 1).
    """
    from ..mpc import LLMTaskMPC

    attempts: list[dict[str, Any]] = []
    mpc_config_code = ""  # Initialize for exception handling
    feedback_context: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            # Generate code from LLM
            if attempt == 1:
                if feedback_data:
                    # Iteration 2+: build context + call LLM in one shot
                    response, feedback_context = generate_constraints(
                        system_prompt, user_message, **feedback_data
                    )
                else:
                    # Iteration 1: user_message is the whole prompt
                    response, _ = generate_constraints(system_prompt, user_message)
            else:
                # Subsequent attempts use repair prompts with detailed error feedback
                failed_code = attempts[-1]["code"]
                error_msg = attempts[-1]["error"]
                # Use the LLM's mpc_dt (from feedback_data or pipeline state)
                # so the repair prompt shows the correct dt, not the base config.
                repair_dt = (
                    feedback_data.get("mpc_dt") if feedback_data else None
                ) or getattr(self, "_last_mpc_dt", None)
                prompt = create_repair_prompt(
                    command, failed_code, error_msg, attempt, mpc_dt=repair_dt
                )
                response, _ = generate_constraints(system_prompt, prompt)

            # Extract code from response with improved extraction
            mpc_config_code = extract_raw_code(response)

            if not mpc_config_code.strip():
                attempts.append(
                    {
                        "attempt": attempt,
                        "code": mpc_config_code,
                        "error": "No code extracted from LLM response - check response format",
                        "success": False,
                        "failure_stage": "no_code_extracted",
                    }
                )
                continue

            # Create fresh LLM MPC instance for this attempt
            task_mpc = LLMTaskMPC(self.kindyn_model, use_slack=self.use_slack)

            # Initialize with previous iteration's slack weights
            if self.current_slack_weights:
                task_mpc.slack_weights = self.current_slack_weights.copy()

            # Test the MPC configuration code with SafeExecutor
            success, error_msg = self.safe_executor.execute_mpc_configuration_code(
                mpc_config_code, task_mpc
            )

            if not success:
                # Log detailed failure reason
                attempts.append(
                    {
                        "attempt": attempt,
                        "code": mpc_config_code,
                        "error": error_msg,
                        "success": False,
                        "failure_stage": "mpc_configuration",
                    }
                )
                continue

            # Store the configured MPC for later use
            self.current_task_mpc = task_mpc

            # Get configuration summary for logging
            config_summary = task_mpc.get_configuration_summary()
            task_name = config_summary["task_name"]

            # Save the LLM's slack weights for next iteration
            if task_mpc.slack_weights:
                self.current_slack_weights = task_mpc.slack_weights.copy()

            # Success!
            attempts.append(
                {
                    "attempt": attempt,
                    "code": mpc_config_code,
                    "error": "",
                    "success": True,
                    "task_name": task_name,
                    "failure_stage": "none",
                    "config_summary": config_summary,
                    "slack_weights": task_mpc.slack_weights.copy()
                    if task_mpc.slack_weights
                    else {},
                }
            )

            logger.info(f"Constraints generated (attempt {attempt})")
            return mpc_config_code, task_name, attempts, feedback_context

        except Exception as e:
            # Catch any unexpected errors and provide details
            import traceback

            error_details = (
                f"Unexpected error: {str(e)}\nTraceback: {traceback.format_exc()}"
            )

            attempts.append(
                {
                    "attempt": attempt,
                    "code": mpc_config_code,
                    "error": error_details,
                    "success": False,
                    "failure_stage": "unexpected_exception",
                }
            )
            logger.error(f"Attempt {attempt} failed: {str(e)[:100]}")
            continue

    # All attempts failed
    logger.error(f"All {max_attempts} constraint attempts failed")

    # Analyze failure patterns
    failure_stages = [attempt["failure_stage"] for attempt in attempts]

    last_error = attempts[-1]["error"]
    raise ValueError(
        f"Failed to generate valid constraints after {max_attempts} attempts.\n"
        f"Last error: {last_error}\n"
        f"Common failure stages: {set(failure_stages)}"
    )
