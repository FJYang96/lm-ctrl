"""Constraint generation with LLM retry logic."""

from typing import TYPE_CHECKING, Any

from ..logging_config import logger

if TYPE_CHECKING:
    from .feedback_pipeline import FeedbackPipeline


def generate_constraints_with_retry(
    self: "FeedbackPipeline",
    system_prompt: str,
    user_message: str,
    context: str | None = None,
    command: str = "",
    max_attempts: int = 10,
    images: list[str] | None = None,
) -> tuple[str, str, list[dict[str, Any]]]:
    """
    Generate constraints using the LLM with comprehensive auto-retry on failures.

    This implements the full repair loop with detailed error feedback to the LLM.
    Uses vision API when images are provided for enhanced feedback.

    Returns:
        Tuple of (final_code, function_name, attempt_log)
    """
    from ..mpc import LLMTaskMPC

    attempts: list[dict[str, Any]] = []
    mpc_config_code = ""  # Initialize for exception handling

    for attempt in range(1, max_attempts + 1):
        try:
            # Generate code from LLM
            if attempt == 1:
                # First attempt uses original prompt
                prompt = user_message
                if context:
                    prompt = f"{context}\n\n{user_message}"
            else:
                # Subsequent attempts use repair prompts with detailed error feedback
                failed_code = attempts[-1]["code"]
                error_msg = attempts[-1]["error"]
                prompt = self.constraint_generator.create_repair_prompt(
                    command, failed_code, error_msg, attempt
                )

            # Call LLM with vision if images available, otherwise standard call
            if images and len(images) > 0:
                response = self.llm_client.generate_constraints_with_vision(
                    system_prompt, prompt, None, images
                )
            else:
                response = self.llm_client.generate_constraints(
                    system_prompt, prompt, None
                )

            # Extract code from response with improved extraction
            mpc_config_code = self.llm_client.extract_raw_code(response)

            if not mpc_config_code.strip():
                attempts.append(
                    {
                        "attempt": attempt,
                        "code": mpc_config_code,
                        "error": "No code extracted from LLM response - check response format",
                        "success": False,
                    }
                )
                continue

            # Create fresh LLM MPC instance for this attempt
            use_slack = getattr(self, "use_slack", True)
            task_mpc = LLMTaskMPC(self.kindyn_model, self.config, use_slack=use_slack)

            # Initialize with previous iteration's slack weights (if any)
            if hasattr(self, "current_slack_weights") and self.current_slack_weights:
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
            self.llm_mpc_code = mpc_config_code

            # Get configuration summary for logging
            config_summary = task_mpc.get_configuration_summary()
            task_name = config_summary.get("task_name", "unknown")

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
            return mpc_config_code, task_name, attempts

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
    failure_stages = [attempt.get("failure_stage", "unknown") for attempt in attempts]
    common_errors: dict[str, int] = {}
    for attempt_dict in attempts:
        error = attempt_dict.get("error", "")[:100]
        common_errors[error] = common_errors.get(error, 0) + 1

    last_error = attempts[-1]["error"] if attempts else "No attempts recorded"
    raise ValueError(
        f"Failed to generate valid constraints after {max_attempts} attempts.\n"
        f"Last error: {last_error}\n"
        f"Common failure stages: {set(failure_stages)}"
    )
