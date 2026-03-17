"""Configuration management for LLM Task MPC."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .llm_task_mpc import LLMTaskMPC


def create_task_config(mpc: "LLMTaskMPC") -> Any:
    """Create a config object tailored for this specific task.

    Note: We store original values and restore them after MPC build to prevent
    config pollution between iterations.

    Args:
        mpc: The LLMTaskMPC instance

    Returns:
        The modified task config
    """
    task_config = mpc.base_config

    # Store original values to restore after MPC build
    mpc._original_duration = task_config.mpc_config.duration
    mpc._original_dt = task_config.mpc_config.mpc_dt
    mpc._original_contact_sequence = task_config.mpc_config._contact_sequence
    # Make a copy of the original constraints list to prevent accumulation
    mpc._original_constraints = list(task_config.mpc_config.path_constraints)

    # Temporarily override with LLM-specified parameters
    task_config.mpc_config.duration = mpc.mpc_duration
    task_config.mpc_config.mpc_dt = mpc.mpc_dt
    task_config.mpc_config._contact_sequence = mpc.contact_sequence

    # Add LLM constraints to path constraints (using copy of original)
    task_config.mpc_config.path_constraints = (
        list(mpc._original_constraints) + mpc.constraint_functions
    )

    return task_config


def restore_base_config(mpc: "LLMTaskMPC") -> None:
    """Restore base config to original values after MPC build.

    Args:
        mpc: The LLMTaskMPC instance
    """
    mpc.base_config.mpc_config.path_constraints = mpc._original_constraints
    mpc.base_config.mpc_config.duration = mpc._original_duration
    mpc.base_config.mpc_config.mpc_dt = mpc._original_dt
    mpc.base_config.mpc_config._contact_sequence = mpc._original_contact_sequence
