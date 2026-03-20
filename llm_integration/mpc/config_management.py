"""Configuration management for LLM Task MPC."""

from typing import TYPE_CHECKING, Any

import go2_config

if TYPE_CHECKING:
    from .llm_task_mpc import LLMTaskMPC


def create_task_config(mpc: "LLMTaskMPC") -> Any:
    """Create a config object tailored for this specific task.

    Note: We store original values and restore them after MPC build to prevent
    config pollution between iterations.

    Args:
        mpc: The LLMTaskMPC instance

    Returns:
        The modified go2_config (temporarily mutated)
    """
    # Store original values to restore after MPC build
    mpc._original_duration = go2_config.mpc_config.duration
    mpc._original_dt = go2_config.mpc_config.mpc_dt
    mpc._original_contact_sequence = go2_config.mpc_config._contact_sequence
    # Make a copy of the original constraints list to prevent accumulation
    mpc._original_constraints = list(go2_config.mpc_config.path_constraints)

    # Temporarily override with LLM-specified parameters
    go2_config.mpc_config.duration = mpc.mpc_duration
    go2_config.mpc_config.mpc_dt = mpc.mpc_dt
    go2_config.mpc_config._contact_sequence = mpc.contact_sequence

    # Add LLM constraints to path constraints (using copy of original)
    go2_config.mpc_config.path_constraints = (
        list(mpc._original_constraints) + mpc.constraint_functions
    )

    return go2_config


def restore_base_config(mpc: "LLMTaskMPC") -> None:
    """Restore go2_config to original values after MPC build.

    Args:
        mpc: The LLMTaskMPC instance
    """
    if mpc._original_duration is None:
        return
    go2_config.mpc_config.path_constraints = list(mpc._original_constraints)
    go2_config.mpc_config.duration = mpc._original_duration
    go2_config.mpc_config.mpc_dt = mpc._original_dt
    go2_config.mpc_config._contact_sequence = mpc._original_contact_sequence
