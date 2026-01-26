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
    mpc._original_contact_sequence = getattr(
        task_config.mpc_config, "_contact_sequence", None
    )
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

    # Ensure path constraint parameters exist (critical for optimization feasibility)
    if not hasattr(task_config.mpc_config, "path_constraint_params"):
        task_config.mpc_config.path_constraint_params = {
            "COMPLEMENTARITY_EPS": 1e-3,
            "SWING_GRF_EPS": 0.0,
            "STANCE_HEIGHT_EPS": 0.04,
            "NO_SLIP_EPS": 0.01,
            "BODY_CLEARANCE_MIN": 0.02,
        }

    return task_config


def restore_base_config(mpc: "LLMTaskMPC") -> None:
    """Restore base config to original values after MPC build.

    Args:
        mpc: The LLMTaskMPC instance
    """
    if hasattr(mpc, "_original_constraints"):
        mpc.base_config.mpc_config.path_constraints = mpc._original_constraints
    if hasattr(mpc, "_original_duration"):
        mpc.base_config.mpc_config.duration = mpc._original_duration
    if hasattr(mpc, "_original_dt"):
        mpc.base_config.mpc_config.mpc_dt = mpc._original_dt
    if hasattr(mpc, "_original_contact_sequence"):
        mpc.base_config.mpc_config._contact_sequence = mpc._original_contact_sequence
