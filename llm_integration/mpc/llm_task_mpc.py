"""
LLM-Specific MPC System - Task-Aware Trajectory Optimization

This is a separate MPC implementation specifically designed for LLM-generated
robot behaviors. Unlike the existing MPC system which assumes jumping motions,
this system allows the LLM to specify:

1. Contact sequences appropriate for each task
2. Task-specific constraints
3. Custom MPC parameters (duration, phases, etc.)

The LLM generates both the MPC configuration and constraints together,
creating a cohesive optimization setup for each unique behavior.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from mpc.dynamics.model import KinoDynamic_Model
from mpc.mpc_opti import QuadrupedMPCOpti
from mpc.mpc_opti_slack import QuadrupedMPCOptiSlack

from .config_management import create_task_config, restore_base_config
from .constraint_wrapper import (
    evaluate_constraint_violations,
    wrap_constraint_for_contact_phases,
)
from .contact_utils import (
    analyze_contact_phases,
    create_contact_sequence,
    create_phase_sequence,
)

logger = logging.getLogger("llm_integration")


class LLMTaskMPC:
    """
    LLM-specific MPC that can be completely configured by LLM-generated code.

    This allows the LLM to specify contact sequences, constraints, and parameters
    tailored to each specific robot behavior (jump, turn, squat, etc.).
    """

    # Assign imported functions directly as methods (no wrapper needed)
    _create_contact_sequence = staticmethod(create_contact_sequence)

    def __init__(
        self,
        kindyn_model: KinoDynamic_Model,
        base_config: Any,
        use_slack: bool = True,
    ):
        """
        Initialize LLM Task MPC.

        Args:
            kindyn_model: Robot kinodynamic model
            base_config: Base configuration (will be modified by LLM)
            use_slack: Whether to use slack formulation for robust optimization
        """
        self.kindyn_model = kindyn_model
        self.base_config = base_config
        self.use_slack = use_slack

        # LLM-configurable parameters
        self.task_name = "unknown"
        self.contact_sequence: np.ndarray | None = None
        self.constraint_functions: list[Any] = []
        self.mpc_duration = 1.0
        self.mpc_dt = 0.02
        self.phases: dict[str, dict[str, float | list[int]]] = {}

        # Slack formulation settings
        self.slack_weights: dict[str, float] = {}
        self.last_hardness_report: dict[str, dict[str, Any]] | None = None

        # Reference trajectory support
        self.ref_trajectory_func: Callable[..., Any] | None = None
        self.ref_trajectory_data: dict[str, np.ndarray] | None = None

        # MPC instance (created when configured)
        self.mpc: QuadrupedMPCOpti | QuadrupedMPCOptiSlack | None = None
        self.is_configured = False

        # Solver info for feedback (populated after solve)
        self.solver_iterations: int | None = None
        self.last_error: str | None = None
        self.infeasibility_info: str | None = None

        # Original config values (stored during task config creation)
        self._original_duration: float | None = None
        self._original_dt: float | None = None
        self._original_contact_sequence: np.ndarray | None = None
        self._original_constraints: list[Any] = []

    def configure_from_llm(self, llm_config_code: str) -> tuple[bool, str]:
        """
        Configure the MPC using LLM-generated configuration code.

        Args:
            llm_config_code: Python code that configures this MPC instance

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Use SafeExecutor's comprehensive globals for consistent execution environment
            from ..executor.safe_executor import SafeConstraintExecutor

            safe_executor = SafeConstraintExecutor()

            # Extract any imports from LLM code for dynamic processing
            imports_needed = safe_executor.extract_imports_from_code(llm_config_code)
            exec_globals = safe_executor._create_restricted_globals(imports_needed)

            # Add MPC-specific objects to the execution environment
            exec_globals["mpc"] = self
            exec_globals["create_contact_sequence"] = self._create_contact_sequence
            exec_globals["create_phase_sequence"] = self._create_phase_sequence

            # Execute LLM configuration code with full CasADi environment + dynamic imports
            exec(llm_config_code, exec_globals)

            # Validate configuration
            if self.contact_sequence is None:
                return (
                    False,
                    "No contact sequence specified. Use mpc.set_contact_sequence()",
                )

            if not self.constraint_functions:
                return False, "No constraints specified. Use mpc.add_constraint()"

            if self.ref_trajectory_func is None:
                return (
                    False,
                    "No reference trajectory specified. Use mpc.set_reference_trajectory(func)",
                )

            # Auto-fix contact_sequence dimensions if needed
            expected_horizon = int(self.mpc_duration / self.mpc_dt)
            if self.contact_sequence.shape[1] != expected_horizon:
                # Resize contact_sequence to match expected horizon
                old_seq = self.contact_sequence
                new_seq = np.zeros((4, expected_horizon))
                copy_len = min(old_seq.shape[1], expected_horizon)
                new_seq[:, :copy_len] = old_seq[:, :copy_len]
                # Repeat last column for any remaining timesteps
                if copy_len < expected_horizon:
                    new_seq[:, copy_len:] = old_seq[:, -1:].repeat(
                        expected_horizon - copy_len, axis=1
                    )
                self.contact_sequence = new_seq

            # Create MPC with LLM configuration
            success, error = self._build_mpc()
            if not success:
                return False, f"MPC build failed: {error}"

            self.is_configured = True
            return True, ""

        except Exception as e:
            return False, f"Configuration execution failed: {str(e)}"

    def set_contact_sequence(self, contact_sequence: np.ndarray) -> None:
        """Set the contact sequence for this task."""
        self.contact_sequence = contact_sequence

    def add_constraint(self, constraint_func: Callable[..., Any]) -> None:
        """Add a constraint function to this task."""
        # Wrap the constraint function to be contact-aware for jumping tasks
        wrapped_constraint = wrap_constraint_for_contact_phases(constraint_func)
        self.constraint_functions.append(wrapped_constraint)

    def set_duration(self, duration: float) -> None:
        """Set the MPC duration."""
        self.mpc_duration = duration

    def set_time_step(self, dt: float) -> None:
        """Set the MPC time step."""
        self.mpc_dt = dt

    def set_task_name(self, name: str) -> None:
        """Set the task name for this MPC."""
        self.task_name = name

    def set_slack_weights(self, weights: dict[str, float]) -> None:
        """
        Set custom slack penalty weights for constraint types.

        Higher weight = harder constraint (solver avoids using slack).
        Lower weight = softer constraint (solver uses slack more freely).

        Args:
            weights: Dict mapping constraint names to penalty weights.
                     e.g. {"friction_cone_constraints": 1e6, "contact_aware_constraint": 1e4}
        """
        self.slack_weights.update(weights)

    def set_reference_trajectory(self, func: Callable[..., Any]) -> None:
        """
        Set a function that generates per-timestep reference trajectories.

        The function will be called with:
            func(initial_state, horizon, contact_sequence, mpc_dt, robot_mass)
        and must return a tuple (X_ref, U_ref) where:
            X_ref: np.ndarray of shape (states_dim, horizon+1) — state reference
            U_ref: np.ndarray of shape (inputs_dim, horizon) — input reference

        Args:
            func: Reference trajectory generator function
        """
        self.ref_trajectory_func = func

    def _create_phase_sequence(
        self, phase_list: list[tuple[str, float, list[int]]]
    ) -> np.ndarray:
        """
        Helper to create contact sequence from a simple phase list.

        Args:
            phase_list: List of (phase_name, duration, [FL,FR,RL,RR]) tuples

        Returns:
            Contact sequence array
        """
        return create_phase_sequence(phase_list, self.mpc_duration, self.mpc_dt)

    def _build_mpc(self) -> tuple[bool, str]:
        """
        Build the MPC instance with LLM-specified configuration.

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Create modified config for this task
            task_config = create_task_config(self)

            # Build MPC with task configuration
            if self.use_slack:
                self.mpc = QuadrupedMPCOptiSlack(
                    model=self.kindyn_model,
                    config=task_config,
                    build=True,
                    use_slack=True,
                    slack_weights=self.slack_weights if self.slack_weights else None,
                )
            else:
                self.mpc = QuadrupedMPCOpti(
                    model=self.kindyn_model, config=task_config, build=True
                )

            # Restore base config to prevent pollution between iterations
            restore_base_config(self)

            return True, ""

        except Exception as e:
            # Restore base config even on failure
            restore_base_config(self)
            return False, str(e)

    def solve_trajectory(
        self,
        initial_state: np.ndarray,
        ref: np.ndarray,
        warmstart: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Solve trajectory optimization with LLM-configured MPC.

        Args:
            initial_state: Initial robot state
            ref: Reference trajectory

        Returns:
            Tuple of (state_traj, grf_traj, joint_vel_traj, status)
        """
        if not self.is_configured or self.mpc is None:
            raise ValueError("MPC not configured. Call configure_from_llm() first.")

        # Use the LLM-specified contact sequence
        if self.contact_sequence is None:
            raise ValueError("Contact sequence not set. Configure MPC first.")

        # Auto-fix contact_sequence column count to match MPC horizon
        if hasattr(self.mpc, "horizon"):
            expected_cols = self.mpc.horizon
            if self.contact_sequence.shape[1] != expected_cols:
                old_seq = self.contact_sequence
                new_seq = np.zeros((4, expected_cols))
                copy_len = min(old_seq.shape[1], expected_cols)
                new_seq[:, :copy_len] = old_seq[:, :copy_len]
                if copy_len < expected_cols:
                    new_seq[:, copy_len:] = old_seq[:, -1:].repeat(
                        expected_cols - copy_len, axis=1
                    )
                self.contact_sequence = new_seq

        # Reset solver info before solving
        self.solver_iterations = None
        self.last_error = None
        self.infeasibility_info = None
        self.last_hardness_report = None
        self.ref_trajectory_data = None

        # Execute reference trajectory function if set
        ref_traj_dict = None
        if self.ref_trajectory_func is not None:
            try:
                horizon = self.mpc.horizon
                mpc_dt = self.mpc_dt
                robot_mass = self.base_config.robot_data.mass

                X_ref, U_ref = self.ref_trajectory_func(
                    initial_state, horizon, self.contact_sequence, mpc_dt, robot_mass
                )

                # Validate shapes
                expected_x_shape = (self.mpc.states_dim, horizon + 1)
                expected_u_shape = (self.mpc.inputs_dim, horizon)
                if X_ref.shape != expected_x_shape:
                    raise ValueError(
                        f"X_ref shape {X_ref.shape} != expected {expected_x_shape}"
                    )
                if U_ref.shape != expected_u_shape:
                    raise ValueError(
                        f"U_ref shape {U_ref.shape} != expected {expected_u_shape}"
                    )

                # Check for NaN/Inf
                if np.any(np.isnan(X_ref)) or np.any(np.isinf(X_ref)):
                    raise ValueError("X_ref contains NaN or Inf values")
                if np.any(np.isnan(U_ref)) or np.any(np.isinf(U_ref)):
                    raise ValueError("U_ref contains NaN or Inf values")

                ref_traj_dict = {"X_ref": X_ref, "U_ref": U_ref}
                self.ref_trajectory_data = ref_traj_dict

                logger.info(
                    f"Reference trajectory generated: "
                    f"height range [{X_ref[2, :].min():.3f}, {X_ref[2, :].max():.3f}]m, "
                    f"pitch range [{X_ref[7, :].min():.2f}, {X_ref[7, :].max():.2f}]rad"
                )
            except Exception as e:
                logger.warning(f"Reference trajectory generation failed: {e}")
                # Fall back to no trajectory reference — solver will use heuristic guess

        try:
            result = self.mpc.solve_trajectory(
                initial_state,
                ref,
                self.contact_sequence,
                warmstart=warmstart,
                ref_trajectory=ref_traj_dict,
            )

            # Try to extract solver stats from the MPC/opti instance
            if hasattr(self.mpc, "opti") and self.mpc.opti is not None:
                try:
                    stats = self.mpc.opti.stats()
                    self.solver_iterations = stats.get("iter_count", None)
                    if result[3] != 0:  # Non-zero status means failure
                        self.last_error = stats.get("return_status", "Unknown error")
                except Exception:
                    pass  # Stats not available

            # Analyze constraint hardness if using slack formulation
            if self.use_slack and isinstance(self.mpc, QuadrupedMPCOptiSlack):
                try:
                    self.last_hardness_report = self.mpc.analyze_constraint_hardness()
                except Exception:
                    self.last_hardness_report = None

            return result

        except Exception as e:
            self.last_error = str(e)
            # Still try to get hardness report on failure
            if self.use_slack and isinstance(self.mpc, QuadrupedMPCOptiSlack):
                try:
                    self.last_hardness_report = self.mpc.analyze_constraint_hardness()
                except Exception:
                    self.last_hardness_report = None
            raise

    def get_configuration_summary(self) -> dict[str, Any]:
        """Get a summary of the current LLM configuration."""
        if self.contact_sequence is not None:
            horizon = self.contact_sequence.shape[1]
            contact_phases = self._analyze_contact_phases()
        else:
            horizon = 0
            contact_phases = []

        return {
            "task_name": self.task_name,
            "duration": self.mpc_duration,
            "time_step": self.mpc_dt,
            "horizon": horizon,
            "num_constraints": len(self.constraint_functions),
            "contact_phases": contact_phases,
            "phases": self.phases,
            "is_configured": self.is_configured,
            "use_slack": self.use_slack,
        }

    def evaluate_constraint_violations(
        self, X_debug: np.ndarray, U_debug: np.ndarray
    ) -> dict[str, Any]:
        """
        Evaluate LLM-generated constraints against a (possibly failed) trajectory.

        This helps diagnose why optimization failed by showing exactly which
        LLM constraints are violated and at which timesteps.

        Args:
            X_debug: State trajectory (states_dim x horizon+1) from opti.debug.value(X)
            U_debug: Input trajectory (inputs_dim x horizon) from opti.debug.value(U)

        Returns:
            Dictionary with constraint violation details
        """
        return evaluate_constraint_violations(
            self.constraint_functions,
            self.contact_sequence,
            self.kindyn_model,
            self.base_config,
            X_debug,
            U_debug,
        )

    def _analyze_contact_phases(self) -> list[dict[str, Any]]:
        """Analyze the contact sequence to identify distinct phases."""
        return analyze_contact_phases(self.contact_sequence, self.mpc_dt)
