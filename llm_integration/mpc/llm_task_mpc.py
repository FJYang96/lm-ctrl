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

import ast
import logging
from collections.abc import Callable
from typing import Any

import numpy as np

import go2_config
from mpc.dynamics.model import KinoDynamic_Model
from mpc.mpc_opti_slack import PHYSICS_CONSTRAINT_NAMES, QuadrupedMPCOptiSlack

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


def _strip_imports(tree: ast.AST) -> None:
    """Remove all Import/ImportFrom nodes at any nesting level in-place."""
    import ast as _ast

    for node in _ast.walk(tree):
        # Only containers with a 'body' list can hold import statements
        for field_name in ("body", "orelse", "finalbody", "handlers"):
            stmts = getattr(node, field_name, None)
            if isinstance(stmts, list):
                stmts[:] = [
                    s
                    for s in stmts
                    if not isinstance(s, (_ast.Import, _ast.ImportFrom))
                ]


class _LLMTaskMPCProxy:
    """Proxy that exposes only the LLM-callable API of LLMTaskMPC.

    Prevents LLM-generated code from traversing internal attributes like
    base_config, kindyn_model, or the underlying solver instance.
    """

    _ALLOWED = frozenset(
        {
            "set_task_name",
            "set_duration",
            "set_time_step",
            "set_contact_sequence",
            "add_constraint",
            "set_reference_trajectory",
            "set_slack_weights",
            "_create_phase_sequence",
            "_create_contact_sequence",
        }
    )

    def __init__(self, real: LLMTaskMPC) -> None:
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name: str) -> Any:
        if name not in self._ALLOWED:
            raise AttributeError(f"Access to mpc.{name} is not allowed in LLM code")
        return getattr(object.__getattribute__(self, "_real"), name)


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
        use_slack: bool = True,
    ):
        """
        Initialize LLM Task MPC.

        Args:
            kindyn_model: Robot kinodynamic model
            use_slack: Whether to use slack formulation for robust optimization
        """
        self.kindyn_model = kindyn_model
        self.use_slack = use_slack

        # LLM-configurable parameters
        self.task_name = "unknown"
        self.contact_sequence: np.ndarray | None = None
        self.constraint_functions: list[Any] = []
        self.mpc_duration = go2_config.mpc_config.duration
        self.mpc_dt = go2_config.mpc_config.mpc_dt
        self.phases: dict[str, dict[str, float | list[int]]] = {}

        # Slack formulation settings
        self.slack_weights: dict[str, float] = {}
        self.last_hardness_report: dict[str, dict[str, Any]] | None = None

        # Reference trajectory support
        self.ref_trajectory_func: Callable[..., Any] | None = None
        self.ref_trajectory_data: dict[str, np.ndarray] | None = None

        # MPC instance (created when configured)
        self.mpc: QuadrupedMPCOptiSlack | None = None
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

            # Validate code safety (AST-level check for dunder access, dangerous calls, etc.)
            is_safe, safety_error = safe_executor.validate_code_safety(llm_config_code)
            if not is_safe:
                return False, f"Code validation failed: {safety_error}"

            # Extract imports from LLM code for dynamic processing, then strip
            # them from the code.  process_dynamic_imports pre-populates the
            # namespace so the import statements are no longer needed at exec
            # time.  This lets us omit __import__ from __builtins__.
            imports_needed = safe_executor.extract_imports_from_code(llm_config_code)
            exec_globals = safe_executor._create_restricted_globals(imports_needed)

            # Strip ALL import statements from code (including those nested
            # inside function bodies) — symbols already in exec_globals
            import ast as _ast

            _tree = _ast.parse(llm_config_code)
            _strip_imports(_tree)
            _ast.fix_missing_locations(_tree)
            exec_code = _ast.unparse(_tree)

            # Add MPC-specific objects to the execution environment
            exec_globals["mpc"] = _LLMTaskMPCProxy(self)
            exec_globals["create_contact_sequence"] = self._create_contact_sequence
            exec_globals["create_phase_sequence"] = self._create_phase_sequence

            # Execute LLM configuration code (imports already resolved)
            exec(exec_code, exec_globals)

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
        # Guard: rename LLM constraints that collide with physics constraint names,
        # otherwise they would be treated as hard (no slack) by the solver.
        func_name = getattr(constraint_func, "__name__", "")
        if func_name in PHYSICS_CONSTRAINT_NAMES:
            safe_name = f"llm_{func_name}"
            logger.warning(
                f"LLM constraint '{func_name}' collides with a physics constraint "
                f"name — renaming to '{safe_name}' to preserve soft enforcement."
            )
            # Wrap instead of mutating the original function object in-place
            import functools

            renamed = functools.wraps(constraint_func)(
                lambda *a, _f=constraint_func, **kw: _f(*a, **kw)
            )
            renamed.__name__ = safe_name
            constraint_func = renamed

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
            weights: Dict mapping YOUR constraint function names to penalty weights.
                     e.g. {"height_constraint": 1e4, "velocity_constraint": 1e2}
                     Physics constraints are always hard and cannot be softened.
        """
        self.slack_weights.update(weights)

    def set_reference_trajectory(self, func: Callable[..., Any]) -> None:
        """
        Set a function that generates per-timestep reference trajectories.

        The reference trajectory serves a dual role:
        1. **Cost target**: The solver's cost function penalizes deviation from
           X_ref/U_ref at every timestep. This is what steers the optimized
           trajectory toward the desired motion.
        2. **Initial guess**: The solver uses X_ref/U_ref as its warm-start
           initial iterate, improving convergence speed.

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
            _task_config = create_task_config(self)

            # Build MPC with slack formulation — always required for LLM tasks
            if not self.use_slack:
                raise ValueError(
                    "Slack formulation is required for LLM tasks. "
                    "Cannot build MPC with use_slack=False."
                )
            self.mpc = QuadrupedMPCOptiSlack(
                model=self.kindyn_model,
                build=True,
                use_slack=True,
                slack_weights=self.slack_weights if self.slack_weights else None,
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

        # Execute reference trajectory function (required — validated in configure_from_llm)
        assert self.ref_trajectory_func is not None
        horizon = self.mpc.horizon
        robot_mass = go2_config.robot_data.mass

        X_ref, U_ref = self.ref_trajectory_func(
            initial_state, horizon, self.contact_sequence, self.mpc_dt, robot_mass
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

        try:
            result = self.mpc.solve_trajectory(
                initial_state,
                ref,
                self.contact_sequence,
                ref_trajectory=ref_traj_dict,
            )

            # Extract solver stats
            stats = self.mpc.opti.stats()
            self.solver_iterations = stats["iter_count"]
            if result[3] != 0:  # Non-zero status means failure
                self.last_error = stats["return_status"]

            # Analyze constraint hardness if using slack formulation
            if self.use_slack and isinstance(self.mpc, QuadrupedMPCOptiSlack):
                self.last_hardness_report = self.mpc.analyze_constraint_hardness()

            return result

        except Exception as e:
            self.last_error = str(e)
            # Still try to get hardness report on failure
            if self.use_slack and isinstance(self.mpc, QuadrupedMPCOptiSlack):
                self.last_hardness_report = self.mpc.analyze_constraint_hardness()
            raise

    def get_configuration_summary(self) -> dict[str, Any]:
        """Get a summary of the current LLM configuration."""
        assert self.contact_sequence is not None
        horizon = self.contact_sequence.shape[1]
        contact_phases = self._analyze_contact_phases()

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
            X_debug,
            U_debug,
        )

    def _analyze_contact_phases(self) -> list[dict[str, Any]]:
        """Analyze the contact sequence to identify distinct phases."""
        return analyze_contact_phases(self.contact_sequence, self.mpc_dt)
