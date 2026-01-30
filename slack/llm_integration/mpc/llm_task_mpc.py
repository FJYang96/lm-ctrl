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

Slack Formulation Support:
When use_slack=True, the MPC uses slack variables to:
- Make optimization more robust (avoid infeasibility)
- Analyze constraint "hardness" to identify overly restrictive constraints
- Provide quantitative feedback for LLM to improve constraints
"""

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
    classify_contact_pattern,
    create_contact_sequence,
    create_phase_sequence,
)


class LLMTaskMPC:
    """
    LLM-specific MPC that can be completely configured by LLM-generated code.

    This allows the LLM to specify contact sequences, constraints, and parameters
    tailored to each specific robot behavior (jump, turn, squat, etc.).
    """

    # Assign imported functions directly as methods (no wrapper needed)
    _create_contact_sequence = staticmethod(create_contact_sequence)
    _classify_contact_pattern = staticmethod(classify_contact_pattern)
    _wrap_constraint_for_contact_phases = staticmethod(
        wrap_constraint_for_contact_phases
    )

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
            use_slack: Whether to use slack formulation for constraint analysis
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
        self.slack_weights: dict[str, float] = {}  # LLM-configurable slack weights

        # MPC instance (created when configured)
        self.mpc: QuadrupedMPCOpti | QuadrupedMPCOptiSlack | None = None
        self.is_configured = False

        # Solver info for feedback (populated after solve)
        self.solver_iterations: int | None = None
        self.last_error: str | None = None
        self.infeasibility_info: str | None = None

        # Slack analysis (populated after solve when use_slack=True)
        self.last_hardness_report: dict[str, dict] | None = None

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

            # Auto-fix contact_sequence dimension if LLM set duration after contact_sequence
            expected_horizon = int(self.mpc_duration / self.mpc_dt)
            current_horizon = self.contact_sequence.shape[1]

            if current_horizon != expected_horizon:
                print(f"⚠️ [CONFIG] contact_sequence dimension mismatch detected: "
                      f"got {current_horizon}, expected {expected_horizon} "
                      f"(duration={self.mpc_duration}s, dt={self.mpc_dt}s)")

                if current_horizon < expected_horizon:
                    # Extend: repeat the last column
                    padding_needed = expected_horizon - current_horizon
                    last_col = self.contact_sequence[:, -1:].repeat(padding_needed, axis=1)
                    self.contact_sequence = np.hstack([self.contact_sequence, last_col])
                    print(f"✅ [CONFIG] Extended contact_sequence by {padding_needed} timesteps")
                else:
                    # Truncate
                    self.contact_sequence = self.contact_sequence[:, :expected_horizon]
                    print(f"✅ [CONFIG] Truncated contact_sequence to {expected_horizon} timesteps")

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

        Lower weights = constraints are easier to violate (faster convergence)
        Higher weights = constraints are harder to violate (stricter)

        Available constraint names:
            - "contact_aware_constraint": Your LLM-generated constraints
            - "friction_cone_constraints": Ground friction limits
            - "foot_height_constraints": Feet above ground
            - "body_clearance_constraints": Body above ground
            - "complementarity_constraints": Contact force/velocity consistency
            - "joint_limits_constraints": Joint angle limits
            - "input_limits_constraints": Actuator limits

        Example:
            mpc.set_slack_weights({
                "contact_aware_constraint": 1e4,  # Make your constraints stricter
                "body_clearance_constraints": 1e1,  # Allow body tilt during flip
                "complementarity_constraints": 1e0,  # Very relaxed (hard to satisfy)
            })

        Args:
            weights: Dictionary mapping constraint names to penalty weights
        """
        self.slack_weights.update(weights)

    def add_phase(
        self, name: str, start_time: float, duration: float, contact_pattern: list[int]
    ) -> None:
        """
        Add a motion phase (e.g., stance, flight, landing).

        Args:
            name: Phase name (e.g., "stance", "flight", "landing")
            start_time: Start time of phase (seconds)
            duration: Duration of phase (seconds)
            contact_pattern: Contact state for each foot [FL, FR, RL, RR] (1=contact, 0=flight)
        """
        self.phases[name] = {
            "start_time": start_time,
            "duration": duration,
            "contact_pattern": contact_pattern,
        }

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

        Uses QuadrupedMPCOptiSlack when use_slack=True for constraint hardness analysis.

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Create modified config for this task
            task_config = create_task_config(self)

            # Debug: print constraint info
            num_path_constraints = len(task_config.mpc_config.path_constraints)
            num_llm_constraints = len(self.constraint_functions)
            print(f"[BUILD MPC] use_slack={self.use_slack}")
            print(f"[BUILD MPC] Total path_constraints: {num_path_constraints}")
            print(f"[BUILD MPC] LLM constraint_functions: {num_llm_constraints}")

            # Build MPC with task configuration
            if self.use_slack:
                print("[BUILD MPC] Creating QuadrupedMPCOptiSlack...")
                # Pass LLM-configured slack weights if any
                if self.slack_weights:
                    print("[LLM SLACK WEIGHTS] LLM customized the following weights:")
                    for name, weight in self.slack_weights.items():
                        print(f"    {name}: {weight:.0e}")
                else:
                    print("[BUILD MPC] Using default slack weights (LLM did not customize)")
                self.mpc = QuadrupedMPCOptiSlack(
                    model=self.kindyn_model,
                    config=task_config,
                    build=True,
                    use_slack=True,
                    slack_weights=self.slack_weights if self.slack_weights else None,
                )
                # Print final merged weights
                print("[FINAL SLACK WEIGHTS] After merging with defaults:")
                for name, weight in self.mpc.slack_weights.items():
                    marker = "← LLM" if self.slack_weights and name in self.slack_weights else ""
                    print(f" {name}: {weight:.0e} {marker}")
                print(f"[BUILD MPC] Slack MPC created with {len(self.mpc.slack_variables)} slack variable groups")
            else:
                print("[BUILD MPC] Creating QuadrupedMPCOpti...")
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
        self, initial_state: np.ndarray, ref: np.ndarray
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

        # Calculate expected horizon from duration and dt
        expected_horizon = int(self.mpc_duration / self.mpc_dt)
        current_horizon = self.contact_sequence.shape[1]

        # Auto-adjust contact_sequence if dimension mismatch
        if current_horizon != expected_horizon:
            print(f"[DIMENSION FIX] contact_sequence mismatch: "
                  f"got {current_horizon} timesteps, expected {expected_horizon} "
                  f"(duration={self.mpc_duration}s, dt={self.mpc_dt}s)")

            if current_horizon < expected_horizon:
                # Extend: repeat the last column
                padding_needed = expected_horizon - current_horizon
                last_col = self.contact_sequence[:, -1:].repeat(padding_needed, axis=1)
                self.contact_sequence = np.hstack([self.contact_sequence, last_col])
                print(f"[DIMENSION FIX] Extended contact_sequence by {padding_needed} "
                      f"timesteps (repeating last contact pattern)")
            else:
                # Truncate
                self.contact_sequence = self.contact_sequence[:, :expected_horizon]
                print(f"[DIMENSION FIX] Truncated contact_sequence to {expected_horizon} timesteps")

        # Reset solver info before solving
        self.solver_iterations = None
        self.last_error = None
        self.infeasibility_info = None
        self.last_hardness_report = None

        try:
            result = self.mpc.solve_trajectory(
                initial_state, ref, self.contact_sequence
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
            if self.use_slack and hasattr(self.mpc, "analyze_constraint_hardness"):
                self.last_hardness_report = self.mpc.analyze_constraint_hardness()

            return result

        except Exception as e:
            self.last_error = str(e)
            # Still try to get hardness report even on failure
            if self.use_slack and hasattr(self.mpc, "analyze_constraint_hardness"):
                try:
                    self.last_hardness_report = self.mpc.analyze_constraint_hardness()
                except Exception:
                    pass
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

    def analyze_constraint_hardness(self) -> dict[str, dict]:
        """
        Analyze constraint hardness using slack variable values.

        Returns:
            Dictionary mapping constraint names to hardness metrics:
            {
                "constraint_name": {
                    "total_slack_L1": float,
                    "max_slack_Linf": float,
                    "mean_slack_per_dim": float,
                    "total_dims": int,
                    "active_timesteps": list[int],
                }
            }
        """
        if not self.use_slack:
            return {}

        if self.last_hardness_report is not None:
            return self.last_hardness_report

        if self.mpc is not None and hasattr(self.mpc, "analyze_constraint_hardness"):
            return self.mpc.analyze_constraint_hardness()

        return {}

    def print_constraint_hardness_report(self) -> None:
        """Print a formatted constraint hardness report."""
        if self.mpc is not None and hasattr(self.mpc, "print_constraint_hardness_report"):
            self.mpc.print_constraint_hardness_report()
        else:
            print("No slack analysis available. Use use_slack=True.")

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
