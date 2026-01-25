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

from collections.abc import Callable
from typing import Any, cast

import numpy as np

from mpc.dynamics.model import KinoDynamic_Model
from mpc.mpc_opti import QuadrupedMPCOpti


class LLMTaskMPC:
    """
    LLM-specific MPC that can be completely configured by LLM-generated code.

    This allows the LLM to specify contact sequences, constraints, and parameters
    tailored to each specific robot behavior (jump, turn, squat, etc.).
    """

    def __init__(self, kindyn_model: KinoDynamic_Model, base_config: Any):
        """
        Initialize LLM Task MPC.

        Args:
            kindyn_model: Robot kinodynamic model
            base_config: Base configuration (will be modified by LLM)
        """
        self.kindyn_model = kindyn_model
        self.base_config = base_config

        # LLM-configurable parameters
        self.task_name = "unknown"
        self.contact_sequence: np.ndarray | None = None
        self.constraint_functions: list[Any] = []
        self.mpc_duration = 1.0
        self.mpc_dt = 0.02
        self.phases: dict[str, dict[str, float | list[int]]] = {}

        # MPC instance (created when configured)
        self.mpc: QuadrupedMPCOpti | None = None
        self.is_configured = False

        # Solver info for feedback (populated after solve)
        self.solver_iterations: int | None = None
        self.last_error: str | None = None
        self.infeasibility_info: str | None = None

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
            from .safe_executor import SafeConstraintExecutor

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
        wrapped_constraint = self._wrap_constraint_for_contact_phases(constraint_func)
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

    def _create_contact_sequence(
        self, total_duration: float, dt: float, phases: dict[str, dict[str, Any]]
    ) -> np.ndarray:
        """
        Helper function to create contact sequence from phase descriptions.

        Args:
            total_duration: Total trajectory duration
            dt: Time step
            phases: Dictionary of phases with timing and contact patterns

        Returns:
            Contact sequence array (4 x N) where N = total_duration/dt
        """
        horizon = int(total_duration / dt)
        contact_sequence = np.ones((4, horizon))  # Default: all feet in contact

        for _, phase_data in phases.items():
            start_step = int(phase_data["start_time"] / dt)
            duration_steps = int(phase_data["duration"] / dt)
            end_step = min(start_step + duration_steps, horizon)

            for foot in range(4):
                contact_sequence[foot, start_step:end_step] = phase_data[
                    "contact_pattern"
                ][foot]

        return contact_sequence

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
        horizon = int(self.mpc_duration / self.mpc_dt)
        contact_sequence = np.ones((4, horizon))

        current_step = 0
        for _, duration, contact_pattern in phase_list:
            duration_steps = int(duration / self.mpc_dt)
            end_step = min(current_step + duration_steps, horizon)

            for foot in range(4):
                contact_sequence[foot, current_step:end_step] = contact_pattern[foot]

            current_step = end_step
            if current_step >= horizon:
                break

        return contact_sequence

    def _build_mpc(self) -> tuple[bool, str]:
        """
        Build the MPC instance with LLM-specified configuration.

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Create modified config for this task
            task_config = self._create_task_config()

            # Build MPC with task configuration
            self.mpc = QuadrupedMPCOpti(
                model=self.kindyn_model, config=task_config, build=True
            )

            # Restore base config to prevent pollution between iterations
            self._restore_base_config()

            return True, ""

        except Exception as e:
            # Restore base config even on failure
            self._restore_base_config()
            return False, str(e)

    def _create_task_config(self) -> Any:
        """Create a config object tailored for this specific task.

        Note: We store original values and restore them after MPC build to prevent
        config pollution between iterations.
        """
        task_config = self.base_config

        # Store original values to restore after MPC build
        self._original_duration = task_config.mpc_config.duration
        self._original_dt = task_config.mpc_config.mpc_dt
        self._original_contact_sequence = getattr(
            task_config.mpc_config, "_contact_sequence", None
        )
        # Make a copy of the original constraints list to prevent accumulation
        self._original_constraints = list(task_config.mpc_config.path_constraints)

        # Temporarily override with LLM-specified parameters
        task_config.mpc_config.duration = self.mpc_duration
        task_config.mpc_config.mpc_dt = self.mpc_dt
        task_config.mpc_config._contact_sequence = self.contact_sequence

        # Add LLM constraints to path constraints (using copy of original)
        task_config.mpc_config.path_constraints = (
            list(self._original_constraints) + self.constraint_functions
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

    def _restore_base_config(self) -> None:
        """Restore base config to original values after MPC build."""
        if hasattr(self, "_original_constraints"):
            self.base_config.mpc_config.path_constraints = self._original_constraints
        if hasattr(self, "_original_duration"):
            self.base_config.mpc_config.duration = self._original_duration
        if hasattr(self, "_original_dt"):
            self.base_config.mpc_config.mpc_dt = self._original_dt
        if hasattr(self, "_original_contact_sequence"):
            self.base_config.mpc_config._contact_sequence = (
                self._original_contact_sequence
            )

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

        # Reset solver info before solving
        self.solver_iterations = None
        self.last_error = None
        self.infeasibility_info = None

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

            return result

        except Exception as e:
            self.last_error = str(e)
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
        violations: dict[str, Any] = {
            "llm_constraints": [],
            "by_constraint": {},
            "summary": [],
        }

        if not self.constraint_functions:
            violations["summary"].append("No LLM constraints to evaluate")
            return violations

        if self.contact_sequence is None:
            violations["summary"].append("No contact sequence configured")
            return violations

        horizon = self.contact_sequence.shape[1]

        # Track violations per constraint
        for i in range(len(self.constraint_functions)):
            violations["by_constraint"][i] = []

        # Evaluate each constraint at each timestep (skip k=0 as MPC does)
        for k in range(1, min(horizon, X_debug.shape[1])):
            x_k = X_debug[:, k]
            u_k = U_debug[:, k] if k < U_debug.shape[1] else U_debug[:, -1]
            contact_k = self.contact_sequence[:, k]

            for i, constraint_func in enumerate(self.constraint_functions):
                try:
                    # Call the constraint function
                    result = constraint_func(
                        x_k,
                        u_k,
                        self.kindyn_model,
                        self.base_config,
                        contact_k,
                        k,
                        horizon,
                    )

                    # Extract value, lower, upper bounds
                    if isinstance(result, tuple) and len(result) == 3:
                        expr_value, lower, upper = result

                        # Convert CasADi types to float if needed
                        try:
                            if hasattr(expr_value, "full"):
                                expr_value = float(expr_value.full().flatten()[0])
                            elif hasattr(expr_value, "__float__"):
                                expr_value = float(expr_value)
                        except Exception:
                            pass  # Keep as-is if conversion fails

                        try:
                            if hasattr(lower, "__float__"):
                                lower = float(lower)
                            if hasattr(upper, "__float__"):
                                upper = float(upper)
                        except Exception:
                            pass

                        # Check for violations
                        if isinstance(expr_value, (int, float)) and isinstance(
                            lower, (int, float)
                        ):
                            if expr_value < lower:
                                violation_msg = (
                                    f"Constraint {i} at k={k}: "
                                    f"value={expr_value:.4f} < lower={lower:.4f}"
                                )
                                violations["llm_constraints"].append(violation_msg)
                                violations["by_constraint"][i].append(
                                    {
                                        "k": k,
                                        "value": expr_value,
                                        "lower": lower,
                                        "type": "below_lower",
                                    }
                                )

                        if isinstance(expr_value, (int, float)) and isinstance(
                            upper, (int, float)
                        ):
                            if expr_value > upper:
                                violation_msg = (
                                    f"Constraint {i} at k={k}: "
                                    f"value={expr_value:.4f} > upper={upper:.4f}"
                                )
                                violations["llm_constraints"].append(violation_msg)
                                violations["by_constraint"][i].append(
                                    {
                                        "k": k,
                                        "value": expr_value,
                                        "upper": upper,
                                        "type": "above_upper",
                                    }
                                )

                except Exception as e:
                    # Record evaluation error but continue
                    violations["llm_constraints"].append(
                        f"Constraint {i} at k={k}: evaluation error - {str(e)[:50]}"
                    )

        # Generate summary
        for i, constraint_violations in violations["by_constraint"].items():
            if constraint_violations:
                violations["summary"].append(
                    f"LLM constraint {i}: violated at {len(constraint_violations)} timesteps"
                )

        if not violations["summary"]:
            violations["summary"].append(
                "No LLM constraint violations detected in evaluated trajectory"
            )

        return violations

    def _analyze_contact_phases(self) -> list[dict[str, Any]]:
        """Analyze the contact sequence to identify distinct phases."""
        if self.contact_sequence is None:
            return []

        phases = []
        current_pattern = None
        phase_start = 0

        for step in range(self.contact_sequence.shape[1]):
            pattern = tuple(self.contact_sequence[:, step])

            if pattern != current_pattern:
                if current_pattern is not None:
                    phases.append(
                        {
                            "start_time": phase_start * self.mpc_dt,
                            "duration": (step - phase_start) * self.mpc_dt,
                            "contact_pattern": list(current_pattern),
                            "phase_type": self._classify_contact_pattern(
                                current_pattern
                            ),
                        }
                    )
                current_pattern = pattern
                phase_start = step

        # Add final phase
        if current_pattern is not None:
            phases.append(
                {
                    "start_time": phase_start * self.mpc_dt,
                    "duration": (self.contact_sequence.shape[1] - phase_start)
                    * self.mpc_dt,
                    "contact_pattern": list(current_pattern),
                    "phase_type": self._classify_contact_pattern(current_pattern),
                }
            )

        return phases

    def _classify_contact_pattern(self, pattern: tuple[float, ...]) -> str:
        """Classify a contact pattern into a phase type."""
        num_contacts = sum(pattern)

        if num_contacts == 4:
            return "stance"
        elif num_contacts == 0:
            return "flight"
        elif num_contacts == 2:
            if pattern[0] == pattern[1] and pattern[2] == pattern[3]:
                return "trot" if pattern[0] != pattern[2] else "pace"
            else:
                return "diagonal"
        elif num_contacts == 3:
            return "tripod"
        elif num_contacts == 1:
            return "single_foot"
        else:
            return "unknown"

    def _wrap_constraint_for_contact_phases(
        self, constraint_func: Callable[..., Any]
    ) -> Callable[..., Any]:
        """
        Wrap LLM-generated constraints to be contact-aware.

        Supports both 5-argument and 7-argument constraint signatures:
        - 5 args: (x_k, u_k, kindyn_model, config, contact_k)
        - 7 args: (x_k, u_k, kindyn_model, config, contact_k, k, horizon)

        The LLM is responsible for generating constraints that are appropriate
        for the task's contact sequence.
        """
        import inspect

        # Detect the number of parameters in the constraint function
        try:
            sig = inspect.signature(constraint_func)
            num_params = len(sig.parameters)
        except (ValueError, TypeError):
            num_params = 5  # Default to 5-arg signature

        def contact_aware_constraint(
            x_k: Any,
            u_k: Any,
            kindyn_model: Any,
            config: Any,
            contact_k: Any,
            k: int = 0,
            horizon: int = 1,
        ) -> tuple[Any, Any, Any]:
            # Call with appropriate number of arguments based on function signature
            if num_params >= 7:
                return cast(
                    tuple[Any, Any, Any],
                    constraint_func(
                        x_k, u_k, kindyn_model, config, contact_k, k, horizon
                    ),
                )
            else:
                return cast(
                    tuple[Any, Any, Any],
                    constraint_func(x_k, u_k, kindyn_model, config, contact_k),
                )

        return contact_aware_constraint
