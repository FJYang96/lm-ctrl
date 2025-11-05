"""
Iterative refinement loop for LLM-based constraint generation
Implements Algorithm 1 from the research paper
"""

import time
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .code_parser import ConstraintValidator, SafeCodeParser
from .feedback import (
    FeedbackGenerator,
    ResultsLogger,
    TrajectoryAnalyzer,
    TrajectoryMetrics,
)
from .llm_client import LLMClient
from .prompts import SYSTEM_PROMPT


class IterativeRefinementEngine:
    """
    Main engine for iterative constraint refinement using LLM feedback

    Implements the iterative pipeline:
    1. Generate constraints from LLM
    2. Solve trajectory optimization
    3. Analyze results
    4. Provide feedback to LLM
    5. Repeat until success or max iterations
    """

    def __init__(
        self,
        kinodynamic_model: Any,
        config: Any,
        max_iterations: int = 10,
        log_dir: str = "results/llm_iterations",
    ):
        """
        Initialize the iterative refinement engine

        Args:
            kinodynamic_model: Robot dynamics model
            config: Configuration object
            max_iterations: Maximum number of iterations
            log_dir: Directory for logging results
        """
        self.kinodynamic_model = kinodynamic_model
        self.config = config
        self.max_iterations = max_iterations

        # Initialize components
        self.llm_client = LLMClient()
        self.code_parser = SafeCodeParser()
        self.constraint_validator = ConstraintValidator()
        self.trajectory_analyzer = TrajectoryAnalyzer(config)
        self.feedback_generator = FeedbackGenerator()
        self.results_logger = ResultsLogger(log_dir)

        # State tracking
        self.current_iteration = 0
        self.context_history = []
        self.best_result = None
        self.success_achieved = False

    def run_refinement_loop(
        self,
        command: str,
        initial_state: np.ndarray,
        contact_sequence: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Tuple[bool, Optional[Any], Dict[str, Any]]:
        """
        Run the complete iterative refinement loop

        Args:
            command: Natural language command (e.g., "do a backflip")
            initial_state: Robot initial state
            contact_sequence: Contact schedule for optimization
            reference_trajectory: Reference trajectory for tracking

        Returns:
            Tuple of (success, best_constraint_function, summary_results)
        """

        print(f"🚀 Starting iterative refinement for command: '{command}'")
        print(f"📊 Max iterations: {self.max_iterations}")

        # Initialize context with system prompt and command
        context = f"{SYSTEM_PROMPT}\n\n## User Command\n\n{command}\n\nGenerate constraint function for this command:"

        # Track best result across iterations
        best_constraint_function = None
        best_score = -1.0

        for iteration in range(1, self.max_iterations + 1):
            self.current_iteration = iteration

            print(f"\n{'='*50}")
            print(f"🔄 ITERATION {iteration}/{self.max_iterations}")
            print(f"{'='*50}")

            try:
                # Step 1: Generate constraints from LLM
                print("🧠 Generating constraints from LLM...")
                constraints_code = self._generate_constraints(context)

                if constraints_code is None:
                    print("❌ Failed to generate constraints, skipping iteration")
                    continue

                # Step 2: Parse and validate constraints
                print("🔍 Parsing and validating constraint function...")
                constraint_function = self._parse_constraints(constraints_code)

                if constraint_function is None:
                    print("❌ Failed to parse constraints, skipping iteration")
                    continue

                # Step 3: Solve trajectory optimization
                print("⚡ Solving trajectory optimization...")
                optimization_result = self._solve_trajectory_optimization(
                    constraint_function,
                    initial_state,
                    contact_sequence,
                    reference_trajectory,
                )

                if optimization_result is None:
                    print("❌ Trajectory optimization failed, skipping iteration")
                    continue

                state_traj, input_traj, grf_traj, solver_status, solve_time = (
                    optimization_result
                )

                # Step 4: Analyze trajectory results
                print("📈 Analyzing trajectory results...")
                metrics = self.trajectory_analyzer.analyze_trajectory(
                    state_traj,
                    input_traj,
                    grf_traj,
                    contact_sequence,
                    solver_status,
                    solve_time,
                )

                # Step 5: Calculate performance score
                score = self._calculate_performance_score(metrics)
                print(f"📊 Performance score: {score:.3f}")

                # Update best result if this is better
                if score > best_score:
                    best_score = score
                    best_constraint_function = constraint_function
                    self.best_result = {
                        "iteration": iteration,
                        "metrics": metrics,
                        "constraints_code": constraints_code,
                        "state_traj": state_traj,
                        "input_traj": input_traj,
                        "grf_traj": grf_traj,
                    }
                    print(f"✨ New best result! Score: {score:.3f}")

                # Step 6: Check for success
                if metrics.overall_success:
                    print(f"🎉 SUCCESS achieved in iteration {iteration}!")
                    self.success_achieved = True

                    # Log final successful result
                    feedback = "SUCCESS: All criteria met!"
                    self.results_logger.log_iteration(
                        iteration, command, constraints_code, metrics, feedback
                    )

                    return True, constraint_function, self._generate_summary()

                # Step 7: Generate feedback for next iteration
                print("💭 Generating feedback for next iteration...")
                feedback = self.feedback_generator.generate_feedback(
                    command, constraints_code, metrics, iteration
                )

                # Log this iteration
                self.results_logger.log_iteration(
                    iteration, command, constraints_code, metrics, feedback
                )

                # Step 8: Update context for next iteration
                context = feedback

                print(f"📝 Iteration {iteration} complete")

            except Exception as e:
                print(f"❌ Error in iteration {iteration}: {e}")
                print(traceback.format_exc())
                continue

        # If we reach here, max iterations exceeded without success
        print(f"\n⏰ Max iterations ({self.max_iterations}) reached")

        if best_constraint_function is not None:
            print(
                f"🥈 Returning best result from iteration {self.best_result['iteration']} (score: {best_score:.3f})"
            )
            return False, best_constraint_function, self._generate_summary()
        else:
            print("💔 No valid constraints generated")
            return False, None, self._generate_summary()

    def _generate_constraints(self, context: str) -> Optional[str]:
        """Generate constraint function code from LLM"""
        try:
            response = self.llm_client.generate_response(context)
            constraints_code = self.code_parser.extract_function_from_response(response)

            print(f"📝 Generated {len(constraints_code)} characters of constraint code")
            return constraints_code

        except Exception as e:
            print(f"❌ LLM generation error: {e}")
            return None

    def _parse_constraints(self, constraints_code: str) -> Optional[Any]:
        """Parse and validate constraint function"""
        try:
            # Parse the constraint function
            constraint_function = self.code_parser.parse_constraint_function(
                constraints_code
            )

            # Test with dummy inputs
            self.code_parser.test_constraint_function(
                constraint_function, self.kinodynamic_model, self.config
            )

            print("✅ Constraint function parsed and validated successfully")
            return constraint_function

        except Exception as e:
            print(f"❌ Constraint parsing error: {e}")
            return None

    def _solve_trajectory_optimization(
        self,
        constraint_function: Any,
        initial_state: np.ndarray,
        contact_sequence: np.ndarray,
        reference_trajectory: np.ndarray,
    ) -> Optional[Tuple]:
        """Solve trajectory optimization with generated constraints"""
        try:
            # Import here to avoid circular imports
            from mpc.mpc_opti import QuadrupedMPCOpti

            # Create MPC instance with additional constraints
            mpc = QuadrupedMPCOpti(
                model=self.kinodynamic_model,
                config=self.config,
                build=True,
                additional_constraints=constraint_function,
            )

            start_time = time.time()

            # Solve trajectory
            state_traj, grf_traj, joint_vel_traj, status = mpc.solve_trajectory(
                initial_state, reference_trajectory, contact_sequence
            )

            solve_time = time.time() - start_time

            # Combine into input trajectory format
            input_traj = np.concatenate([joint_vel_traj, grf_traj], axis=1)

            print(f"⚡ Optimization completed in {solve_time:.2f}s, status: {status}")

            return state_traj, input_traj, grf_traj, status, solve_time

        except Exception as e:
            print(f"❌ Optimization error: {e}")
            return None

    def _calculate_performance_score(self, metrics: TrajectoryMetrics) -> float:
        """Calculate overall performance score for ranking iterations"""
        score = 0.0

        # Optimization convergence (30% weight)
        if metrics.optimization_converged:
            score += 0.30

        # Constraint satisfaction (20% weight)
        if metrics.max_violation < 0.1:
            score += 0.20
        elif metrics.max_violation < 0.5:
            score += 0.10

        # Maneuver-specific criteria (50% weight total)
        if metrics.height_clearance_achieved:
            score += 0.15
        if metrics.rotation_target_achieved:
            score += 0.20
        if metrics.landing_stable:
            score += 0.15

        return score

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of refinement process"""
        summary = {
            "total_iterations": self.current_iteration,
            "success_achieved": self.success_achieved,
            "best_iteration": (
                self.best_result["iteration"] if self.best_result else None
            ),
            "best_score": (
                self._calculate_performance_score(self.best_result["metrics"])
                if self.best_result
                else 0.0
            ),
        }

        if self.best_result:
            summary["best_metrics"] = {
                "converged": self.best_result["metrics"].optimization_converged,
                "height_clearance": self.best_result[
                    "metrics"
                ].height_clearance_achieved,
                "rotation_target": self.best_result["metrics"].rotation_target_achieved,
                "landing_stable": self.best_result["metrics"].landing_stable,
                "max_violation": self.best_result["metrics"].max_violation,
                "solve_time": self.best_result["metrics"].solve_time,
            }

        # Get iteration history from logger
        iteration_summary = self.results_logger.get_iteration_summary()
        summary.update(iteration_summary)

        return summary


# Convenience function for easy usage
def run_llm_constraint_generation(
    command: str,
    kinodynamic_model: Any,
    config: Any,
    initial_state: np.ndarray,
    contact_sequence: np.ndarray,
    reference_trajectory: np.ndarray,
    max_iterations: int = 10,
) -> Tuple[bool, Optional[Any], Dict[str, Any]]:
    """
    Convenience function to run LLM-based constraint generation

    Args:
        command: Natural language command
        kinodynamic_model: Robot dynamics model
        config: Configuration object
        initial_state: Initial robot state
        contact_sequence: Contact schedule
        reference_trajectory: Reference trajectory
        max_iterations: Maximum refinement iterations

    Returns:
        Tuple of (success, best_constraint_function, summary)
    """

    engine = IterativeRefinementEngine(
        kinodynamic_model=kinodynamic_model,
        config=config,
        max_iterations=max_iterations,
    )

    return engine.run_refinement_loop(
        command=command,
        initial_state=initial_state,
        contact_sequence=contact_sequence,
        reference_trajectory=reference_trajectory,
    )
