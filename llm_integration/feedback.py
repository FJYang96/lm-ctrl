"""
Feedback mechanism to analyze simulation results and provide context for LLM iterations
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass
class TrajectoryMetrics:
    """Metrics extracted from a trajectory optimization result"""

    # Optimization status
    optimization_converged: bool
    solver_status: int
    constraint_violations: List[Dict[str, float]]
    max_violation: float

    # Trajectory analysis
    initial_height: float
    max_height: float
    final_height: float
    initial_pitch: float
    final_pitch: float
    pitch_change: float

    # Success criteria for maneuvers
    height_clearance_achieved: bool
    rotation_target_achieved: bool
    landing_stable: bool
    overall_success: bool

    # Performance metrics
    solve_time: float
    num_iterations: int


class TrajectoryAnalyzer:
    """
    Analyze trajectory optimization results and provide feedback for LLM iteration
    """

    def __init__(self, config: Any):
        """
        Initialize trajectory analyzer

        Args:
            config: Configuration object with maneuver parameters
        """
        self.config = config

    def analyze_trajectory(
        self,
        state_traj: np.ndarray,
        input_traj: np.ndarray,
        grf_traj: np.ndarray,
        contact_sequence: np.ndarray,
        solver_status: int,
        solve_time: float = 0.0,
        num_iterations: int = 0,
    ) -> TrajectoryMetrics:
        """
        Analyze trajectory results and extract key metrics

        Args:
            state_traj: State trajectory (N+1, 24)
            input_traj: Input trajectory (N, 24)
            grf_traj: Ground reaction forces (N, 12)
            contact_sequence: Contact flags (4, N)
            solver_status: Solver return status
            solve_time: Time taken to solve
            num_iterations: Solver iterations

        Returns:
            TrajectoryMetrics object with analysis results
        """

        # Basic trajectory properties
        initial_height = float(state_traj[0, 2])  # z position
        max_height = float(np.max(state_traj[:, 2]))
        final_height = float(state_traj[-1, 2])

        initial_pitch = float(state_traj[0, 7])  # pitch angle
        final_pitch = float(state_traj[-1, 7])
        pitch_change = final_pitch - initial_pitch

        # Constraint violation analysis
        constraint_violations = self._analyze_constraint_violations(
            state_traj, input_traj, grf_traj, contact_sequence
        )
        max_violation = max([v["magnitude"] for v in constraint_violations] + [0.0])

        # Success criteria evaluation
        metrics = TrajectoryMetrics(
            optimization_converged=(solver_status == 0),
            solver_status=solver_status,
            constraint_violations=constraint_violations,
            max_violation=max_violation,
            initial_height=initial_height,
            max_height=max_height,
            final_height=final_height,
            initial_pitch=initial_pitch,
            final_pitch=final_pitch,
            pitch_change=pitch_change,
            height_clearance_achieved=self._check_height_clearance(state_traj),
            rotation_target_achieved=self._check_rotation_target(
                initial_pitch, final_pitch
            ),
            landing_stable=self._check_landing_stability(state_traj),
            overall_success=False,  # Will be set below
            solve_time=solve_time,
            num_iterations=num_iterations,
        )

        # Overall success evaluation
        metrics.overall_success = (
            metrics.optimization_converged
            and metrics.height_clearance_achieved
            and metrics.rotation_target_achieved
            and metrics.landing_stable
            and metrics.max_violation < 0.1
        )

        return metrics

    def _analyze_constraint_violations(
        self,
        state_traj: np.ndarray,
        input_traj: np.ndarray,
        grf_traj: np.ndarray,
        contact_sequence: np.ndarray,
    ) -> List[Dict[str, float]]:
        """Analyze constraint violations in the trajectory"""
        violations = []

        # Check foot height violations (stance feet should be at z=0)
        for i in range(len(state_traj) - 1):
            for leg in range(4):
                if contact_sequence[leg, i] > 0.5:  # In stance
                    # This is simplified - in practice would use forward kinematics
                    # For now, assume foot height can be estimated from base height
                    estimated_foot_height = (
                        state_traj[i, 2] - 0.3
                    )  # Rough approximation
                    if abs(estimated_foot_height) > 0.05:  # 5cm tolerance
                        violations.append(
                            {
                                "type": "foot_height",
                                "time_step": i,
                                "leg": leg,
                                "magnitude": abs(estimated_foot_height),
                                "description": f"Stance foot {leg} not on ground at step {i}",
                            }
                        )

        # Check force violations (negative normal forces)
        for i in range(len(grf_traj)):
            for leg in range(4):
                normal_force = grf_traj[i, leg * 3 + 2]  # z-component of force
                if contact_sequence[leg, i] > 0.5 and normal_force < 0:
                    violations.append(
                        {
                            "type": "negative_force",
                            "time_step": i,
                            "leg": leg,
                            "magnitude": abs(normal_force),
                            "description": f"Negative normal force on leg {leg} at step {i}",
                        }
                    )

        # Check velocity discontinuities
        for i in range(1, len(state_traj)):
            vel_change = np.linalg.norm(state_traj[i, 3:6] - state_traj[i - 1, 3:6])
            if vel_change > 5.0:  # Large velocity jump
                violations.append(
                    {
                        "type": "velocity_discontinuity",
                        "time_step": i,
                        "leg": -1,
                        "magnitude": vel_change,
                        "description": f"Large velocity discontinuity at step {i}",
                    }
                )

        return violations

    def _check_height_clearance(self, state_traj: np.ndarray) -> bool:
        """Check if robot achieved sufficient height clearance"""
        max_height = np.max(state_traj[:, 2])
        initial_height = state_traj[0, 2]
        clearance = max_height - initial_height

        # For backflip, need at least 20cm additional height
        return clearance >= 0.20

    def _check_rotation_target(self, initial_pitch: float, final_pitch: float) -> bool:
        """Check if rotation target was achieved"""
        pitch_change = final_pitch - initial_pitch
        target_rotation = 2 * np.pi  # Full backflip

        # Allow 10% tolerance
        error = abs(pitch_change - target_rotation)
        return error < 0.1 * target_rotation

    def _check_landing_stability(self, state_traj: np.ndarray) -> bool:
        """Check if landing is stable"""
        final_velocity = np.linalg.norm(state_traj[-1, 3:6])
        final_angular_velocity = np.linalg.norm(state_traj[-1, 9:12])

        # Stable if velocities are small
        return final_velocity < 1.0 and final_angular_velocity < 2.0


class FeedbackGenerator:
    """
    Generate structured feedback for LLM based on trajectory analysis
    """

    def __init__(self):
        """Initialize feedback generator"""
        pass

    def generate_feedback(
        self,
        command: str,
        previous_constraints: str,
        metrics: TrajectoryMetrics,
        iteration: int,
    ) -> str:
        """
        Generate feedback text for LLM based on trajectory analysis

        Args:
            command: Original user command
            previous_constraints: Previously generated constraint code
            metrics: Analysis results from trajectory
            iteration: Current iteration number

        Returns:
            Formatted feedback string for LLM
        """

        # Identify key issues
        issues = self._identify_issues(metrics)
        suggestions = self._generate_suggestions(metrics, issues)

        feedback = f"""
## Iteration {iteration} Results Analysis

**Command**: {command}

**Previous Constraint Function**:
```python
{previous_constraints}
```

## Optimization Results

**Solver Status**: {"✅ Converged" if metrics.optimization_converged else "❌ Failed"}
- Return code: {metrics.solver_status}
- Solve time: {metrics.solve_time:.2f}s
- Iterations: {metrics.num_iterations}

**Constraint Violations**: {len(metrics.constraint_violations)} total
- Maximum violation: {metrics.max_violation:.4f}

### Detailed Violations:
"""

        for violation in metrics.constraint_violations[:5]:  # Show top 5
            feedback += f"- {violation['type']}: {violation['description']} (magnitude: {violation['magnitude']:.3f})\n"

        if len(metrics.constraint_violations) > 5:
            feedback += (
                f"- ... and {len(metrics.constraint_violations) - 5} more violations\n"
            )

        feedback += f"""

## Trajectory Analysis

**Height Profile**:
- Initial: {metrics.initial_height:.3f}m
- Maximum: {metrics.max_height:.3f}m
- Final: {metrics.final_height:.3f}m
- Clearance: {metrics.max_height - metrics.initial_height:.3f}m

**Rotation Analysis**:
- Initial pitch: {metrics.initial_pitch:.3f} rad ({np.degrees(metrics.initial_pitch):.1f}°)
- Final pitch: {metrics.final_pitch:.3f} rad ({np.degrees(metrics.final_pitch):.1f}°)
- Total rotation: {metrics.pitch_change:.3f} rad ({np.degrees(metrics.pitch_change):.1f}°)
- Target rotation: {2*np.pi:.3f} rad (360°)
- Rotation error: {abs(metrics.pitch_change - 2*np.pi):.3f} rad

## Success Criteria Evaluation

✅❌ **Height Clearance**: {"✅ Achieved" if metrics.height_clearance_achieved else "❌ Insufficient"}
✅❌ **Rotation Target**: {"✅ Achieved" if metrics.rotation_target_achieved else "❌ Missing target"}
✅❌ **Landing Stability**: {"✅ Stable" if metrics.landing_stable else "❌ Unstable"}
✅❌ **Overall Success**: {"✅ SUCCESS" if metrics.overall_success else "❌ NEEDS IMPROVEMENT"}

## Issues Identified

{self._format_issues(issues)}

## Specific Improvement Suggestions

{self._format_suggestions(suggestions)}

## Guidelines for Next Iteration

1. **Address Primary Issues**: Focus on the most critical problems first
2. **Gradual Refinement**: Make incremental changes rather than complete rewrites
3. **Physical Feasibility**: Ensure all constraints are physically achievable
4. **Numerical Stability**: Avoid overly tight bounds that cause solver issues

Please generate an improved constraint function that specifically addresses the identified issues while maintaining the overall structure that worked well.
"""

        return feedback

    def _identify_issues(self, metrics: TrajectoryMetrics) -> List[Dict[str, Any]]:
        """Identify primary issues from metrics"""
        issues = []

        if not metrics.optimization_converged:
            issues.append(
                {
                    "category": "optimization",
                    "severity": "critical",
                    "description": "Solver failed to converge",
                    "details": f"Status code: {metrics.solver_status}",
                }
            )

        if metrics.max_violation > 0.1:
            issues.append(
                {
                    "category": "constraints",
                    "severity": "high",
                    "description": "Large constraint violations",
                    "details": f"Max violation: {metrics.max_violation:.3f}",
                }
            )

        if not metrics.height_clearance_achieved:
            issues.append(
                {
                    "category": "maneuver",
                    "severity": "high",
                    "description": "Insufficient height for maneuver",
                    "details": f"Clearance: {metrics.max_height - metrics.initial_height:.3f}m (need >0.20m)",
                }
            )

        if not metrics.rotation_target_achieved:
            rotation_error = abs(metrics.pitch_change - 2 * np.pi)
            issues.append(
                {
                    "category": "maneuver",
                    "severity": "high",
                    "description": "Incomplete rotation",
                    "details": f"Error: {rotation_error:.3f} rad ({np.degrees(rotation_error):.1f}°)",
                }
            )

        if not metrics.landing_stable:
            issues.append(
                {
                    "category": "maneuver",
                    "severity": "medium",
                    "description": "Unstable landing",
                    "details": "High final velocities",
                }
            )

        return issues

    def _generate_suggestions(
        self, metrics: TrajectoryMetrics, issues: List[Dict]
    ) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []

        for issue in issues:
            if issue["category"] == "optimization":
                suggestions.append("Relax constraint bounds to improve convergence")
                suggestions.append("Add slack variables for critical constraints")

            elif issue["category"] == "constraints":
                suggestions.append(
                    "Review constraint formulations for numerical stability"
                )
                suggestions.append("Check for conflicting constraints")

            elif issue["category"] == "maneuver":
                if "height" in issue["description"].lower():
                    suggestions.append(
                        "Add explicit minimum height constraints during flight phase"
                    )
                    suggestions.append("Increase takeoff velocity targets")

                if "rotation" in issue["description"].lower():
                    suggestions.append(
                        "Add progressive rotation constraints throughout trajectory"
                    )
                    suggestions.append("Ensure sufficient time for full rotation")

                if "landing" in issue["description"].lower():
                    suggestions.append(
                        "Add terminal velocity constraints for stable landing"
                    )
                    suggestions.append("Constrain final angular velocities")

        return list(set(suggestions))  # Remove duplicates

    def _format_issues(self, issues: List[Dict]) -> str:
        """Format issues for display"""
        if not issues:
            return "✅ No major issues identified!"

        formatted = ""
        for i, issue in enumerate(issues, 1):
            severity_icon = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢",
            }
            icon = severity_icon.get(issue["severity"], "❓")

            formatted += f"{i}. {icon} **{issue['description']}**\n"
            formatted += f"   - Category: {issue['category']}\n"
            formatted += f"   - Details: {issue['details']}\n\n"

        return formatted

    def _format_suggestions(self, suggestions: List[str]) -> str:
        """Format suggestions for display"""
        if not suggestions:
            return "🎯 Trajectory looks good! Consider minor refinements only."

        formatted = ""
        for i, suggestion in enumerate(suggestions, 1):
            formatted += f"{i}. {suggestion}\n"

        return formatted


class ResultsLogger:
    """
    Log iteration results for analysis and debugging
    """

    def __init__(self, log_dir: str = "results/llm_iterations"):
        """Initialize results logger"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_iteration(
        self,
        iteration: int,
        command: str,
        constraints_code: str,
        metrics: TrajectoryMetrics,
        feedback: str,
    ) -> None:
        """Log results from a single iteration"""

        iteration_data = {
            "iteration": iteration,
            "command": command,
            "constraints_code": constraints_code,
            "metrics": {
                "optimization_converged": metrics.optimization_converged,
                "solver_status": metrics.solver_status,
                "max_violation": metrics.max_violation,
                "height_clearance_achieved": metrics.height_clearance_achieved,
                "rotation_target_achieved": metrics.rotation_target_achieved,
                "landing_stable": metrics.landing_stable,
                "overall_success": metrics.overall_success,
                "solve_time": metrics.solve_time,
                "initial_height": metrics.initial_height,
                "max_height": metrics.max_height,
                "final_height": metrics.final_height,
                "pitch_change": metrics.pitch_change,
            },
            "feedback": feedback,
            "constraint_violations": metrics.constraint_violations,
        }

        # Save to JSON file
        log_file = self.log_dir / f"iteration_{iteration:03d}.json"
        with open(log_file, "w") as f:
            json.dump(iteration_data, f, indent=2)

        print(f"💾 Logged iteration {iteration} results to {log_file}")

    def get_iteration_summary(self) -> Dict[str, Any]:
        """Get summary of all iterations"""
        log_files = sorted(self.log_dir.glob("iteration_*.json"))

        summary = {
            "total_iterations": len(log_files),
            "success_rate": 0.0,
            "convergence_rate": 0.0,
            "best_iteration": None,
            "iteration_history": [],
        }

        if not log_files:
            return summary

        successes = 0
        convergences = 0
        best_score = -1

        for log_file in log_files:
            with open(log_file) as f:
                data = json.load(f)

            metrics = data["metrics"]

            if metrics["overall_success"]:
                successes += 1
            if metrics["optimization_converged"]:
                convergences += 1

            # Score based on multiple criteria
            score = (
                int(metrics["optimization_converged"]) * 0.3
                + int(metrics["height_clearance_achieved"]) * 0.2
                + int(metrics["rotation_target_achieved"]) * 0.3
                + int(metrics["landing_stable"]) * 0.2
            )

            if score > best_score:
                best_score = score
                summary["best_iteration"] = data["iteration"]

            summary["iteration_history"].append(
                {
                    "iteration": data["iteration"],
                    "success": metrics["overall_success"],
                    "converged": metrics["optimization_converged"],
                    "score": score,
                }
            )

        summary["success_rate"] = successes / len(log_files)
        summary["convergence_rate"] = convergences / len(log_files)

        return summary
