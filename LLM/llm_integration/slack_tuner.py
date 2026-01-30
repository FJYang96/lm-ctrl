"""
LLM-based Slack Weight Tuner for iterative constraint optimization.

This module uses LLM to adaptively adjust slack weights based on
optimization results and constraint hardness analysis.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .llm_client import LLMClient, get_llm_client


# =============================================================================
# Prompts for Slack Tuning
# =============================================================================

SLACK_TUNING_SYSTEM_PROMPT = """You are an expert in numerical optimization and robotics.
Your task is to tune slack variable weights for a quadruped robot trajectory optimization problem.

## Background
- Slack variables allow constraints to be violated with a penalty cost
- Higher weight = constraint is harder to violate (more "hard")
- Lower weight = constraint can be violated more easily (more "soft")
- If optimization fails, some constraints might be too tight
- If constraints are violated too much, the solution might be physically invalid

## Constraint Types
1. **friction_cone_constraints**: Ensures ground reaction forces stay within friction cone
   - Critical for stance phase, can be relaxed during flight
   
2. **foot_height_constraints**: Keeps feet at correct height (on ground during stance)
   - Important for contact stability
   
3. **foot_velocity_constraints**: Limits foot sliding during stance (no-slip)
   - Can be slightly relaxed if needed
   
4. **joint_limits_constraints**: Hardware joint angle limits
   - Should generally stay high (safety)
   
5. **input_limits_constraints**: Actuator force/velocity limits
   - Should generally stay high (hardware limits)
   
6. **body_clearance_constraints**: Keeps robot body above ground
   - Important for safety
   
7. **complementarity_constraints**: Contact force-velocity complementarity
   - Can be relaxed more than others

## Tuning Strategy
1. If solver FAILS: Reduce weight of constraints with highest violation
2. If solver SUCCEEDS but violations are high: Slightly increase weights
3. If solver SUCCEEDS with low violations: Good! Can try increasing weights to test
4. Consider task context (backflip needs different weights than walking)

## Output Format
Return ONLY a JSON object with the new weights. No explanation outside the JSON.
"""

SLACK_TUNING_USER_PROMPT = """## Task Description
{task_description}

## Current Slack Weights
```json
{current_weights}
```

## Optimization Result
- Solver Status: {solver_status}
- Iterations Attempted: {iteration}

## Constraint Hardness Analysis
{hardness_report}

## History of Previous Adjustments
{history_summary}

## Instructions
Based on the above information, suggest new slack weights.
- If solver failed, try relaxing the hardest constraints
- If solver succeeded, try to reduce violations further
- Consider the specific task requirements

Return your answer as a JSON object:
```json
{{
    "friction_cone_constraints": <number>,
    "foot_height_constraints": <number>,
    "foot_velocity_constraints": <number>,
    "joint_limits_constraints": <number>,
    "input_limits_constraints": <number>,
    "body_clearance_constraints": <number>,
    "complementarity_constraints": <number>,
    "reasoning": "<brief explanation of changes>"
}}
```
"""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TuningIteration:
    """Record of a single tuning iteration."""
    iteration: int
    weights_before: dict[str, float]
    weights_after: dict[str, float]
    solver_status: int
    total_violation: float
    hardness_report: dict[str, dict]
    llm_reasoning: str = ""


@dataclass
class TuningResult:
    """Result of the complete tuning process."""
    success: bool
    final_weights: dict[str, float]
    best_weights: dict[str, float]
    best_violation: float
    iterations: list[TuningIteration]
    trajectory: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


# =============================================================================
# LLM Slack Tuner
# =============================================================================

class LLMSlackTuner:
    """
    Uses LLM to iteratively tune slack weights based on optimization feedback.
    """
    
    # Default initial weights
    DEFAULT_WEIGHTS = {
        "friction_cone_constraints": 1e5,
        "foot_height_constraints": 1e4,
        "foot_velocity_constraints": 1e3,
        "joint_limits_constraints": 1e5,
        "input_limits_constraints": 1e5,
        "body_clearance_constraints": 1e4,
        "complementarity_constraints": 1e2,
    }
    
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        initial_weights: dict[str, float] | None = None,
    ):
        """
        Initialize the slack tuner.
        
        Args:
            llm_client: LLM client instance (creates default if None)
            initial_weights: Starting slack weights (uses defaults if None)
        """
        self.llm_client = llm_client or get_llm_client()
        self.current_weights = (initial_weights or self.DEFAULT_WEIGHTS).copy()
        self.history: list[TuningIteration] = []
    
    def format_hardness_report(self, hardness: dict[str, dict]) -> str:
        """Format hardness analysis as a readable string."""
        if not hardness:
            return "No hardness data available (solver may have failed early)"
        
        lines = []
        lines.append("| Constraint | Total Violation | Max Violation | Active Steps |")
        lines.append("|------------|-----------------|---------------|--------------|")
        
        for name, metrics in hardness.items():
            total = metrics.get("total_violation", 0)
            max_v = metrics.get("max_violation", 0)
            active = len(metrics.get("active_timesteps", []))
            
            # Status indicator
            if total > 0.1:
                status = "🔴 HIGH"
            elif total > 0.01:
                status = "🟡 MED"
            elif total > 0:
                status = "🟢 LOW"
            else:
                status = "✅ ZERO"
            
            lines.append(f"| {name:<30} | {total:>13.6f} | {max_v:>13.6f} | {active:>4} steps {status} |")
        
        return "\n".join(lines)
    
    def format_history_summary(self) -> str:
        """Format tuning history as a summary."""
        if not self.history:
            return "This is the first iteration."
        
        lines = ["Previous iterations:"]
        for record in self.history[-3:]:  # Last 3 iterations
            status = "✅ SUCCESS" if record.solver_status == 0 else "❌ FAILED"
            lines.append(f"  - Iter {record.iteration}: {status}, violation={record.total_violation:.6f}")
            if record.llm_reasoning:
                lines.append(f"    Reasoning: {record.llm_reasoning[:100]}...")
        
        return "\n".join(lines)
    
    def get_updated_weights(
        self,
        task_description: str,
        solver_status: int,
        hardness_report: dict[str, dict],
        iteration: int,
    ) -> dict[str, float]:
        """
        Ask LLM for updated slack weights based on current results.
        
        Args:
            task_description: Description of the task (e.g., "backflip")
            solver_status: 0 for success, non-zero for failure
            hardness_report: Constraint hardness analysis from MPC
            iteration: Current iteration number
            
        Returns:
            Dictionary of new slack weights
        """
        # Calculate total violation
        total_violation = sum(
            m.get("total_violation", 0) for m in hardness_report.values()
        )
        
        # Format the prompt
        prompt = SLACK_TUNING_USER_PROMPT.format(
            task_description=task_description,
            current_weights=json.dumps(self.current_weights, indent=2),
            solver_status="SUCCESS ✅" if solver_status == 0 else "FAILED ❌",
            iteration=iteration,
            hardness_report=self.format_hardness_report(hardness_report),
            history_summary=self.format_history_summary(),
        )
        
        # Get LLM response
        try:
            # Combine system prompt and user prompt
            full_prompt = f"{SLACK_TUNING_SYSTEM_PROMPT}\n\n{prompt}"
            response = self.llm_client.generate_response(full_prompt)
            new_weights, reasoning = self._parse_response(response)
        except Exception as e:
            print(f"LLM call failed: {e}")
            # Fallback: simple heuristic adjustment
            new_weights = self._fallback_adjustment(solver_status, hardness_report)
            reasoning = "Fallback heuristic (LLM failed)"
        
        # Record history
        self.history.append(TuningIteration(
            iteration=iteration,
            weights_before=self.current_weights.copy(),
            weights_after=new_weights.copy(),
            solver_status=solver_status,
            total_violation=total_violation,
            hardness_report=hardness_report,
            llm_reasoning=reasoning,
        ))
        
        # Update current weights
        self.current_weights = new_weights.copy()
        
        return new_weights
    
    def _parse_response(self, response: str) -> tuple[dict[str, float], str]:
        """Parse LLM response to extract weights and reasoning."""
        # Find JSON block
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            raise ValueError("No JSON found in response")
        
        data = json.loads(json_match.group())
        
        # Extract reasoning
        reasoning = data.pop("reasoning", "")
        
        # Merge with current weights (in case some are missing)
        new_weights = self.current_weights.copy()
        for key, value in data.items():
            if key in new_weights:
                # Handle scientific notation strings
                if isinstance(value, str):
                    value = float(value)
                new_weights[key] = float(value)
        
        return new_weights, reasoning
    
    def _fallback_adjustment(
        self,
        solver_status: int,
        hardness_report: dict[str, dict],
    ) -> dict[str, float]:
        """Simple heuristic adjustment when LLM fails."""
        new_weights = self.current_weights.copy()
        
        if solver_status != 0:
            # Solver failed - reduce weights of hardest constraints
            if hardness_report:
                # Find constraint with highest violation
                hardest = max(
                    hardness_report.items(),
                    key=lambda x: x[1].get("total_violation", 0)
                )
                if hardest[0] in new_weights:
                    new_weights[hardest[0]] *= 0.1  # Reduce by 10x
        else:
            # Solver succeeded - slightly increase weights if violations exist
            for name, metrics in hardness_report.items():
                if metrics.get("total_violation", 0) > 0.01 and name in new_weights:
                    new_weights[name] *= 1.5  # Increase by 1.5x
        
        return new_weights
    
    def reset(self, initial_weights: dict[str, float] | None = None) -> None:
        """Reset tuner state for a new tuning session."""
        self.current_weights = (initial_weights or self.DEFAULT_WEIGHTS).copy()
        self.history = []


# =============================================================================
# Main Tuning Loop
# =============================================================================

def run_slack_tuning_loop(
    model: Any,
    config: Any,
    initial_state: np.ndarray,
    ref: np.ndarray,
    contact_sequence: np.ndarray,
    task_description: str = "quadruped hopping motion",
    max_iterations: int = 5,
    convergence_threshold: float = 0.01,
    initial_weights: dict[str, float] | None = None,
    verbose: bool = True,
) -> TuningResult:
    """
    Run iterative slack weight tuning with LLM feedback.
    
    Args:
        model: KinoDynamic_Model instance
        config: Configuration object
        initial_state: Initial robot state
        ref: Reference trajectory
        contact_sequence: Contact sequence array
        task_description: Description of the task for LLM context
        max_iterations: Maximum tuning iterations
        convergence_threshold: Stop if total violation below this
        initial_weights: Starting slack weights
        verbose: Print progress information
        
    Returns:
        TuningResult with best weights and trajectory
    """
    from mpc.mpc_opti_slack import QuadrupedMPCOptiSlack
    
    # Initialize tuner
    tuner = LLMSlackTuner(initial_weights=initial_weights)
    
    # Track best result
    best_weights = tuner.current_weights.copy()
    best_violation = float('inf')
    best_trajectory = None
    
    if verbose:
        print("\n" + "=" * 70)
        print("LLM SLACK WEIGHT TUNING")
        print(f"Task: {task_description}")
        print(f"Max iterations: {max_iterations}")
        print("=" * 70)
    
    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n{'─' * 70}")
            print(f"ITERATION {iteration}")
            print(f"{'─' * 70}")
            print(f"Current weights: {json.dumps(tuner.current_weights, indent=2)}")
        
        # Create MPC with current slack weights
        try:
            mpc = QuadrupedMPCOptiSlack(
                model=model,
                config=config,
                use_slack=True,
                slack_weights=tuner.current_weights,
            )
        except Exception as e:
            if verbose:
                print(f"❌ MPC creation failed: {e}")
            continue
        
        # Solve trajectory optimization
        if verbose:
            print("\n🔧 Solving trajectory optimization...")
        
        state_traj, grf_traj, joint_vel_traj, status = mpc.solve_trajectory(
            initial_state, ref, contact_sequence
        )
        
        # Analyze constraint hardness
        hardness = mpc.analyze_constraint_hardness()
        total_violation = sum(m.get("total_violation", 0) for m in hardness.values())
        
        if verbose:
            status_str = "✅ SUCCESS" if status == 0 else "❌ FAILED"
            print(f"\nSolver status: {status_str}")
            print(f"Total violation: {total_violation:.6f}")
            mpc.print_constraint_hardness_report()
        
        # Track best result
        if status == 0 and total_violation < best_violation:
            best_violation = total_violation
            best_weights = tuner.current_weights.copy()
            best_trajectory = (state_traj, grf_traj, joint_vel_traj)
            if verbose:
                print("📌 New best result!")
        
        # Check convergence
        if status == 0 and total_violation < convergence_threshold:
            if verbose:
                print(f"\n✅ CONVERGED! Total violation {total_violation:.6f} < {convergence_threshold}")
            break
        
        # Get LLM suggestions for next iteration (unless last iteration)
        if iteration < max_iterations:
            if verbose:
                print("\n🤖 Asking LLM for weight adjustments...")
            
            new_weights = tuner.get_updated_weights(
                task_description=task_description,
                solver_status=status,
                hardness_report=hardness,
                iteration=iteration,
            )
            
            if verbose:
                # Show what changed
                print("\nLLM suggested changes:")
                for key in new_weights:
                    old = tuner.history[-1].weights_before.get(key, 0)
                    new = new_weights[key]
                    if old != new:
                        ratio = new / old if old > 0 else float('inf')
                        direction = "⬆️" if ratio > 1 else "⬇️"
                        print(f"  {key}: {old:.0e} → {new:.0e} ({ratio:.1f}x {direction})")
                
                if tuner.history[-1].llm_reasoning:
                    print(f"\nLLM reasoning: {tuner.history[-1].llm_reasoning}")
    
    # Final summary
    if verbose:
        print("\n" + "=" * 70)
        print("TUNING COMPLETE")
        print("=" * 70)
        print(f"Total iterations: {len(tuner.history)}")
        print(f"Best total violation: {best_violation:.6f}")
        print(f"Best weights: {json.dumps(best_weights, indent=2)}")
    
    return TuningResult(
        success=best_trajectory is not None,
        final_weights=tuner.current_weights,
        best_weights=best_weights,
        best_violation=best_violation,
        iterations=tuner.history,
        trajectory=best_trajectory,
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def tune_for_task(
    model: Any,
    config: Any,
    initial_state: np.ndarray,
    ref: np.ndarray,
    contact_sequence: np.ndarray,
    task: str,
) -> TuningResult:
    """
    Convenience function to tune slack weights for common tasks.
    
    Args:
        model: KinoDynamic_Model instance
        config: Configuration object
        initial_state: Initial state
        ref: Reference trajectory
        contact_sequence: Contact sequence
        task: Task name ("backflip", "jump", "walk", etc.)
        
    Returns:
        TuningResult
    """
    # Task-specific initial weights
    task_weights = {
        "backflip": {
            "friction_cone_constraints": 1e3,  # Relax for flight
            "foot_height_constraints": 1e4,
            "foot_velocity_constraints": 1e2,
            "joint_limits_constraints": 1e5,
            "input_limits_constraints": 1e5,
            "body_clearance_constraints": 1e5,  # Important for backflip
            "complementarity_constraints": 1e1,
        },
        "jump": {
            "friction_cone_constraints": 1e4,
            "foot_height_constraints": 1e4,
            "foot_velocity_constraints": 1e3,
            "joint_limits_constraints": 1e5,
            "input_limits_constraints": 1e5,
            "body_clearance_constraints": 1e4,
            "complementarity_constraints": 1e2,
        },
        "walk": {
            "friction_cone_constraints": 1e5,  # Important for walking
            "foot_height_constraints": 1e5,
            "foot_velocity_constraints": 1e4,  # No-slip important
            "joint_limits_constraints": 1e5,
            "input_limits_constraints": 1e5,
            "body_clearance_constraints": 1e3,
            "complementarity_constraints": 1e3,
        },
    }
    
    initial_weights = task_weights.get(task.lower(), None)
    
    return run_slack_tuning_loop(
        model=model,
        config=config,
        initial_state=initial_state,
        ref=ref,
        contact_sequence=contact_sequence,
        task_description=f"{task} motion for quadruped robot",
        initial_weights=initial_weights,
    )

