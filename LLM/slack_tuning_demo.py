#!/usr/bin/env python3
"""
Demo script for LLM-based Slack Weight Tuning.

This script demonstrates the complete flow:
1. Load robot model and config
2. Run iterative slack tuning with LLM feedback
3. Analyze and visualize results

Usage:
    python slack_tuning_demo.py --task "backflip"
    python slack_tuning_demo.py --task "jump" --max-iterations 10
    python slack_tuning_demo.py --mock  # Use mock LLM for testing
"""

import argparse
import json
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="LLM Slack Tuning Demo")
    parser.add_argument(
        "--task",
        type=str,
        default="hopping",
        help="Task description (e.g., 'backflip', 'jump', 'hopping')",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum tuning iterations",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Convergence threshold for total violation",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM (no API calls)",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default="results/slack_tuning",
        help="Directory to save results",
    )
    args = parser.parse_args()

    # Enable mock mode if requested
    if args.mock:
        os.environ["USE_MOCK_LLM"] = "true"
        print("🔧 Using mock LLM mode")

    # Import after setting environment
    import config
    from llm_integration.slack_tuner import run_slack_tuning_loop, TuningResult
    from mpc.dynamics.model import KinoDynamic_Model
    from utils.simulation import create_reference_trajectory
    from utils import conversion

    print("\n" + "=" * 70)
    print("LLM SLACK TUNING DEMO")
    print("=" * 70)
    print(f"Task: {args.task}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Convergence threshold: {args.threshold}")
    print("=" * 70)

    # =========================================================================
    # Step 1: Setup Robot Model
    # =========================================================================
    print("\n📦 Loading robot model...")
    
    # KinoDynamic_Model expects the full config object
    kinodynamic_model = KinoDynamic_Model(config)
    
    print(f"  Robot: {config.robot}")
    print(f"  Mass: {config.robot_data.mass} kg")

    # =========================================================================
    # Step 2: Setup Initial State and Reference
    # =========================================================================
    print("\n📐 Setting up initial state and reference...")
    
    # Convert simulation state to MPC state format (30D)
    # This properly handles quaternion->euler conversion and adds integral terms
    initial_state, _ = conversion.sim_to_mpc(
        config.experiment.initial_qpos,
        config.experiment.initial_qvel
    )
    
    print(f"  Initial position: {initial_state[0:3]}")
    print(f"  Initial height: {initial_state[2]:.3f} m")
    print(f"  State dimension: {len(initial_state)}")
    
    # Create reference trajectory
    ref = create_reference_trajectory(
        initial_qpos=config.experiment.initial_qpos,
        target_jump_height=0.15,
    )
    
    print(f"  Reference created (target height: +0.15m)")

    # =========================================================================
    # Step 3: Get Contact Sequence
    # =========================================================================
    print("\n📅 Getting contact sequence...")
    
    contact_sequence = config.mpc_config.contact_sequence
    print(f"  Horizon: {contact_sequence.shape[1]} steps")
    print(f"  Contact pattern: stance → flight → landing")

    # =========================================================================
    # Step 4: Run Slack Tuning Loop
    # =========================================================================
    print("\n🚀 Starting LLM Slack Tuning Loop...")
    
    result: TuningResult = run_slack_tuning_loop(
        model=kinodynamic_model,
        config=config,
        initial_state=initial_state,
        ref=ref,
        contact_sequence=contact_sequence,
        task_description=f"{args.task} motion for Unitree Go2 quadruped robot",
        max_iterations=args.max_iterations,
        convergence_threshold=args.threshold,
        verbose=True,
    )

    # =========================================================================
    # Step 5: Print Final Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    print(f"\n✅ Success: {result.success}")
    print(f"📊 Best violation: {result.best_violation:.6f}")
    print(f"🔄 Total iterations: {len(result.iterations)}")
    
    print(f"\n📋 Best Weights:")
    for name, weight in result.best_weights.items():
        print(f"  {name}: {weight:.0e}")
    
    # =========================================================================
    # Step 6: Save Results
    # =========================================================================
    if args.save_results:
        os.makedirs(args.save_results, exist_ok=True)
        
        # Save best weights
        weights_file = os.path.join(args.save_results, "best_weights.json")
        with open(weights_file, "w") as f:
            json.dump(result.best_weights, f, indent=2)
        print(f"\n💾 Saved best weights to: {weights_file}")
        
        # Save iteration history
        history_file = os.path.join(args.save_results, "tuning_history.json")
        history_data = []
        for it in result.iterations:
            history_data.append({
                "iteration": it.iteration,
                "solver_status": it.solver_status,
                "total_violation": it.total_violation,
                "weights_before": it.weights_before,
                "weights_after": it.weights_after,
                "reasoning": it.llm_reasoning,
            })
        with open(history_file, "w") as f:
            json.dump(history_data, f, indent=2)
        print(f"💾 Saved tuning history to: {history_file}")
        
        # Save trajectory if available
        if result.trajectory:
            state_traj, grf_traj, joint_vel_traj = result.trajectory
            np.save(os.path.join(args.save_results, "state_traj.npy"), state_traj)
            np.save(os.path.join(args.save_results, "grf_traj.npy"), grf_traj)
            np.save(os.path.join(args.save_results, "joint_vel_traj.npy"), joint_vel_traj)
            print(f"💾 Saved trajectory to: {args.save_results}/")

    # =========================================================================
    # Step 7: Show Iteration Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("ITERATION SUMMARY")
    print("=" * 70)
    print(f"{'Iter':<5} {'Status':<10} {'Violation':<12} {'Key Changes':<40}")
    print("-" * 70)
    
    for it in result.iterations:
        status = "✅ OK" if it.solver_status == 0 else "❌ FAIL"
        
        # Find biggest weight change
        changes = []
        for key in it.weights_after:
            before = it.weights_before.get(key, 0)
            after = it.weights_after.get(key, 0)
            if before > 0 and after != before:
                ratio = after / before
                if ratio > 1.5 or ratio < 0.67:
                    direction = "↑" if ratio > 1 else "↓"
                    short_name = key.replace("_constraints", "")[:15]
                    changes.append(f"{short_name}{direction}{ratio:.1f}x")
        
        changes_str = ", ".join(changes[:2]) if changes else "no change"
        
        print(f"{it.iteration:<5} {status:<10} {it.total_violation:<12.6f} {changes_str:<40}")
    
    print("=" * 70)
    
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())

