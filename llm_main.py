#!/usr/bin/env python3
"""
LLM-integrated trajectory optimization for quadruped robots

This script demonstrates the complete LLM integration pipeline:
1. Natural language command input
2. LLM-generated constraint functions
3. Iterative refinement based on simulation feedback
4. Final trajectory optimization and execution

Usage:
    python llm_main.py "do a backflip"
    python llm_main.py "jump as high as possible"
    python llm_main.py --mock "do a front flip"
"""

import argparse
import sys
import traceback
from pathlib import Path

# Add project root to path - this must happen before other imports
sys.path.append(str(Path(__file__).parent))

# noqa: E402 - imports after path modification
import imageio
import numpy as np

import config
from llm_integration.config_loader import get_config, print_config_status
from llm_integration.iterative_refinement import run_llm_constraint_generation
from mpc.dynamics.model import KinoDynamic_Model
from utils import conversion
from utils.logging import color_print
from utils.simulation import (
    create_reference_trajectory,
    save_trajectory_results,
    simulate_trajectory,
)
from utils.visualization import (
    plot_trajectory_comparison,
    render_and_save_planned_trajectory,
)


def main() -> None:
    """Main LLM-integrated trajectory optimization pipeline"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="LLM-integrated quadruped trajectory optimization"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="do a backflip",
        help="Natural language command for the robot",
    )
    parser.add_argument(
        "--mock", action="store_true", help="Use mock LLM client for testing"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum LLM refinement iterations",
    )
    parser.add_argument(
        "--config-status",
        action="store_true",
        help="Print configuration status and exit",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable video rendering for faster execution",
    )

    args = parser.parse_args()

    # Print configuration status if requested
    if args.config_status:
        print_config_status()
        return

    # Override config if using mock
    if args.mock:
        import os

        os.environ["USE_MOCK_LLM"] = "true"

    # Load configuration
    llm_config = get_config()

    if args.no_render:
        config.experiment.render = False

    print("🤖 LLM-Integrated Quadruped Trajectory Optimization")
    print("=" * 60)
    print(f"Command: '{args.command}'")
    print(f"LLM Provider: {llm_config.llm_provider}")
    print(f"Model: {llm_config.llm_model}")
    print(f"Max Iterations: {args.max_iterations}")

    try:
        # ========================================================
        # Stage 0: Setup
        # ========================================================
        color_print("blue", "Stage 0: System Setup")

        # Initialize robot model
        kinodynamic_model = KinoDynamic_Model(config)

        # Setup initial conditions
        initial_state, _ = conversion.sim_to_mpc(
            config.experiment.initial_qpos, config.experiment.initial_qvel
        )

        # Create reference trajectory (basic)
        ref = create_reference_trajectory(config.experiment.initial_qpos)

        # Get contact sequence (for now, use existing hopping sequence)
        # TODO: Generate contact sequence based on command
        contact_sequence = config.mpc_config.contact_sequence

        color_print("green", "✅ System setup complete")

        # ========================================================
        # Stage 1: LLM-Based Constraint Generation
        # ========================================================
        color_print("blue", "Stage 1: LLM-Based Constraint Generation")

        # Run iterative LLM constraint generation
        success, best_constraint_function, summary = run_llm_constraint_generation(
            command=args.command,
            kinodynamic_model=kinodynamic_model,
            config=config,
            initial_state=initial_state,
            contact_sequence=contact_sequence,
            reference_trajectory=ref,
            max_iterations=args.max_iterations,
        )

        # Print summary results
        print(f"\n{'=' * 50}")
        print("📊 LLM REFINEMENT SUMMARY")
        print(f"{'=' * 50}")
        print(f"Total Iterations: {summary['total_iterations']}")
        print(f"Success Achieved: {'✅ YES' if success else '❌ NO'}")

        if summary.get("best_iteration"):
            print(f"Best Iteration: #{summary['best_iteration']}")
            print(f"Best Score: {summary['best_score']:.3f}")

            if "best_metrics" in summary:
                metrics = summary["best_metrics"]
                print(f"  - Converged: {'✅' if metrics['converged'] else '❌'}")
                print(
                    f"  - Height Clearance: {'✅' if metrics['height_clearance'] else '❌'}"
                )
                print(
                    f"  - Rotation Target: {'✅' if metrics['rotation_target'] else '❌'}"
                )
                print(
                    f"  - Landing Stable: {'✅' if metrics['landing_stable'] else '❌'}"
                )
                print(f"  - Max Violation: {metrics['max_violation']:.4f}")
                print(f"  - Solve Time: {metrics['solve_time']:.2f}s")

        if not best_constraint_function:
            color_print("red", "❌ No valid constraints generated. Using baseline MPC.")

            # Fallback to baseline MPC
            from mpc.mpc_opti import QuadrupedMPCOpti

            mpc = QuadrupedMPCOpti(model=kinodynamic_model, config=config, build=True)

            state_traj, grf_traj, joint_vel_traj, status = mpc.solve_trajectory(
                initial_state, ref, contact_sequence
            )

        else:
            color_print(
                "green", "✅ Using best generated constraints for final optimization"
            )

            # Use the best constraint function for final optimization
            # This would require modifying QuadrupedMPCOpti to accept additional constraints
            # For now, use the trajectory from the best iteration
            state_traj = summary.get("best_state_traj")
            grf_traj = summary.get("best_grf_traj")
            joint_vel_traj = summary.get("best_input_traj", {}).get("joint_vel")

            if state_traj is None:
                color_print(
                    "yellow", "⚠️ No trajectory from LLM process, running baseline"
                )

                from mpc.mpc_opti import QuadrupedMPCOpti

                mpc = QuadrupedMPCOpti(
                    model=kinodynamic_model, config=config, build=True
                )

                state_traj, grf_traj, joint_vel_traj, status = mpc.solve_trajectory(
                    initial_state, ref, contact_sequence
                )

        # ========================================================
        # Stage 2: Trajectory Execution and Visualization
        # ========================================================
        color_print("blue", "Stage 2: Trajectory Execution and Visualization")

        if state_traj is not None:
            # Combine input trajectory
            input_traj = np.concatenate([joint_vel_traj, grf_traj], axis=1)

            # Save trajectory results
            suffix = "_llm"
            save_trajectory_results(
                state_traj, joint_vel_traj, grf_traj, contact_sequence, suffix
            )

            # Render planned trajectory if enabled
            planned_traj_images = None
            if config.experiment.render:
                color_print("yellow", "🎬 Rendering planned trajectory...")
                from gym_quadruped.quadruped_env import QuadrupedEnv

                env = QuadrupedEnv(
                    robot=config.robot,
                    scene="flat",
                    ground_friction_coeff=config.experiment.mu_ground,
                    state_obs_names=(
                        QuadrupedEnv._DEFAULT_OBS + ("contact_forces:base",)
                    ),
                    sim_dt=config.experiment.sim_dt,
                )

                planned_traj_images = render_and_save_planned_trajectory(
                    state_traj, input_traj, env, suffix
                )

            # Compute joint torques and simulate
            color_print("yellow", "⚙️ Computing joint torques and simulating...")
            from utils.inv_dyn import compute_joint_torques

            joint_torques_traj = compute_joint_torques(
                kinodynamic_model,
                state_traj,
                input_traj,
                contact_sequence,
                config.mpc_config.mpc_dt,
            )
            np.save(f"results/joint_torques_traj{suffix}.npy", joint_torques_traj)

            # Simulate trajectory
            if config.experiment.render:
                qpos_traj, qvel_traj, sim_grf_traj, sim_images = simulate_trajectory(
                    env, joint_torques_traj, planned_traj_images
                )

                # Save simulation video
                fps = 1 / config.experiment.sim_dt
                imageio.mimsave(f"results/trajectory{suffix}.mp4", sim_images, fps=fps)

                # Plot comparison
                plot_trajectory_comparison(
                    state_traj,
                    input_traj,
                    qpos_traj,
                    qvel_traj,
                    sim_grf_traj,
                    quantities=config.plot_quantities,
                    mpc_dt=config.mpc_config.mpc_dt,
                    sim_dt=config.experiment.sim_dt,
                    save_path=f"results/trajectory_comparison{suffix}.png",
                    show_plot=False,
                )

            color_print("green", f"✅ Results saved with suffix '{suffix}'")

        else:
            color_print("red", "❌ No valid trajectory generated")

        # ========================================================
        # Stage 3: Results Summary
        # ========================================================
        color_print("blue", "Stage 3: Results Summary")

        print(f"\n🎯 FINAL RESULTS for '{args.command}':")
        print(f"   Command Processing: {'✅ SUCCESS' if success else '❌ PARTIAL'}")
        print(
            f"   Trajectory Generated: {'✅ YES' if state_traj is not None else '❌ NO'}"
        )
        print(
            f"   Videos Rendered: {'✅ YES' if config.experiment.render else '⏭️ SKIPPED'}"
        )

        if success:
            print(
                f"\n🏆 Successfully generated constraints for '{args.command}' in {summary['total_iterations']} iterations!"
            )
        else:
            print(
                f"\n🔄 Generated partial solution after {summary['total_iterations']} iterations."
            )
            print("   Consider adjusting the command or increasing max iterations.")

        print("\n📁 Results saved in: results/")
        print(f"   - Trajectories: results/*{suffix}.npy")
        if config.experiment.render:
            print(f"   - Videos: results/*{suffix}.mp4")
            print(f"   - Plots: results/*{suffix}.png")
        print(f"   - LLM Logs: {llm_config.results_dir}/")

    except KeyboardInterrupt:
        color_print("yellow", "\n⏹️ Process interrupted by user")
        sys.exit(1)

    except Exception as e:
        color_print("red", f"\n💥 Error: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
