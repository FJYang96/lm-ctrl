#!/usr/bin/env python3
"""
LLM-Enhanced Quadruped Control - Main Entry Point

This script implements the LLM feedback pipeline described in the research PDF,
allowing users to generate quadruped behaviors through natural language commands.

Usage:
    python llm_main.py "do a backflip"
    python llm_main.py "jump forward"
    python llm_main.py "perform a front flip"

The system will:
1. Generate optimization constraints using an LLM (Claude)
2. Solve trajectory optimization with those constraints
3. Simulate the resulting trajectory
4. Collect feedback and iterate to improve results

Requirements:
- Set ANTHROPIC_API_KEY in .env file
- Ensure all dependencies are installed
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def main() -> int:
    """Main entry point for LLM-enhanced quadruped control."""

    parser = argparse.ArgumentParser(
        description="LLM-Enhanced Quadruped Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llm_main.py "do a backflip"
  python llm_main.py "jump as high as possible"
  python llm_main.py "perform a front flip" --max-iterations 10
  python llm_main.py "spin in a circle" --no-slack
  python llm_main.py "jump forward" --max-iterations 5
        """,
    )

    parser.add_argument(
        "command", type=str, help="Natural language command for the quadruped robot"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum number of LLM iterations (default: 20)",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/llm_iterations",
        help="Directory to save results (default: results/llm_iterations)",
    )

    parser.add_argument(
        "--config",
        type=str,
        choices=["standard", "complementarity"],
        default="complementarity",
        help="MPC configuration mode (default: complementarity)",
    )

    parser.add_argument(
        "--no-slack",
        action="store_true",
        default=False,
        help="Disable slack formulation (use hard constraints instead)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Validate command
    if not args.command.strip():
        print("Error: Please provide a non-empty command")
        return 1

    # Set environment variables from arguments
    os.environ["MAX_LLM_ITERATIONS"] = str(args.max_iterations)
    os.environ["RESULTS_DIR"] = args.results_dir

    # Check for API key
    if not check_api_key():
        return 1

    print("=" * 60)
    print("LLM-Enhanced Quadruped Control Pipeline")
    print("=" * 60)
    print(f"Command: '{args.command}'")
    print(f"Max LLM iterations: {args.max_iterations}")
    print(f"Slack formulation: {'disabled' if args.no_slack else 'enabled'}")
    print(f"Config mode: {args.config}")
    print(f"Results directory: {args.results_dir}")
    print()

    try:
        # Import and run pipeline
        print("Loading pipeline components...")
        import config
        from llm_integration import FeedbackPipeline

        # Update config for the selected mode
        config.CONSTRAINT_MODE = args.config

        # Initialize and run pipeline
        use_slack = not args.no_slack
        print("Initializing feedback pipeline...")
        pipeline = FeedbackPipeline(config, use_slack=use_slack)

        start_time = time.time()
        results = pipeline.run_pipeline(args.command)
        end_time = time.time()

        # Print summary results
        print_results_summary(results, end_time - start_time)

        # Return appropriate exit code
        return 0 if results["pipeline_success"] else 1

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        return 130  # Standard exit code for SIGINT

    except ImportError as e:
        print(f"\nImport Error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements_llm.txt")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    except Exception as e:
        print(f"\nError running pipeline: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def check_api_key() -> bool:
    """Check if the required API key is configured."""
    try:
        from dotenv import load_dotenv

        load_dotenv()

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            print("Error: ANTHROPIC_API_KEY not configured")
            print()
            print("Please set your Anthropic API key in the .env file:")
            print("1. Edit the .env file in this directory")
            print("2. Replace 'your_api_key_here' with your actual API key")
            print()
            print("Get an API key from: https://console.anthropic.com/")
            return False

        return True

    except ImportError:
        print("Error: Required package 'python-dotenv' not installed")
        print("Install with: pip install python-dotenv")
        return False


def print_results_summary(results: dict[str, Any], elapsed_time: float) -> None:
    """Print a summary of the pipeline results."""

    print("\\n" + "=" * 60)
    print("PIPELINE RESULTS SUMMARY")
    print("=" * 60)

    command = results["command"]
    total_iterations = results["total_iterations"]
    best_score = results["best_score"]
    pipeline_success = results["pipeline_success"]

    print(f"Command: {command}")
    print(f"Total iterations: {total_iterations}")
    print(f"Best score achieved: {best_score:.3f}")
    print(f"Pipeline success: {'YES' if pipeline_success else 'NO'}")
    print(f"Total time: {elapsed_time:.1f} seconds")

    # Print best iteration details if available
    best_iteration = results.get("best_iteration")
    if best_iteration:
        print()
        print("BEST ITERATION DETAILS:")
        print(f"- Iteration: {best_iteration['iteration']}")

        opt_result = best_iteration.get("optimization", {})
        print(f"- Optimization: {'SUCCESS' if opt_result.get('success') else 'FAILED'}")

        sim_result = best_iteration.get("simulation", {})
        print(f"- Simulation: {'SUCCESS' if sim_result.get('success') else 'FAILED'}")

        if sim_result.get("tracking_error") is not None:
            print(f"- Tracking error: {sim_result['tracking_error']:.3f}")

        # Print trajectory metrics if available
        traj_analysis = opt_result.get("trajectory_analysis", {})
        if traj_analysis:
            print(f"- Max height: {traj_analysis.get('max_com_height', 'N/A'):.3f}m")
            if "total_pitch_rotation" in traj_analysis:
                rotation_deg = traj_analysis["total_pitch_rotation"] * 180 / 3.14159
                print(f"- Pitch rotation: {rotation_deg:.1f} degrees")

    print()
    print(f"Results saved to: {results['results_directory']}")

    # Print final recommendation
    print()
    if pipeline_success:
        print("✓ Pipeline completed successfully!")
        print("  Check the results directory for trajectory videos and detailed logs.")
    else:
        print("⚠ Pipeline completed with limited success.")
        print("  Review the iteration logs to understand optimization challenges.")
        print("  Consider:")
        print("  - Adjusting the command to be more specific")
        print("  - Running additional iterations")
        print("  - Checking for constraint conflicts")


if __name__ == "__main__":
    sys.exit(main())
