"""Evaluate a trained JAX tracking policy: compute tracking errors and save video.

Usage:
    python -m rl.evaluate \
        --model-path rl/trained_models/best_model \
        --state-traj results/state_traj.npy \
        --grf-traj results/grf_traj.npy \
        --joint-vel-traj results/joint_vel_traj.npy
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio
import jax.numpy as jnp
import numpy as np

import config
from mpc.dynamics.model import KinoDynamic_Model
from utils.conversion import sim_to_mpc

from .callbacks import diagnose_termination
from .ppo import ActorCritic, NormalizerState, load_checkpoint
from .rollout import execute_policy_rollout


def compute_tracking_error(
    planned_state: np.ndarray, qpos_traj: np.ndarray, qvel_traj: np.ndarray
) -> dict[str, float]:
    """RMS tracking error between planned and simulated trajectories."""
    sim_states = []
    for i in range(min(len(qpos_traj), planned_state.shape[0])):
        sim_state, _ = sim_to_mpc(qpos_traj[i], qvel_traj[i])
        sim_states.append(sim_state)

    sim_traj = np.array(sim_states)
    n = min(planned_state.shape[0], sim_traj.shape[0])

    pos_err = np.sqrt(np.mean((planned_state[:n, 0:3] - sim_traj[:n, 0:3]) ** 2))
    ori_err = np.sqrt(np.mean((planned_state[:n, 6:9] - sim_traj[:n, 6:9]) ** 2))
    joint_err = np.sqrt(np.mean((planned_state[:n, 12:24] - sim_traj[:n, 12:24]) ** 2))

    return {"pos_rms": pos_err, "ori_rms": ori_err, "joint_rms": joint_err}


def evaluate(args: argparse.Namespace) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")

    state_traj = np.load(args.state_traj)
    grf_traj = np.load(args.grf_traj)
    joint_vel_traj = np.load(args.joint_vel_traj)

    kindyn = KinoDynamic_Model(config)

    # Load JAX policy
    print(f"Loading JAX policy from {args.model_path}...")
    params, normalizer, step_count = load_checkpoint(args.model_path)
    network = ActorCritic(action_dim=12)
    apply_fn = network.apply
    print(f"  Loaded checkpoint at step {step_count}")

    # Diagnostic: termination analysis
    diagnose_termination(
        state_traj, grf_traj, joint_vel_traj,
        kindyn, params, apply_fn, normalizer=normalizer,
    )

    # Full rollout
    config.experiment.render = True
    qpos_rl, qvel_rl, grf_rl, images = execute_policy_rollout(
        state_traj, grf_traj, joint_vel_traj,
        kindyn, params, apply_fn, normalizer=normalizer, render=True,
    )

    # Tracking errors
    n_tracked = len(qpos_rl)
    err = compute_tracking_error(state_traj, qpos_rl, qvel_rl) if n_tracked > 0 else {
        "pos_rms": float("nan"), "ori_rms": float("nan"), "joint_rms": float("nan")
    }

    print("\n" + "=" * 40)
    print("RL TRACKING ERRORS")
    print("=" * 40)
    print(f"  Steps tracked: {n_tracked}/{state_traj.shape[0] - 1}")
    print(f"  Position RMS:  {err['pos_rms']:.4f} m")
    print(f"  Orientation RMS: {err['ori_rms']:.4f} rad")
    print(f"  Joint RMS:     {err['joint_rms']:.4f} rad")
    print("=" * 40)

    # Save video
    if images:
        output_path = Path(args.output_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fps = int(1.0 / config.mpc_config.mpc_dt)
            imageio.mimsave(str(output_path), images, fps=fps)
            print(f"\nVideo saved to {output_path}")
        except Exception as e:
            print(f"\nFailed to save video: {e}")
    else:
        print("\nNo frames captured -- video not saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate JAX RL tracking policy")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--state-traj", type=str, default="results/state_traj.npy")
    parser.add_argument("--grf-traj", type=str, default="results/grf_traj.npy")
    parser.add_argument(
        "--joint-vel-traj", type=str, default="results/joint_vel_traj.npy"
    )
    parser.add_argument("--output-video", type=str, default="results/rl_tracking.mp4")
    args = parser.parse_args()
    evaluate(args)
