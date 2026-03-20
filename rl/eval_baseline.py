"""Evaluate PD+feedforward baseline (zero RL residuals) for tracking comparison.

Usage:
    MUJOCO_GL=egl python -m rl.eval_baseline \
        --state-traj results/state_traj.npy \
        --grf-traj results/grf_traj.npy \
        --joint-vel-traj results/joint_vel_traj.npy \
        --output-video results/baseline_tracking.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio
import numpy as np

import go2_config as config
from mpc.dynamics.model import KinoDynamic_Model
from utils.conversion import sim_to_mpc

from .feedforward import FeedforwardComputer
from .reference import ReferenceTrajectory
from .tracking_env import Go2TrackingEnv


def compute_tracking_error(
    planned_state: np.ndarray, qpos_traj: np.ndarray, qvel_traj: np.ndarray
) -> dict[str, float]:
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


def run_baseline(args: argparse.Namespace) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")

    state_traj = np.load(args.state_traj)
    grf_traj = np.load(args.grf_traj)
    joint_vel_traj = np.load(args.joint_vel_traj)

    kindyn = KinoDynamic_Model()

    sim_dt = 0.001
    control_dt = config.mpc_config.mpc_dt

    ref = ReferenceTrajectory(
        state_traj=state_traj,
        joint_vel_traj=joint_vel_traj,
        grf_traj=grf_traj,
        control_dt=control_dt,
    )
    ff = FeedforwardComputer(kindyn)
    ref.set_feedforward(ff.precompute_trajectory(ref))

    env = Go2TrackingEnv(ref=ref, sim_dt=sim_dt, control_dt=control_dt, randomize=False)
    obs = env.reset()

    import mujoco

    renderer = None
    try:
        renderer = mujoco.Renderer(env.mjModel, height=480, width=640)
    except Exception:
        print("Warning: Could not create renderer, skipping video capture")

    zero_action = np.zeros(12)
    num_steps = ref.max_phase

    qpos_traj_out = []
    qvel_traj_out = []
    images = []
    rewards = []
    reward_components = {
        "rw_pos": [],
        "rw_ori": [],
        "rw_joint": [],
        "rw_smooth": [],
        "rw_torque": [],
    }

    print(f"Running PD+feedforward baseline for {num_steps} steps (zero residuals)...")

    for _phase in range(num_steps):
        obs, reward, terminated, truncated, info = env.step(zero_action)

        sim_obs = env._quad_env._get_obs()
        qpos_traj_out.append(sim_obs["qpos"].copy())
        qvel_traj_out.append(sim_obs["qvel"].copy())
        rewards.append(reward)
        for k in reward_components:
            reward_components[k].append(info.get(k, 0.0))

        if renderer is not None:
            try:
                renderer.update_scene(env.mjData)
                images.append(renderer.render())
            except Exception:
                pass

        if terminated or truncated:
            break

    if renderer is not None:
        try:
            renderer.close()
        except Exception:
            pass
    env.close()

    n_tracked = len(qpos_traj_out)
    qpos_arr = np.array(qpos_traj_out) if qpos_traj_out else np.zeros((0, 19))
    qvel_arr = np.array(qvel_traj_out) if qvel_traj_out else np.zeros((0, 18))

    err = (
        compute_tracking_error(state_traj, qpos_arr, qvel_arr)
        if n_tracked > 0
        else {
            "pos_rms": float("nan"),
            "ori_rms": float("nan"),
            "joint_rms": float("nan"),
        }
    )

    print()
    print("=" * 50)
    print("BASELINE (PD + Feedforward, zero RL residuals)")
    print("=" * 50)
    print(f"  Steps tracked:   {n_tracked}/{num_steps}")
    print(f"  Mean reward:     {np.mean(rewards):.4f}")
    print(f"  Position RMS:    {err['pos_rms']:.4f} m")
    print(f"  Orientation RMS: {err['ori_rms']:.4f} rad")
    print(f"  Joint RMS:       {err['joint_rms']:.4f} rad")
    print()
    print("  Reward components (mean):")
    for k, v in reward_components.items():
        print(f"    {k:12s}: {np.mean(v):.4f}")
    print("=" * 50)

    # Save video
    if images:
        output_path = Path(args.output_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fps = int(1.0 / control_dt)
            imageio.mimsave(str(output_path), images, fps=fps)
            print(f"Video saved to {output_path}")
        except Exception as e:
            print(f"Failed to save video: {e}")
    else:
        print("No frames captured -- video not saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate PD+feedforward baseline (no RL)"
    )
    parser.add_argument("--state-traj", type=str, required=True)
    parser.add_argument("--grf-traj", type=str, required=True)
    parser.add_argument("--joint-vel-traj", type=str, required=True)
    parser.add_argument(
        "--output-video", type=str, default="results/baseline_tracking.mp4"
    )
    args = parser.parse_args()
    run_baseline(args)
