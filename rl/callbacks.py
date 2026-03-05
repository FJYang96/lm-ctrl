"""Logging, diagnostics, and plotting for RL training and evaluation.

All logging goes through this module and writes to rl/diagnostics.log.
No SB3 dependency — pure Python + matplotlib.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_PATH = Path("rl/diagnostics.log")

RAW_KEYS = ["pos_error", "ori_error", "joint_error", "action_rate", "max_torque"]
COMPONENT_KEYS = ["rw_pos", "rw_ori", "rw_joint", "rw_smooth", "rw_torque"]
COMPONENT_LABELS = [
    "Position (w=0.3)",
    "Orientation (w=0.3)",
    "Joint (w=0.2)",
    "Smoothness (w=0.1)",
    "Torque (w=0.1)",
]
COMPONENT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def get_log() -> logging.Logger:
    """Get or create the RL diagnostics file logger."""
    logger = logging.getLogger("rl.diagnostics")
    if not logger.handlers:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(LOG_PATH, mode="a")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


def write_training_header_jax(total_timesteps: int, num_envs: int, ref: Any) -> None:
    """Write a training header block for JAX training."""
    log = get_log()
    log.info("")
    log.info("=" * 60)
    log.info("TRAINING: JAX PPO LEARNING LOG")
    log.info("=" * 60)
    log.info(f"Timesteps: {total_timesteps}  Envs: {num_envs}")
    log.info(f"Trajectory: {ref.max_phase} steps, {ref.duration:.2f}s")
    log.info("=" * 60)


# Keep old name for backward compat
write_training_header = write_training_header_jax


def save_reward_curve(
    rewards: np.ndarray, timesteps: np.ndarray, plot_dir: str, total_steps: int
) -> None:
    """Save reward curve plot."""
    if len(rewards) < 2:
        return

    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    window = max(1, min(50, len(rewards) // 4))
    kernel = np.ones(window) / window
    smooth_r = np.convolve(rewards, kernel, mode="valid")
    smooth_s = timesteps[window - 1:]

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(smooth_s, smooth_r, linewidth=0.8)
    ax.set_ylabel("Episode Return")
    ax.set_xlabel("Timesteps")
    ax.set_title(f"Training Progress ({total_steps:,} steps)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path / "reward_curve.png", dpi=100)
    plt.close(fig)


def save_component_plot(
    component_history: dict[str, list[float]],
    timesteps: np.ndarray,
    plot_dir: str,
    total_steps: int,
) -> None:
    """Save per-component reward breakdown plot."""
    if len(timesteps) < 2:
        return

    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    window = max(1, min(50, len(timesteps) // 4))
    kernel = np.ones(window) / window
    smooth_s = timesteps[window - 1:]

    fig, axes = plt.subplots(len(COMPONENT_KEYS), 1, figsize=(10, 12), sharex=True)
    fig.suptitle(
        f"Raw Reward Components ({total_steps:,} steps)", fontsize=13
    )

    for ax, key, label, color in zip(
        axes, COMPONENT_KEYS, COMPONENT_LABELS, COMPONENT_COLORS, strict=False
    ):
        vals = np.array(component_history.get(key, []))
        if len(vals) >= window:
            smooth_v = np.convolve(vals, kernel, mode="valid")
            ax.plot(smooth_s[:len(smooth_v)], smooth_v, linewidth=0.8, color=color)
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(-0.01, 1.02)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timesteps")
    plt.tight_layout()
    plt.savefig(plot_path / "reward_components.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# EVALUATION: post-training termination diagnostics
# ---------------------------------------------------------------------------


class RewardTracker:
    """Track per-episode rewards and components for logging."""

    def __init__(self):
        self.ep_rewards: list[float] = []
        self.ep_lengths: list[int] = []
        self.component_history: dict[str, list[float]] = {k: [] for k in COMPONENT_KEYS}

    def log_episode(self, reward: float, length: int, components: dict[str, float] | None = None) -> None:
        self.ep_rewards.append(reward)
        self.ep_lengths.append(length)
        if components:
            for k in COMPONENT_KEYS:
                if k in components:
                    self.component_history[k].append(components[k])


def diagnose_termination(
    state_traj: np.ndarray,
    grf_traj: np.ndarray,
    joint_vel_traj: np.ndarray,
    kindyn: Any,
    params: Any,
    apply_fn: Any,
    normalizer: Any = None,
) -> None:
    """Run one episode in the CPU tracking env and write per-step diagnostics."""
    import config as cfg

    from .feedforward import FeedforwardComputer
    from .reference import ReferenceTrajectory
    from .tracking_env import Go2TrackingEnv

    log = get_log()

    control_dt = cfg.mpc_config.mpc_dt
    sim_dt = 0.001

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

    log.info("")
    log.info("=" * 60)
    log.info("EVALUATION: TERMINATION DIAGNOSTICS")
    log.info("(single episode, no randomization)")
    log.info("=" * 60)
    log.info(f"Trajectory: {ref.max_phase} steps, {ref.duration:.2f}s")
    log.info(
        f"Thresholds: pos>{2.5 * env.SIGMA_POS:.3f}m  ori>{2.5 * env.SIGMA_ORI:.3f}rad  "
        f"joint>{2.5 * env.SIGMA_JOINT:.3f}rad  smooth>{2.5 * env.SIGMA_SMOOTH:.3f}rad  "
        f"torque>{2.5 * env.SIGMA_TORQUE:.1f}Nm"
    )
    log.info("-" * 60)
    log.info(
        f"{'step':>5}  {'reward':>7}  {'pos':>7}  {'ori':>7}  "
        f"{'joint':>7}  {'smooth':>7}  {'torque':>7}"
    )
    log.info("-" * 60)

    import jax.numpy as jnp

    from .ppo import normalize_obs as norm_obs

    for step_idx in range(ref.max_phase):
        obs_jnp = jnp.array(obs)
        if normalizer is not None:
            obs_jnp = norm_obs(normalizer, obs_jnp)

        mean, _, _ = apply_fn(params, obs_jnp)
        action = np.array(mean)

        obs, reward, terminated, truncated, info = env.step(action)

        log.info(
            f"{step_idx:5d}  {reward:7.4f}  {info['pos_error']:7.4f}  "
            f"{info['ori_error']:7.4f}  {info['joint_error']:7.4f}  "
            f"{info['action_rate']:7.4f}  {info['max_torque']:7.1f}"
        )

        if terminated:
            log.info("")
            log.info(
                f">>> TERMINATED at step {step_idx}: "
                f"{info.get('termination_reason', 'unknown')}"
            )
            break
        if truncated:
            log.info("")
            log.info(f">>> Completed full trajectory ({step_idx + 1} steps)")
            break

    env.close()
    log.info("=" * 60)
    print(f"Diagnostics written to {LOG_PATH}")
