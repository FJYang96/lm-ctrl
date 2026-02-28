"""Logging, diagnostics, and plotting callbacks for RL training and evaluation.

All logging goes through this module and writes to rl/diagnostics.log.

Log sections:
  [TRAINING]   — header, per-update PPO stats, reward plots
  [EVALUATION] — per-step termination diagnostics after training
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

LOG_PATH = Path("rl/diagnostics.log")

# Raw error keys (used for termination diagnostics)
RAW_KEYS = ["pos_error", "ori_error", "joint_error", "action_rate", "max_torque"]

# Weighted reward component keys (actual contribution to total reward)
COMPONENT_KEYS = ["rw_pos", "rw_ori", "rw_joint", "rw_smooth", "rw_torque"]
COMPONENT_LABELS = [
    "Position (w=0.4)",
    "Orientation (w=0.2)",
    "Joint (w=0.2)",
    "Smoothness (w=0.1)",
    "Torque (w=0.1)",
]
COMPONENT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def get_log() -> logging.Logger:
    """Get or create the RL diagnostics file logger (appends to rl/diagnostics.log)."""
    logger = logging.getLogger("rl.diagnostics")
    if not logger.handlers:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(LOG_PATH, mode="a")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
    return logger


def write_training_header(total_timesteps: int, num_envs: int, ref: Any) -> None:
    """Write a training header block to the diagnostics log."""
    log = get_log()
    log.info("")
    log.info("=" * 60)
    log.info("TRAINING: PPO LEARNING LOG")
    log.info("=" * 60)
    log.info(f"Timesteps: {total_timesteps}  Envs: {num_envs}")
    log.info(f"Trajectory: {ref.max_phase} steps, {ref.duration:.2f}s")
    log.info("=" * 60)


class VecNormalizeSaveCallback(BaseCallback):  # type: ignore[misc]
    """Periodically save VecNormalize stats so interrupted training is usable."""

    def __init__(self, save_path: str, save_freq: int = 25_000, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self._last_save = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self.save_freq:
            self.training_env.save(self.save_path)
            self._last_save = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        self.training_env.save(self.save_path)


class TrainingLogCallback(BaseCallback):  # type: ignore[misc]
    """Log SB3 training stats to rl/diagnostics.log every PPO update."""

    def __init__(self, log_freq: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._last_update = -1

    def _on_step(self) -> bool:
        # SB3 increments n_updates after each PPO update cycle
        n_updates = self.model._n_updates
        if n_updates == self._last_update:
            return True
        self._last_update = n_updates

        if n_updates % self.log_freq != 0:
            return True

        log = get_log()
        ts = self.num_timesteps

        # Collect key metrics from SB3's logger
        parts = [f"[step {ts:>9,}  update {n_updates:>5}]"]

        # Eval reward (from EvalCallback, if available)
        if self.locals.get("infos"):
            recent_rewards = []
            for info in self.locals["infos"]:
                ep = info.get("episode")
                if ep is not None:
                    recent_rewards.append(ep["r"])
            if recent_rewards:
                parts.append(f"ep_reward={np.mean(recent_rewards):.4f}")

        # Training metrics from the logger's name_to_value dict
        logger_dict: dict[str, Any] = {}
        if hasattr(self.model, "logger") and self.model.logger is not None:
            logger_dict = getattr(self.model.logger, "name_to_value", {})

        key_metrics = [
            ("train/policy_gradient_loss", "pg_loss"),
            ("train/value_loss", "vf_loss"),
            ("train/approx_kl", "kl"),
            ("train/clip_fraction", "clip"),
            ("train/entropy_loss", "entropy"),
            ("train/explained_variance", "expl_var"),
            ("train/std", "std"),
            ("train/learning_rate", "lr"),
        ]
        for sb3_key, short_name in key_metrics:
            val = logger_dict.get(sb3_key)
            if val is not None:
                parts.append(f"{short_name}={val:.5g}")

        log.info("  ".join(parts))
        return True


class RewardPlotCallback(BaseCallback):  # type: ignore[misc]
    """Periodically save reward curves and per-component breakdown plots.

    Saves to plot_dir:
      - reward_curve.png: episode reward + length over training
      - reward_components.png: per-component error trends over training

    Component data comes from the info dict returned by Go2TrackingEnv.step().
    """

    def __init__(self, plot_dir: str, plot_freq: int = 100_000, verbose: int = 0):
        super().__init__(verbose)
        self.plot_dir = Path(plot_dir)
        self.plot_freq = plot_freq

        # Episode-level tracking
        self.ep_rewards: list[float] = []
        self.ep_lengths: list[int] = []
        self.ep_timesteps: list[int] = []

        # Per-component tracking (mean over each episode)
        self.component_history: dict[str, list[float]] = {k: [] for k in COMPONENT_KEYS}
        self.component_timesteps: list[int] = []

        # Accumulator for current episode (per env): reward sum, length, components
        self._ep_reward: dict[int, float] = {}
        self._ep_len: dict[int, int] = {}
        self._ep_components: dict[int, dict[str, list[float]]] = {}
        self._last_plot_step: int = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])

        for env_idx, info in enumerate(infos):
            # Init accumulators for new episodes
            if env_idx not in self._ep_reward:
                self._ep_reward[env_idx] = 0.0
                self._ep_len[env_idx] = 0
                self._ep_components[env_idx] = {k: [] for k in COMPONENT_KEYS}

            # Accumulate per-step data
            if env_idx < len(rewards):
                self._ep_reward[env_idx] += float(rewards[env_idx])
            self._ep_len[env_idx] += 1
            for key in COMPONENT_KEYS:
                if key in info:
                    self._ep_components[env_idx][key].append(info[key])

            # Episode finished (terminated or truncated)
            done = dones[env_idx] if env_idx < len(dones) else False
            if done:
                self.ep_rewards.append(self._ep_reward[env_idx])
                self.ep_lengths.append(self._ep_len[env_idx])
                self.ep_timesteps.append(self.num_timesteps)

                if self._ep_components.get(env_idx):
                    for key in COMPONENT_KEYS:
                        vals = self._ep_components[env_idx][key]
                        mean_val = np.mean(vals) if vals else 0.0
                        self.component_history[key].append(float(mean_val))
                    self.component_timesteps.append(self.num_timesteps)

                # Reset accumulators for next episode
                self._ep_reward[env_idx] = 0.0
                self._ep_len[env_idx] = 0
                self._ep_components[env_idx] = {k: [] for k in COMPONENT_KEYS}

        # Save plots periodically
        if self.num_timesteps - self._last_plot_step >= self.plot_freq:
            self._save_plots()
            self._last_plot_step = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        self._save_plots()

    def _save_plots(self) -> None:
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self._save_reward_curve()
        self._save_component_plot()

    def _save_reward_curve(self) -> None:
        if len(self.ep_rewards) < 2:
            return

        rewards = np.array(self.ep_rewards)
        lengths = np.array(self.ep_lengths)
        steps = np.array(self.ep_timesteps)

        window = max(1, min(100, len(rewards) // 4))
        kernel = np.ones(window) / window
        smooth_r = np.convolve(rewards, kernel, mode="valid")
        smooth_l = np.convolve(lengths, kernel, mode="valid")
        smooth_s = steps[window - 1 :]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        ax1.plot(smooth_s, smooth_r, linewidth=0.8)
        ax1.set_ylabel("Episode Reward")
        ax1.set_title(f"Training Progress ({self.num_timesteps:,} steps)")
        ax1.grid(True, alpha=0.3)

        ax2.plot(smooth_s, smooth_l, linewidth=0.8, color="tab:orange")
        ax2.set_ylabel("Episode Length (steps)")
        ax2.set_xlabel("Timesteps")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plot_dir / "reward_curve.png", dpi=100)
        plt.close(fig)

    def _save_component_plot(self) -> None:
        if len(self.component_timesteps) < 2:
            return

        steps = np.array(self.component_timesteps)
        window = max(1, min(100, len(steps) // 4))
        kernel = np.ones(window) / window
        smooth_s = steps[window - 1 :]

        fig, axes = plt.subplots(len(COMPONENT_KEYS), 1, figsize=(10, 12), sharex=True)
        fig.suptitle(
            f"Raw Reward Components (mean per episode, {self.num_timesteps:,} steps)",
            fontsize=13,
        )

        for ax, key, label, color in zip(
            axes, COMPONENT_KEYS, COMPONENT_LABELS, COMPONENT_COLORS, strict=False
        ):
            vals = np.array(self.component_history[key])
            smooth_v = np.convolve(vals, kernel, mode="valid")
            ax.plot(smooth_s, smooth_v, linewidth=0.8, color=color)
            ax.set_ylabel(label, fontsize=9)
            ax.set_ylim(-0.01, 1.02)  # raw Gaussian rewards are in [0, 1]
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Timesteps")
        plt.tight_layout()
        plt.savefig(self.plot_dir / "reward_components.png", dpi=100)
        plt.close(fig)


# ---------------------------------------------------------------------------
# EVALUATION: post-training termination diagnostics
# ---------------------------------------------------------------------------


def diagnose_termination(
    state_traj: np.ndarray,
    grf_traj: np.ndarray,
    joint_vel_traj: np.ndarray,
    kindyn: Any,
    policy: Any,
    vec_normalize_path: str | None = None,
) -> None:
    """Run one episode in the tracking env and write per-step diagnostics.

    Written under an [EVALUATION] header in rl/diagnostics.log.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    import config

    from .feedforward import FeedforwardComputer
    from .reference import ReferenceTrajectory
    from .tracking_env import Go2TrackingEnv

    log = get_log()

    control_dt = config.mpc_config.mpc_dt
    sim_dt = 0.001  # 1kHz physics, matching RL training (OPT-Mimic)

    ref = ReferenceTrajectory(
        state_traj=state_traj,
        joint_vel_traj=joint_vel_traj,
        grf_traj=grf_traj,
        control_dt=control_dt,
    )
    ff = FeedforwardComputer(kindyn)
    ref.set_feedforward(ff.precompute_trajectory(ref))

    env = Go2TrackingEnv(ref=ref, sim_dt=sim_dt, control_dt=control_dt, randomize=False)

    # Load normalizer if available
    normalizer = None
    if vec_normalize_path and Path(vec_normalize_path).exists():
        dummy = DummyVecEnv(
            [
                lambda: Go2TrackingEnv(
                    ref=ref, sim_dt=sim_dt, control_dt=control_dt, randomize=False
                )
            ]
        )
        normalizer = VecNormalize.load(vec_normalize_path, dummy)
        normalizer.training = False
        normalizer.norm_reward = False

    obs, _ = env.reset()

    log.info("")
    log.info("=" * 60)
    log.info("EVALUATION: TERMINATION DIAGNOSTICS")
    log.info("(single episode, no randomization)")
    log.info("=" * 60)
    log.info(f"Trajectory: {ref.max_phase} steps, {ref.duration:.2f}s")
    log.info(
        f"Thresholds: pos>{2.5*env.SIGMA_POS:.3f}m  ori>{2.5*env.SIGMA_ORI:.3f}rad  "
        f"joint>{2.5*env.SIGMA_JOINT:.3f}rad  smooth>{2.5*env.SIGMA_SMOOTH:.3f}rad  "
        f"torque>{2.5*env.SIGMA_TORQUE:.1f}Nm"
    )
    log.info("-" * 60)
    log.info(
        f"{'step':>5}  {'reward':>7}  {'pos':>7}  {'ori':>7}  "
        f"{'joint':>7}  {'smooth':>7}  {'torque':>7}"
    )
    log.info("-" * 60)

    for step in range(ref.max_phase):
        if normalizer is not None:
            obs_norm = normalizer.normalize_obs(obs)
        else:
            obs_norm = obs
        action, _ = policy.predict(obs_norm, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        log.info(
            f"{step:5d}  {reward:7.4f}  {info['pos_error']:7.4f}  "
            f"{info['ori_error']:7.4f}  {info['joint_error']:7.4f}  "
            f"{info['action_rate']:7.4f}  {info['max_torque']:7.1f}"
        )

        if terminated:
            log.info("")
            log.info(
                f">>> TERMINATED at step {step}: "
                f"{info.get('termination_reason', 'unknown')}"
            )
            break
        if truncated:
            log.info("")
            log.info(f">>> Completed full trajectory ({step+1} steps)")
            break

    if normalizer is not None:
        normalizer.venv.close()
    log.info("=" * 60)
    print(f"Diagnostics written to {LOG_PATH}")
