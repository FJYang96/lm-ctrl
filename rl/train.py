"""Train a per-trajectory Go2 tracking policy using PPO (OPT-Mimic).

Usage:
    python -m rl.train \
        --state-traj results/state_traj.npy \
        --grf-traj results/grf_traj.npy \
        --joint-vel-traj results/joint_vel_traj.npy \
        --output-dir rl/trained_models \
        --total-timesteps 2000000 \
        --num-envs 16
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import config
from mpc.dynamics.model import KinoDynamic_Model

from .feedforward import FeedforwardComputer
from .reference import ReferenceTrajectory
from .tracking_env import Go2TrackingEnv


def build_reference(
    state_traj_path: str,
    grf_traj_path: str,
    joint_vel_traj_path: str,
    contact_sequence_path: str | None = None,
    control_dt: float = 0.02,
) -> ReferenceTrajectory:
    """Load MPC trajectory and precompute feedforward torques."""
    ref = ReferenceTrajectory.from_files(
        state_traj_path,
        joint_vel_traj_path,
        grf_traj_path,
        contact_sequence_path=contact_sequence_path,
        control_dt=control_dt,
    )
    kindyn = KinoDynamic_Model(config)
    ff = FeedforwardComputer(kindyn)
    ref.set_feedforward(ff.precompute_trajectory(ref))
    return ref


def make_env(
    ref: ReferenceTrajectory, sim_dt: float, control_dt: float, randomize: bool
) -> Callable[[], Go2TrackingEnv]:
    """Factory for SubprocVecEnv — each subprocess gets its own env."""

    def _init() -> Go2TrackingEnv:
        return Go2TrackingEnv(
            ref=ref, sim_dt=sim_dt, control_dt=control_dt, randomize=randomize
        )

    return _init


def train(args: argparse.Namespace) -> None:
    ref = build_reference(
        args.state_traj,
        args.grf_traj,
        args.joint_vel_traj,
        contact_sequence_path=args.contact_sequence,
        control_dt=args.control_dt,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training envs with domain randomization
    train_env = SubprocVecEnv(
        [
            make_env(ref, args.sim_dt, args.control_dt, randomize=True)
            for _ in range(args.num_envs)
        ]
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Eval env without randomization
    eval_env = SubprocVecEnv(
        [make_env(ref, args.sim_dt, args.control_dt, randomize=False)]
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # OPT-Mimic LR schedule: ×0.99 per update (exponential decay)
    samples_per_update = 20000
    n_updates = args.total_timesteps // samples_per_update

    def lr_schedule(progress_remaining: float) -> float:
        update_idx = n_updates * (1.0 - progress_remaining)
        return float(1e-3 * (0.99**update_idx))

    # PPO — matches OPT-Mimic as closely as SB3 allows
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=lr_schedule,
        n_steps=20000 // args.num_envs,  # ~20000 samples per update (OPT-Mimic)
        batch_size=5000,  # OPT-Mimic: max_samples / 4
        n_epochs=40,  # OPT-Mimic: 40 actor + 40 critic
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=str(output_dir / "tb_logs"),
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[512, 512]),
            activation_fn=torch.nn.ReLU,
            ortho_init=True,
        ),
    )

    callbacks = [
        CheckpointCallback(
            save_freq=50_000,
            save_path=str(output_dir / "checkpoints"),
            name_prefix="tracking_policy",
        ),
        EvalCallback(
            eval_env=eval_env,
            n_eval_episodes=10,
            eval_freq=25_000,
            best_model_save_path=str(output_dir / "best_model"),
            deterministic=True,
        ),
    ]

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    model.save(str(output_dir / "tracking_policy_final"))

    # Save VecNormalize stats for deployment
    train_env.save(str(output_dir / "vec_normalize.pkl"))

    train_env.close()
    eval_env.close()
    print(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OPT-Mimic tracking policy")
    parser.add_argument("--state-traj", type=str, required=True)
    parser.add_argument("--grf-traj", type=str, required=True)
    parser.add_argument("--joint-vel-traj", type=str, required=True)
    parser.add_argument("--contact-sequence", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="rl/trained_models")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--sim-dt", type=float, default=0.001)
    parser.add_argument("--control-dt", type=float, default=0.02)
    args = parser.parse_args()
    train(args)
