"""Train OPT-Mimic tracking policy using Isaac Lab + RSL-RL.

Faithful port of rl/train.py from JAX/MJX to Isaac Lab/PyTorch.

Usage (inside Docker):
    /workspace/isaaclab/isaaclab.sh -p -m rl_isaac.train \
        --state-traj results/llm_iterations/.../state_traj_iter_7.npy \
        --grf-traj results/llm_iterations/.../grf_traj_iter_7.npy \
        --joint-vel-traj results/llm_iterations/.../joint_vel_traj_iter_7.npy \
        --num-envs 4096 --total-timesteps 50000000 --headless
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

# --- Isaac Lab AppLauncher must be called BEFORE any other isaaclab imports ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train OPT-Mimic tracking policy (Isaac Lab)")
parser.add_argument("--state-traj", type=str, required=True)
parser.add_argument("--grf-traj", type=str, required=True)
parser.add_argument("--joint-vel-traj", type=str, required=True)
parser.add_argument("--contact-sequence", type=str, default="")
parser.add_argument("--output-dir", type=str, default="rl_isaac/trained_models")
parser.add_argument("--total-timesteps", type=int, default=50_000_000)
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--n-epochs", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Now safe to import isaaclab and other modules ---
import os
import sys
import types

# Block broken GLFW and mujoco.viewer so gym_quadruped doesn't crash on headless Docker.
# We only need mujoco.Renderer (EGL), not the GLFW-based viewer.
os.environ.setdefault("MUJOCO_GL", "egl")
if "mujoco.viewer" not in sys.modules:
    _fake_viewer = types.ModuleType("mujoco.viewer")
    _fake_viewer.Handle = type("Handle", (), {})  # gym_quadruped.utils.mujoco.visual imports this
    sys.modules["mujoco.viewer"] = _fake_viewer
if "glfw" not in sys.modules:
    _fake_glfw = types.ModuleType("glfw")
    _fake_glfw._glfw = True
    sys.modules["glfw"] = _fake_glfw
    sys.modules["glfw.library"] = types.ModuleType("glfw.library")

import numpy as np
import torch

from rsl_rl.modules import EmpiricalNormalization

from rl_isaac.env_cfg import Go2TrackingEnvCfg
from rl_isaac.tracking_env import Go2TrackingEnv
from rl_isaac.network import OPTMimicActorCritic
from rl_isaac.train_cfg import OPTMimicPPOCfg
from rl_isaac.callbacks import TrainingLogger


def make_env(args) -> Go2TrackingEnv:
    """Create the Isaac Lab tracking environment."""
    cfg = Go2TrackingEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.state_traj_path = args.state_traj
    cfg.grf_traj_path = args.grf_traj
    cfg.joint_vel_traj_path = args.joint_vel_traj
    cfg.contact_sequence_path = args.contact_sequence if args.contact_sequence else ""
    return Go2TrackingEnv(cfg)


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logger (created early so all messages go to experiment.log)
    logger = TrainingLogger(output_dir, args.total_timesteps, args.num_envs, 0)

    # Create environment
    logger.info("Creating environment...")
    env = make_env(args)
    max_phase = env._max_phase
    num_envs = env.num_envs
    device = env.device

    logger.info(f"Environment: {num_envs} envs, max_phase={max_phase}")
    logger.update_header(args.total_timesteps, num_envs, max_phase)

    # PPO config
    ppo_cfg = OPTMimicPPOCfg()
    ppo_cfg.seed = args.seed
    ppo_cfg.num_learning_epochs = args.n_epochs

    # n_steps = max_phase (match episode length for good GAE credit assignment)
    n_steps = max_phase
    samples_per_update = n_steps * num_envs
    n_updates = max(1, args.total_timesteps // samples_per_update)

    # Minibatch size: ~5000 samples per minibatch
    n_minibatches = max(1, (n_steps * num_envs) // 5000)
    ppo_cfg.num_mini_batches = n_minibatches

    logger.info(f"Training: {args.total_timesteps} total steps")
    logger.info(f"  {n_steps} steps/update, {n_updates} updates")
    logger.info(f"  {ppo_cfg.num_learning_epochs} epochs, {n_minibatches} minibatches")

    # Create actor-critic network
    actor_critic = OPTMimicActorCritic(
        num_obs=39,
        num_privileged_obs=0,
        num_actions=12,
        actor_hidden_dims=ppo_cfg.actor_hidden_dims,
        critic_hidden_dims=ppo_cfg.critic_hidden_dims,
    ).to(device)

    # Observation normalization (Welford running mean/var, matches MJX)
    obs_normalizer = EmpiricalNormalization(shape=[39], until=1e8).to(device)

    # Optimizer (matching MJX: Adam with global norm clipping via optax.chain)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=ppo_cfg.learning_rate)

    # LR schedule: 1e-3 * 0.999^(step/denom)
    lr_denom = ppo_cfg.num_learning_epochs * n_minibatches

    def lr_lambda(epoch):
        return 0.999 ** (epoch / lr_denom)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Storage for rollout
    obs_buf = torch.zeros(n_steps, num_envs, 39, device=device)
    act_buf = torch.zeros(n_steps, num_envs, 12, device=device)
    rew_buf = torch.zeros(n_steps, num_envs, device=device)
    done_buf = torch.zeros(n_steps, num_envs, dtype=torch.bool, device=device)
    val_buf = torch.zeros(n_steps, num_envs, device=device)
    lp_buf = torch.zeros(n_steps, num_envs, device=device)

    # Initial reset
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    total_steps = 0
    best_ep_return = -float("inf")
    next_video_step = 1_000_000  # render video every 1M steps (matches MJX train.py)
    t_start = time.time()

    logger.info(f"Starting training ({n_updates} updates)...")

    for update_idx in range(n_updates):
        t_update = time.time()

        # Entropy annealing (matching MJX: linear from 0.002 to 0.0005)
        frac = update_idx / max(1, n_updates - 1)
        ent_coef = ppo_cfg.entropy_coef + (ppo_cfg.entropy_coef_end - ppo_cfg.entropy_coef) * frac

        # ------ Rollout collection ------
        ep_returns = []
        ep_lengths = []
        running_return = torch.zeros(num_envs, device=device)
        running_length = torch.zeros(num_envs, device=device)

        for step in range(n_steps):
            # Normalize observation
            obs_norm = obs_normalizer(obs)

            # Sample action
            with torch.no_grad():
                actions = actor_critic.act(obs_norm)
                values = actor_critic.evaluate(obs_norm)
                log_probs = actor_critic.get_actions_log_prob(actions)

            obs_buf[step] = obs_norm
            act_buf[step] = actions
            val_buf[step] = values
            lp_buf[step] = log_probs

            # Environment step
            obs_dict, rewards, terminated, truncated, info = env.step(actions)
            obs = obs_dict["policy"]
            dones = terminated | truncated

            rew_buf[step] = rewards
            done_buf[step] = dones

            # Track episode returns (vectorized)
            running_return += rewards
            running_length += 1
            done_mask = dones.bool()
            if done_mask.any():
                ep_returns.extend(running_return[done_mask].tolist())
                ep_lengths.extend(running_length[done_mask].tolist())
                running_return[done_mask] = 0
                running_length[done_mask] = 0

        # Last value for GAE
        with torch.no_grad():
            obs_norm = obs_normalizer(obs)
            last_values = actor_critic.evaluate(obs_norm)

        # ------ PPO update ------
        # Compute GAE
        advantages = torch.zeros_like(rew_buf)
        last_gae = torch.zeros(num_envs, device=device)
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_values = last_values
            else:
                next_values = val_buf[t + 1]
            next_non_terminal = (~done_buf[t]).float()
            delta = rew_buf[t] + ppo_cfg.gamma * next_values * next_non_terminal - val_buf[t]
            last_gae = delta + ppo_cfg.gamma * ppo_cfg.lam * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + val_buf

        # Flatten
        total_samples = n_steps * num_envs
        obs_flat = obs_buf.reshape(total_samples, -1)
        act_flat = act_buf.reshape(total_samples, -1)
        lp_flat = lp_buf.reshape(total_samples)
        adv_flat = advantages.reshape(total_samples)
        ret_flat = returns.reshape(total_samples)

        # PPO epochs
        batch_size = total_samples // n_minibatches
        ppo_metrics = {"pg_loss": 0, "vf_loss": 0, "entropy": 0, "approx_kl": 0, "clip_frac": 0}
        n_updates_inner = 0

        for epoch in range(ppo_cfg.num_learning_epochs):
            perm = torch.randperm(total_samples, device=device)
            for mb in range(n_minibatches):
                idx = perm[mb * batch_size:(mb + 1) * batch_size]

                mb_obs = obs_flat[idx]
                mb_act = act_flat[idx]
                mb_old_lp = lp_flat[idx]
                mb_adv = adv_flat[idx]
                mb_ret = ret_flat[idx]

                # Normalize advantages
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Forward pass
                actor_critic._update_distribution(mb_obs)
                new_lp = actor_critic.get_actions_log_prob(mb_act)
                new_values = actor_critic.evaluate(mb_obs)
                entropy = actor_critic.entropy.mean()

                # Policy loss
                ratio = torch.exp(new_lp - mb_old_lp)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * ratio.clamp(1.0 - ppo_cfg.clip_param, 1.0 + ppo_cfg.clip_param)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                vf_loss = 0.5 * ((new_values - mb_ret) ** 2).mean()

                # Total loss
                loss = pg_loss + ppo_cfg.value_loss_coef * vf_loss - ent_coef * entropy

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), ppo_cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()

                # Metrics
                with torch.no_grad():
                    approx_kl = 0.5 * ((new_lp - mb_old_lp) ** 2).mean()
                    clip_frac = ((ratio - 1.0).abs() > ppo_cfg.clip_param).float().mean()

                ppo_metrics["pg_loss"] += pg_loss.item()
                ppo_metrics["vf_loss"] += vf_loss.item()
                ppo_metrics["entropy"] += entropy.item()
                ppo_metrics["approx_kl"] += approx_kl.item()
                ppo_metrics["clip_frac"] += clip_frac.item()
                n_updates_inner += 1

        # Average metrics
        for k in ppo_metrics:
            ppo_metrics[k] /= max(1, n_updates_inner)

        total_steps += samples_per_update
        dt = time.time() - t_update

        # Episode stats
        mean_ep_return = np.mean(ep_returns) if ep_returns else 0.0
        mean_ep_length = np.mean(ep_lengths) if ep_lengths else n_steps

        # Get reward components from env extras
        log_extras = env.extras.get("log", {})

        # Grad norm
        grad_norm = 0.0
        for p in actor_critic.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        mean_std = actor_critic.action_std.mean().item()

        # Log
        logger.log_update(
            update_idx=update_idx,
            total_steps=total_steps,
            ep_return=mean_ep_return,
            ep_length=mean_ep_length,
            r_pos=log_extras.get("r_pos", 0),
            r_ori=log_extras.get("r_ori", 0),
            r_joint=log_extras.get("r_joint", 0),
            r_smooth=log_extras.get("r_smooth", 0),
            r_torque=log_extras.get("r_torque", 0),
            mean_std=mean_std,
            grad_norm=grad_norm,
            pg_loss=ppo_metrics["pg_loss"],
            vf_loss=ppo_metrics["vf_loss"],
            approx_kl=ppo_metrics["approx_kl"],
            clip_frac=ppo_metrics["clip_frac"],
            entropy=ppo_metrics["entropy"],
            ent_coef=ent_coef,
            dt=dt,
            term_info=log_extras,
        )

        # Save best
        if mean_ep_return > best_ep_return:
            best_ep_return = mean_ep_return
            _save_checkpoint(output_dir / "best_model", actor_critic, obs_normalizer, total_steps)

        # Periodic checkpoints
        if (update_idx + 1) % max(1, n_updates // 10) == 0 or update_idx == n_updates - 1:
            _save_checkpoint(
                output_dir / "checkpoints" / f"step_{total_steps}",
                actor_critic, obs_normalizer, total_steps,
            )

        # Periodic video rendering (every 1M steps, matches MJX train.py)
        if total_steps >= next_video_step:
            logger.info(f"  Rendering tracking video at step {total_steps:,}...")
            _render_video(actor_critic, obs_normalizer, args, output_dir, total_steps, logger)
            next_video_step = (total_steps // 1_000_000 + 1) * 1_000_000

        # Periodic plots
        if (update_idx + 1) % max(1, n_updates // 5) == 0 or update_idx == n_updates - 1:
            logger.save_reward_curve(str(output_dir), total_steps)

    elapsed = time.time() - t_start
    logger.info(f"Training complete in {elapsed:.1f}s ({total_steps:,} steps)")
    logger.info(f"Best ep_return: {best_ep_return:.2f}")

    _save_checkpoint(output_dir / "final_model", actor_critic, obs_normalizer, total_steps)
    logger.save_reward_curve(str(output_dir), total_steps)

    # Final best model video
    logger.info("Rendering final best model video...")
    _render_video(actor_critic, obs_normalizer, args, output_dir, total_steps, logger, label="best_model")

    env.close()
    simulation_app.close()


def _save_checkpoint(path: Path, actor_critic, obs_normalizer, step: int):
    """Save checkpoint compatible with evaluation."""
    path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": actor_critic.state_dict(),
        "normalizer_state_dict": obs_normalizer.state_dict(),
        "step": step,
    }, path / "checkpoint.pt")


def _render_video(
    actor_critic, obs_normalizer, args, output_dir: Path, total_steps: int,
    logger=None, label: str = "",
):
    """Render a tracking video using CPU MuJoCo rollout.

    Runs the current best policy on the reference trajectory and saves an MP4.
    This matches rl/train.py's periodic video rendering (every 1M steps).
    """
    def _log(msg):
        if logger:
            logger.info(msg)

    try:
        from rl_isaac.evaluate import execute_rollout

        # Load trajectory data
        state_traj = np.load(args.state_traj)
        grf_traj = np.load(args.grf_traj)
        joint_vel_traj = np.load(args.joint_vel_traj)
        contact_seq = np.load(args.contact_sequence) if args.contact_sequence else None

        # Create a CPU copy of actor_critic for evaluation
        ac_cpu = OPTMimicActorCritic(num_obs=39, num_privileged_obs=0, num_actions=12)
        ac_cpu.load_state_dict({
            k: v.cpu() for k, v in actor_critic.state_dict().items()
        })
        ac_cpu.eval()

        # Get normalizer state for CPU
        norm_state = obs_normalizer.state_dict()
        norm_state_cpu = {k: v.cpu() for k, v in norm_state.items()}

        # Run rollout
        qpos_rl, qvel_rl, grf_rl, images = execute_rollout(
            state_traj, grf_traj, joint_vel_traj,
            ac_cpu, norm_state_cpu, contact_seq,
            render=True,
        )

        if images:
            video_dir = output_dir / "runs"
            video_dir.mkdir(parents=True, exist_ok=True)
            if label:
                video_path = video_dir / f"{label}.mp4"
            else:
                video_path = video_dir / f"step_{total_steps:07d}.mp4"

            import imageio
            fps = 50  # 50Hz control
            imageio.mimsave(str(video_path), images, fps=fps)
            n_tracked = len(qpos_rl)
            _log(f"  Video saved: {video_path} ({n_tracked}/{state_traj.shape[0]-1} steps tracked)")
        else:
            _log(f"  No frames captured for video at step {total_steps}")

    except Exception as e:
        import traceback
        _log(f"  Video render failed: {e}")
        _log(traceback.format_exc())


if __name__ == "__main__":
    train(args_cli)
