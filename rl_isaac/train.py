"""Train OPT-Mimic tracking policy using Isaac Lab + PPO.

Usage (inside Docker):
    /workspace/isaaclab/isaaclab.sh -p -m rl_isaac.train \
        --state-traj results/.../state_traj.npy \
        --grf-traj results/.../grf_traj.npy \
        --joint-vel-traj results/.../joint_vel_traj.npy \
        --num-envs 4096 --total-timesteps 50000000 --headless
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train OPT-Mimic tracking policy (Isaac Lab)")
parser.add_argument("--state-traj", type=str, required=True)
parser.add_argument("--grf-traj", type=str, required=True)
parser.add_argument("--joint-vel-traj", type=str, required=True)
parser.add_argument("--contact-sequence", type=str, default="")
parser.add_argument("--output-dir", type=str, default="rl_isaac/trained_models")
parser.add_argument("--total-timesteps", type=int, default=50_000_000)
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--n-epochs", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Ensure repo root is on sys.path (must be AFTER AppLauncher which resets sys.path)
# AppLauncher may register a 'utils' namespace package that shadows ours,
# so we force-load our utils package into sys.modules.
import os, sys, types, importlib, importlib.util  # noqa: E401,E402
_repo_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _repo_root)
for _mod_name, _mod_file in [
    ("utils", "utils/__init__.py"),
    ("utils.conversion", "utils/conversion.py"),
]:
    _spec = importlib.util.spec_from_file_location(
        _mod_name, str(Path(_repo_root) / _mod_file)
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_mod_name] = _mod
    _spec.loader.exec_module(_mod)

# GLFW/mujoco stubs — gym_quadruped (imported via feedforward→model) needs these in headless mode
os.environ.setdefault("MUJOCO_GL", "egl")
for mod_name, attrs in [("mujoco.viewer", {"Handle": type("Handle", (), {})}), ("glfw", {"_glfw": True})]:
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[mod_name] = m
if "glfw.library" not in sys.modules:
    sys.modules["glfw.library"] = types.ModuleType("glfw.library")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.modules import EmpiricalNormalization  # noqa: E402

from rl_isaac.env_cfg import Go2TrackingEnvCfg  # noqa: E402
from rl_isaac.tracking_env import Go2TrackingEnv  # noqa: E402
from rl_isaac.network import OPTMimicActorCritic  # noqa: E402
from rl_isaac.train_cfg import OPTMimicPPOCfg  # noqa: E402
from rl_isaac.callbacks import TrainingLogger  # noqa: E402
from rl_isaac.ppo import compute_gae, ppo_update  # noqa: E402
from rl_isaac.video import save_checkpoint, render_video  # noqa: E402


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(output_dir, args.total_timesteps, args.num_envs, 0)

    logger.info("Creating environment...")
    cfg = Go2TrackingEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.state_traj_path = args.state_traj
    cfg.grf_traj_path = args.grf_traj
    cfg.joint_vel_traj_path = args.joint_vel_traj
    cfg.contact_sequence_path = args.contact_sequence if args.contact_sequence else ""
    env = Go2TrackingEnv(cfg, render_mode="rgb_array")
    max_phase = env._max_phase
    num_envs = env.num_envs
    device = env.device
    logger.update_header(args.total_timesteps, num_envs, max_phase)

    ppo_cfg = OPTMimicPPOCfg()
    ppo_cfg.seed = args.seed
    ppo_cfg.num_learning_epochs = args.n_epochs
    n_steps = max_phase
    samples_per_update = n_steps * num_envs
    n_updates = max(1, args.total_timesteps // samples_per_update)
    ppo_cfg.num_mini_batches = max(1, samples_per_update // 5000)
    logger.info(f"Training: {args.total_timesteps} steps, {n_updates} updates, "
                f"{ppo_cfg.num_learning_epochs} epochs, {ppo_cfg.num_mini_batches} minibatches")

    actor_critic = OPTMimicActorCritic(
        num_obs=39, num_privileged_obs=0, num_actions=12,
        actor_hidden_dims=ppo_cfg.actor_hidden_dims, critic_hidden_dims=ppo_cfg.critic_hidden_dims,
    ).to(device)
    obs_normalizer = EmpiricalNormalization(shape=[39], until=1e8).to(device)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=ppo_cfg.learning_rate)
    lr_denom = ppo_cfg.num_learning_epochs * ppo_cfg.num_mini_batches
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda ep: 0.999 ** (ep / lr_denom))

    obs_buf = torch.zeros(n_steps, num_envs, 39, device=device)
    act_buf = torch.zeros(n_steps, num_envs, 12, device=device)
    rew_buf = torch.zeros(n_steps, num_envs, device=device)
    done_buf = torch.zeros(n_steps, num_envs, dtype=torch.bool, device=device)
    val_buf = torch.zeros(n_steps, num_envs, device=device)
    lp_buf = torch.zeros(n_steps, num_envs, device=device)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    total_steps = 0
    best_ep_return = -float("inf")
    next_video_step = 1_000_000
    t_start = time.time()
    logger.info(f"Starting training ({n_updates} updates)...")

    for update_idx in range(n_updates):
        t_update = time.time()
        frac = update_idx / max(1, n_updates - 1)
        ent_coef = ppo_cfg.entropy_coef + (ppo_cfg.entropy_coef_end - ppo_cfg.entropy_coef) * frac

        # Rollout collection
        ep_returns, ep_lengths = [], []
        running_return = torch.zeros(num_envs, device=device)
        running_length = torch.zeros(num_envs, device=device)
        for step in range(n_steps):
            obs_norm = obs_normalizer(obs)
            with torch.no_grad():
                actions = actor_critic.act(obs_norm)
                values = actor_critic.evaluate(obs_norm)
                log_probs = actor_critic.get_actions_log_prob(actions)
            obs_buf[step], act_buf[step], val_buf[step], lp_buf[step] = obs_norm, actions, values, log_probs
            obs_dict, rewards, terminated, truncated, _ = env.step(actions)
            obs = obs_dict["policy"]
            dones = terminated | truncated
            rew_buf[step], done_buf[step] = rewards, dones
            running_return += rewards
            running_length += 1
            if dones.any():
                ep_returns.extend(running_return[dones].tolist())
                ep_lengths.extend(running_length[dones].tolist())
                running_return[dones] = 0
                running_length[dones] = 0

        with torch.no_grad():
            last_values = actor_critic.evaluate(obs_normalizer(obs))
        advantages, returns = compute_gae(rew_buf, val_buf, done_buf, last_values, ppo_cfg.gamma, ppo_cfg.lam)
        total_samples = n_steps * num_envs
        ppo_metrics = ppo_update(
            actor_critic, optimizer, scheduler,
            obs_buf.reshape(total_samples, -1), act_buf.reshape(total_samples, -1),
            lp_buf.reshape(total_samples), advantages.reshape(total_samples),
            returns.reshape(total_samples), ppo_cfg, ent_coef,
        )

        total_steps += samples_per_update
        mean_ep_return = np.mean(ep_returns) if ep_returns else 0.0
        mean_ep_length = np.mean(ep_lengths) if ep_lengths else n_steps
        log_extras = env.extras.get("log", {})
        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in actor_critic.parameters() if p.grad is not None) ** 0.5

        logger.log_update(
            update_idx=update_idx, total_steps=total_steps,
            ep_return=mean_ep_return, ep_length=mean_ep_length,
            r_pos=log_extras.get("r_pos", 0), r_ori=log_extras.get("r_ori", 0),
            r_joint=log_extras.get("r_joint", 0), r_smooth=log_extras.get("r_smooth", 0),
            r_torque=log_extras.get("r_torque", 0), mean_std=actor_critic.action_std.mean().item(),
            grad_norm=grad_norm, pg_loss=ppo_metrics["pg_loss"], vf_loss=ppo_metrics["vf_loss"],
            approx_kl=ppo_metrics["approx_kl"], clip_frac=ppo_metrics["clip_frac"],
            entropy=ppo_metrics["entropy"], ent_coef=ent_coef,
            dt=time.time() - t_update, term_info=log_extras,
        )

        if mean_ep_return > best_ep_return:
            best_ep_return = mean_ep_return
            save_checkpoint(output_dir / "best_model", actor_critic, obs_normalizer, total_steps)
        if (update_idx + 1) % max(1, n_updates // 10) == 0 or update_idx == n_updates - 1:
            save_checkpoint(output_dir / "checkpoints" / f"step_{total_steps}", actor_critic, obs_normalizer, total_steps)
        if total_steps >= next_video_step:
            logger.info(f"  Rendering best model video at step {total_steps:,}...")
            obs = render_video(env, actor_critic, obs_normalizer, output_dir, total_steps, logger)
            next_video_step = (total_steps // 1_000_000 + 1) * 1_000_000
        if (update_idx + 1) % max(1, n_updates // 5) == 0 or update_idx == n_updates - 1:
            logger.save_reward_curve(str(output_dir), total_steps)

    logger.info(f"Training complete in {time.time() - t_start:.1f}s ({total_steps:,} steps)")
    logger.info(f"Best ep_return: {best_ep_return:.2f}")
    save_checkpoint(output_dir / "final_model", actor_critic, obs_normalizer, total_steps)
    logger.save_reward_curve(str(output_dir), total_steps)
    logger.info("Rendering final best model video...")
    render_video(env, actor_critic, obs_normalizer, output_dir, total_steps, logger, label="best_model")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    train(args_cli)
