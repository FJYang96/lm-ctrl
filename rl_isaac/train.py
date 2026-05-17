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
import json
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
    rollout_steps = env._episode_horizon
    num_envs = env.num_envs
    device = env.device
    logger.update_header(args.total_timesteps, num_envs, max_phase,
                         env._mpc_dt, env._env_dt, env._phase_inc)

    ppo_cfg = OPTMimicPPOCfg()
    ppo_cfg.seed = args.seed
    ppo_cfg.num_learning_epochs = args.n_epochs
    n_steps = rollout_steps
    ppo_cfg.num_steps_per_env = n_steps
    samples_per_update = n_steps * num_envs
    n_updates = max(1, args.total_timesteps // samples_per_update)
    ppo_cfg.num_mini_batches = max(1, samples_per_update // 5000)
    logger.info(f"Reference: {max_phase} MPC frames at {env._mpc_dt:.3f}s; "
                f"rollout: {n_steps} env steps at {env._env_dt:.3f}s "
                f"(phase_inc={env._phase_inc:.3f})")
    logger.info(f"Training: {args.total_timesteps} steps, {n_updates} updates, "
                f"{ppo_cfg.num_learning_epochs} epochs, {ppo_cfg.num_mini_batches} minibatches")

    target_phase0_prob = float(getattr(cfg, "phase0_reset_prob", 0.0))
    curriculum_enabled = bool(getattr(cfg, "phase0_curriculum", False)) and target_phase0_prob > 0.0
    curriculum_min_frac = float(getattr(cfg, "phase0_curriculum_min_frac", 0.6))
    curriculum_min_frac = min(max(curriculum_min_frac, 0.0), 1.0)
    curriculum_switch_step = None
    curriculum_state = {
        "enabled": curriculum_enabled,
        "initial_phase0_reset_prob": 0.0 if curriculum_enabled else target_phase0_prob,
        "target_phase0_reset_prob": target_phase0_prob,
        "min_switch_frac": curriculum_min_frac,
        "min_switch_step": int(args.total_timesteps * curriculum_min_frac),
        "switched": False,
        "switch_step": None,
        "switch_reason": None,
    }

    def write_curriculum_state():
        with (output_dir / "curriculum.json").open("w") as f:
            json.dump(curriculum_state, f, indent=2)

    if curriculum_enabled:
        env.cfg.phase0_reset_prob = 0.0
        logger.info(
            "Reset curriculum: random-phase DR resets first; "
            f"switch to phase0_reset_prob={target_phase0_prob:.2f} after "
            f"clean phase-0 completion or {curriculum_min_frac:.0%} of training."
        )
    write_curriculum_state()

    actor_critic = OPTMimicActorCritic(
        num_obs=33, num_privileged_obs=0, num_actions=12,
        actor_hidden_dims=ppo_cfg.actor_hidden_dims, critic_hidden_dims=ppo_cfg.critic_hidden_dims,
    ).to(device)
    obs_normalizer = EmpiricalNormalization(shape=[33], until=1e8).to(device)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=ppo_cfg.learning_rate)
    lr_denom = ppo_cfg.num_learning_epochs * ppo_cfg.num_mini_batches
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda ep: 0.999 ** (ep / lr_denom))

    obs_buf = torch.zeros(n_steps, num_envs, 33, device=device)
    act_buf = torch.zeros(n_steps, num_envs, 12, device=device)
    rew_buf = torch.zeros(n_steps, num_envs, device=device)
    done_buf = torch.zeros(n_steps, num_envs, dtype=torch.bool, device=device)
    val_buf = torch.zeros(n_steps, num_envs, device=device)
    lp_buf = torch.zeros(n_steps, num_envs, device=device)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    total_steps = 0
    best_ep_metric = (-float("inf"), -float("inf"))  # (mean_ep_length, mean_ep_return)
    best_phase0_steps = -1
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
        # Phase-0.1 instrumentation: drain per-cause × per-phase termination
        # histogram from the env and append it to term_breakdown.csv. Drain
        # AFTER log_update so the rolled-up counts that update_idx logged are
        # the same ones written to the CSV.
        _term_counts, _term_hist = env.flush_term_diagnostics()
        logger.log_term_breakdown(update_idx, total_steps, _term_hist)
        # Phase-0.2 instrumentation: per-phase tracking-error means.
        _ph_sums, _ph_counts, _ph_metrics = env.flush_phase_errors()
        logger.log_phase_errors(update_idx, total_steps, _ph_sums, _ph_counts, _ph_metrics)
        # Phase-0.3 instrumentation: per-foot contact-mismatch slip stats.
        _sl_c, _sl_f, _sl_o = env.flush_slip_diagnostics()
        logger.log_slip_diagnostics(update_idx, total_steps, _sl_c, _sl_f, _sl_o)

        current_ep_metric = (mean_ep_length, mean_ep_return)
        if current_ep_metric > best_ep_metric:
            best_ep_metric = current_ep_metric
            save_checkpoint(output_dir / "best_training_model", actor_critic, obs_normalizer, total_steps)
        if (update_idx + 1) % max(1, n_updates // 10) == 0 or update_idx == n_updates - 1:
            save_checkpoint(output_dir / "checkpoints" / f"step_{total_steps}", actor_critic, obs_normalizer, total_steps)
        if total_steps >= next_video_step:
            save_checkpoint(
                output_dir / "video_checkpoints" / f"step_{total_steps}",
                actor_critic, obs_normalizer, total_steps,
            )
            logger.info(f"  Rendering current clean phase-0 video at step {total_steps:,}...")
            obs, phase0_steps = render_video(
                env, actor_critic, obs_normalizer, output_dir, total_steps, logger,
                use_best_model=False, clean_eval=True,
            )
            if phase0_steps > best_phase0_steps:
                best_phase0_steps = phase0_steps
                save_checkpoint(output_dir / "best_model", actor_critic, obs_normalizer, total_steps)
                logger.info(
                    f"  New best clean phase-0 model: "
                    f"{best_phase0_steps}/{env._episode_horizon} env steps"
                )
            if curriculum_enabled and not curriculum_state["switched"]:
                reached_clean = best_phase0_steps >= env._episode_horizon
                reached_warmup = total_steps >= curriculum_state["min_switch_step"]
                if reached_clean or reached_warmup:
                    env.cfg.phase0_reset_prob = target_phase0_prob
                    curriculum_switch_step = total_steps
                    reason = "clean_phase0_complete" if reached_clean else "warmup_complete"
                    curriculum_state.update({
                        "switched": True,
                        "switch_step": int(total_steps),
                        "switch_reason": reason,
                    })
                    write_curriculum_state()
                    logger.info(
                        "  Reset curriculum switch: "
                        f"phase0_reset_prob={target_phase0_prob:.2f} "
                        f"at step {total_steps:,} ({reason})."
                    )
            next_video_step = (total_steps // 1_000_000 + 1) * 1_000_000
        if (update_idx + 1) % max(1, n_updates // 5) == 0 or update_idx == n_updates - 1:
            logger.save_reward_curve(str(output_dir), total_steps)

    logger.info(f"Training complete in {time.time() - t_start:.1f}s ({total_steps:,} steps)")
    logger.info(f"Best (ep_length, ep_return): ({best_ep_metric[0]:.2f}, {best_ep_metric[1]:.2f})")
    logger.info(f"Best clean phase-0 steps: {best_phase0_steps}/{env._episode_horizon}")
    if curriculum_enabled and not curriculum_state["switched"]:
        curriculum_state["switch_step"] = int(curriculum_switch_step) if curriculum_switch_step else None
        write_curriculum_state()
    if not (output_dir / "best_model" / "checkpoint.pt").exists():
        save_checkpoint(output_dir / "best_model", actor_critic, obs_normalizer, total_steps)
    save_checkpoint(output_dir / "final_model", actor_critic, obs_normalizer, total_steps)
    logger.save_reward_curve(str(output_dir), total_steps)
    # Final best_model.mp4 render removed: redundant with run_smoke_test.sh
    # Step [2/4] (clean rl_tracking.mp4) and Step [3/4] (DR-on dr_seeds/*.mp4).
    # Periodic runs/step_<N>.mp4 files still show training progression.
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    train(args_cli)
