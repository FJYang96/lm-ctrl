"""Checkpoint saving and video rendering for OPT-Mimic training.

Video rendering runs a deterministic rollout in the training env (PhysX),
temporarily loading the best model weights and restoring them afterward.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .network import OPTMimicActorCritic
from .tracking_env import Go2TrackingEnv


def save_checkpoint(path: Path, actor_critic, obs_normalizer, step: int):
    """Save checkpoint compatible with evaluation."""
    path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": actor_critic.state_dict(),
        "normalizer_state_dict": obs_normalizer.state_dict(),
        "step": step,
    }, path / "checkpoint.pt")


def render_video(
    env: Go2TrackingEnv,
    actor_critic: OPTMimicActorCritic,
    obs_normalizer,
    output_dir: Path,
    total_steps: int,
    logger=None,
    label: str = "",
) -> torch.Tensor:
    """Render best-model video using the training env (PhysX).

    Temporarily loads best model, runs deterministic rollout from phase 0,
    captures frames, restores training weights, resets env.

    Returns new obs tensor after env reset.
    """
    best_path = output_dir / "best_model" / "checkpoint.pt"
    if not best_path.exists():
        if logger:
            logger.info("  No best model checkpoint yet, skipping video.")
        obs_dict, _ = env.reset()
        return obs_dict["policy"]

    train_ac_state = {k: v.clone() for k, v in actor_critic.state_dict().items()}
    train_norm_state = {k: v.clone() for k, v in obs_normalizer.state_dict().items()}
    was_training = actor_critic.training

    try:
        device = env.device
        max_phase = env._max_phase

        best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
        actor_critic.load_state_dict(best_ckpt["model_state_dict"])
        actor_critic.eval()
        if "normalizer_state_dict" in best_ckpt:
            obs_normalizer.load_state_dict(best_ckpt["normalizer_state_dict"])
        obs_normalizer.eval()

        # Reset env to phase 0, no domain randomization
        env.reset()
        env._phase[:] = 0
        env._prev_action[:] = 0
        env._last_torque[:] = 0
        env._first_step[:] = True
        env._joint_offset[:] = 0
        env._torque_scale[:] = 1.0

        all_ids = env._robot._ALL_INDICES
        ref_pos = env._ref_body_pos[0:1].expand(env.num_envs, -1).clone() + env._env_origins
        ref_quat = env._ref_body_quat[0:1].expand(env.num_envs, -1).clone()
        env._robot.write_root_pose_to_sim(torch.cat([ref_pos, ref_quat], dim=-1), all_ids)
        env._robot.write_root_velocity_to_sim(torch.cat([
            env._ref_body_vel[0:1].expand(env.num_envs, -1).clone(),
            env._ref_body_ang_vel[0:1].expand(env.num_envs, -1).clone(),
        ], dim=-1), all_ids)
        env._robot.write_joint_state_to_sim(
            env._to_isaac_order(env._ref_joint_pos[0:1].expand(env.num_envs, -1).clone()),
            env._to_isaac_order(env._ref_joint_vel[0:1].expand(env.num_envs, -1).clone()),
            None, all_ids,
        )

        images = []
        for _ in range(max_phase):
            obs = env._get_observations()["policy"]
            with torch.no_grad():
                actions = actor_critic.act_inference(obs_normalizer(obs))
            _, _, terminated, truncated, _ = env.step(actions)
            frame = env.render()
            if frame is not None:
                images.append(frame)
            if (terminated | truncated).all():
                break

        if images:
            video_dir = output_dir / "runs"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / (f"{label}.mp4" if label else f"step_{total_steps:07d}.mp4")
            import imageio
            imageio.mimsave(str(video_path), images, fps=50)
            if logger:
                logger.info(f"  Video saved: {video_path} ({len(images)}/{max_phase} frames)")
        elif logger:
            logger.info(f"  No frames captured at step {total_steps}")

    except Exception as e:
        import traceback
        if logger:
            logger.info(f"  Video render failed: {e}")
            logger.info(traceback.format_exc())

    actor_critic.load_state_dict(train_ac_state)
    obs_normalizer.load_state_dict(train_norm_state)
    if was_training:
        actor_critic.train()
        obs_normalizer.train()
    obs_dict, _ = env.reset()
    return obs_dict["policy"]
