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
    """Render best-model video showing one robot from starting position.

    Temporarily loads best model, sets env 0 to phase 0 with no DR,
    zooms camera onto env 0, runs deterministic rollout, captures frames,
    restores training weights, resets env.

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

        # Reset env, then set env 0 to phase 0 with no DR.
        env.reset()
        env._phase[0] = 0
        env._prev_action[0] = 0
        env._last_torque[0] = 0
        env._first_step[0] = True
        env._joint_offset[0] = 0
        env._torque_scale[0] = 1.0

        env_0 = torch.tensor([0], device=device, dtype=torch.long)
        env._robot.write_root_pose_to_sim(torch.cat([
            env._ref_body_pos[0:1].clone() + env._env_origins[0:1],
            env._ref_body_quat[0:1].clone(),
        ], dim=-1), env_0)
        env._robot.write_root_velocity_to_sim(torch.cat([
            env._ref_body_vel[0:1].clone(),
            env._ref_body_ang_vel[0:1].clone(),
        ], dim=-1), env_0)
        env._robot.write_joint_state_to_sim(
            env._to_isaac_order(env._ref_joint_pos[0:1].clone()),
            env._to_isaac_order(env._ref_joint_vel[0:1].clone()),
            None, env_0,
        )

        # Zoom camera close to env 0 so only one robot is visible.
        # env_origins[0] is env 0's world position.
        origin = env._env_origins[0].cpu().numpy()
        eye = (origin[0] + 1.5, origin[1] + 1.5, origin[2] + 1.0)
        target = (origin[0], origin[1], origin[2] + 0.3)
        try:
            env.sim.set_camera_view(eye, target)
        except Exception:
            pass  # headless or API unavailable

        # Disable auto-reset so we capture the real post-step state (landing frame).
        original_reset = env._reset_idx
        env._reset_idx = lambda env_ids: None

        images = []
        for _ in range(max_phase):
            obs = env._get_observations()["policy"]
            with torch.no_grad():
                actions = actor_critic.act_inference(obs_normalizer(obs))
            _, _, terminated, truncated, _ = env.step(actions)

            frame = env.render()
            if frame is not None:
                images.append(frame)

            if terminated[0] or truncated[0]:
                break

        env._reset_idx = original_reset

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
