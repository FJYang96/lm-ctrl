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
    use_best_model: bool = True,
    clean_eval: bool = False,
) -> tuple[torch.Tensor, int]:
    """Render a deterministic phase-0 video showing one robot.

    By default this temporarily loads best_model.  Training can pass
    use_best_model=False, clean_eval=True to probe the current policy under
    the same nominal phase-0 condition used by smoke-test evaluation.

    Returns (new_obs, tracked_env_steps) after resetting the env.
    """
    best_path = output_dir / "best_model" / "checkpoint.pt"
    if use_best_model and not best_path.exists():
        if logger:
            logger.info("  No best model checkpoint yet, skipping video.")
        obs_dict, _ = env.reset()
        return obs_dict["policy"], 0

    train_ac_state = None
    train_norm_state = None
    was_training = actor_critic.training
    tracked_steps = 0

    try:
        device = env.device
        max_phase = env._max_phase
        horizon = env._episode_horizon
        video_fps = int(round(1.0 / env._env_dt))

        if use_best_model:
            train_ac_state = {k: v.clone() for k, v in actor_critic.state_dict().items()}
            train_norm_state = {k: v.clone() for k, v in obs_normalizer.state_dict().items()}
            best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
            actor_critic.load_state_dict(best_ckpt["model_state_dict"])
            if "normalizer_state_dict" in best_ckpt:
                obs_normalizer.load_state_dict(best_ckpt["normalizer_state_dict"])
        actor_critic.eval()
        obs_normalizer.eval()

        # Reset env, then put the rendered rollout on the phase-0 reference.
        # For clean checkpoint selection, mirror evaluate.py across the whole
        # vectorized batch so the observation normalizer sees the same rollout
        # distribution as the smoke-test clean eval.
        env.reset()
        if clean_eval:
            reset_ids = env._robot._ALL_INDICES
            ref_phases = torch.zeros(env.num_envs, device=device)
            env._phase[:] = 0.0
            env._prev_action[:] = 0
            env._last_torque[:] = 0
            env._first_step[:] = True
            env._joint_offset[:] = 0.0
            env._torque_scale[:] = 1.0
        else:
            reset_ids = torch.tensor([0], device=device, dtype=torch.long)
            ref_phases = torch.zeros(1, device=device)
            env._phase[0] = 0.0
            env._prev_action[0] = 0
            env._last_torque[0] = 0
            env._first_step[0] = True

        if clean_eval:
            mat = env._robot.root_physx_view.get_material_properties()
            reset_ids_cpu = torch.arange(env.num_envs, dtype=torch.long)
            mat[reset_ids_cpu, :, 0] = 1.0
            mat[reset_ids_cpu, :, 1] = 1.0
            mat[reset_ids_cpu, :, 2] = 0.0
            env._robot.root_physx_view.set_material_properties(mat, reset_ids_cpu)

        ref0 = env._sample_reference(ref_phases)
        ref_pos = ref0["body_pos"].clone()
        if clean_eval:
            ref_pos = ref_pos + env._env_origins
        else:
            ref_pos = ref_pos + env._env_origins[0:1]
        env._robot.write_root_pose_to_sim(torch.cat([
            ref_pos,
            ref0["body_quat"].clone(),
        ], dim=-1), reset_ids)
        env._robot.write_root_velocity_to_sim(torch.cat([
            ref0["body_vel"].clone(),
            ref0["body_ang_vel"].clone(),
        ], dim=-1), reset_ids)
        env._robot.write_joint_state_to_sim(
            env._to_isaac_order(ref0["joint_pos"].clone()),
            env._to_isaac_order(ref0["joint_vel"].clone()),
            None, reset_ids,
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
        for _ in range(horizon):
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
        tracked_steps = len(images)

        if images:
            video_dir = output_dir / "runs"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / (f"{label}.mp4" if label else f"step_{total_steps:07d}.mp4")
            import imageio
            imageio.mimsave(str(video_path), images, fps=video_fps)
            if logger:
                logger.info(f"  Video saved: {video_path} ({len(images)}/{horizon} env steps, {max_phase} MPC frames)")
        elif logger:
            logger.info(f"  No frames captured at step {total_steps}")

    except Exception as e:
        import traceback
        if logger:
            logger.info(f"  Video render failed: {e}")
            logger.info(traceback.format_exc())

    if train_ac_state is not None:
        actor_critic.load_state_dict(train_ac_state)
    if train_norm_state is not None:
        obs_normalizer.load_state_dict(train_norm_state)
    if was_training:
        actor_critic.train()
        obs_normalizer.train()
    obs_dict, _ = env.reset()
    return obs_dict["policy"], tracked_steps
