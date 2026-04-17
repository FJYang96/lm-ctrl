"""Evaluate trained OPT-Mimic policy using Isaac Lab (PhysX).

Uses the same Go2TrackingEnv and physics engine as training for consistent
evaluation.  No MuJoCo dependency.

Usage (trained policy):
    /workspace/isaaclab/isaaclab.sh -p -m rl_isaac.evaluate \
        --model-path rl_isaac/trained_models/.../best_model \
        --state-traj results/.../state_traj.npy \
        --grf-traj results/.../grf_traj.npy \
        --joint-vel-traj results/.../joint_vel_traj.npy \
        --headless

Usage (PD+FF baseline, no RL residuals):
    /workspace/isaaclab/isaaclab.sh -p -m rl_isaac.evaluate \
        --baseline \
        --state-traj results/.../state_traj.npy \
        --grf-traj results/.../grf_traj.npy \
        --joint-vel-traj results/.../joint_vel_traj.npy \
        --headless
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate OPT-Mimic policy (Isaac Lab)")
parser.add_argument("--model-path", type=str, default="")
parser.add_argument("--state-traj", type=str, required=True)
parser.add_argument("--grf-traj", type=str, required=True)
parser.add_argument("--joint-vel-traj", type=str, required=True)
parser.add_argument("--contact-sequence", type=str, default="")
parser.add_argument("--output-video", type=str, default="results/rl_isaac_tracking.mp4")
parser.add_argument("--baseline", action="store_true",
                    help="Run PD+FF baseline (zero RL residuals)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Ensure repo root is on sys.path (must be AFTER AppLauncher which resets sys.path)
# AppLauncher may register a 'utils' namespace package that shadows ours,
# so we force-load our utils package into sys.modules.
import sys, importlib, importlib.util  # noqa: E402,E401
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

# -- safe to import after AppLauncher --
import numpy as np
import torch

from rsl_rl.modules import EmpiricalNormalization

from rl_isaac.env_cfg import Go2TrackingEnvCfg
from rl_isaac.tracking_env import Go2TrackingEnv
from rl_isaac.network import OPTMimicActorCritic


def load_checkpoint(model_path: str, device: str = "cpu"):
    """Load Isaac Lab checkpoint."""
    ckpt_file = Path(model_path) / "checkpoint.pt"
    if not ckpt_file.exists():
        ckpt_file = Path(model_path)
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    ac = OPTMimicActorCritic(num_obs=33, num_privileged_obs=0, num_actions=12)
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.eval()
    return ac, ckpt.get("normalizer_state_dict", None), ckpt.get("step", 0)


def execute_rollout(env, actor_critic, obs_normalizer, render=True):
    """Run deterministic rollout from phase 0 in Isaac Lab.

    The env handles PD + feedforward + residual internally via _apply_action.
    We just provide actions (or zeros for baseline) and collect results.

    Returns (positions, images).
    """
    device = env.device
    max_phase = env._max_phase

    # Reset env and force to phase 0 with no DR
    env.reset()
    env._phase[:] = 0
    env._prev_action[:] = 0
    env._last_torque[:] = 0
    env._first_step[:] = True
    env._joint_offset[:] = 0
    env._torque_scale[:] = 1.0

    # Write reference state at phase 0
    all_ids = env._robot._ALL_INDICES
    ref_pos = env._ref_body_pos[0:1].expand(env.num_envs, -1).clone() + env._env_origins
    ref_quat = env._ref_body_quat[0:1].expand(env.num_envs, -1).clone()
    ref_vel = env._ref_body_vel[0:1].expand(env.num_envs, -1).clone()
    ref_ang_vel = env._ref_body_ang_vel[0:1].expand(env.num_envs, -1).clone()
    ref_jpos = env._ref_joint_pos[0:1].expand(env.num_envs, -1).clone()
    ref_jvel = env._ref_joint_vel[0:1].expand(env.num_envs, -1).clone()

    env._robot.write_root_pose_to_sim(
        torch.cat([ref_pos, ref_quat], dim=-1), all_ids,
    )
    env._robot.write_root_velocity_to_sim(
        torch.cat([ref_vel, ref_ang_vel], dim=-1), all_ids,
    )
    env._robot.write_joint_state_to_sim(
        env._to_isaac_order(ref_jpos),
        env._to_isaac_order(ref_jvel),
        None, all_ids,
    )

    images = []
    positions = []
    use_policy = actor_critic is not None

    # Disable auto-reset so we can capture the final frame after termination.
    # Isaac Lab's step() calls _reset_idx on done envs, which teleports the robot.
    original_reset = env._reset_idx
    env._reset_idx = lambda env_ids: None

    for step_idx in range(max_phase):
        if use_policy:
            obs = env._get_observations()["policy"]
            with torch.no_grad():
                obs_norm = obs_normalizer(obs)
                actions = actor_critic.act_inference(obs_norm)
        else:
            actions = torch.zeros(env.num_envs, 12, device=device)

        _, _, terminated, truncated, _ = env.step(actions)

        # Record state after step (auto-reset is disabled so this is real)
        root_pos = (env._robot.data.root_pos_w[0] - env._env_origins[0]).cpu().numpy()
        root_quat = env._robot.data.root_quat_w[0].cpu().numpy()
        joint_pos = env._to_mpc_order(env._robot.data.joint_pos)[0].cpu().numpy()
        positions.append({
            "root_pos": root_pos.copy(),
            "root_quat": root_quat.copy(),
            "joint_pos": joint_pos.copy(),
        })

        if render:
            frame = env.render()
            if frame is not None:
                images.append(frame)

        if terminated[0] or truncated[0]:
            print(f"  Env 0 terminated at step {step_idx + 1}/{max_phase}")
            break

    env._reset_idx = original_reset
    return positions, images


def main():
    if not args_cli.baseline and not args_cli.model_path:
        print("ERROR: provide --model-path or --baseline")
        simulation_app.close()
        return

    # Create env with 1 env for clean evaluation
    from isaaclab.envs.common import ViewerCfg

    cfg = Go2TrackingEnvCfg()
    cfg.scene.num_envs = 1
    cfg.state_traj_path = args_cli.state_traj
    cfg.grf_traj_path = args_cli.grf_traj
    cfg.joint_vel_traj_path = args_cli.joint_vel_traj
    cfg.contact_sequence_path = (
        args_cli.contact_sequence if args_cli.contact_sequence else ""
    )
    # Close-up camera with fixed world-frame position
    cfg.viewer = ViewerCfg(
        eye=(2.5, 2.5, 1.5),
        lookat=(0.0, 0.0, 0.4),
        resolution=(1920, 1088),  # divisible by 16 for codec compatibility
        origin_type="world",
    )

    env = Go2TrackingEnv(cfg, render_mode="rgb_array")
    device = env.device
    max_phase = env._max_phase

    # Load model or use baseline
    actor_critic = None
    obs_normalizer = EmpiricalNormalization(shape=[33], until=1e8).to(device)
    obs_normalizer.eval()

    if args_cli.baseline:
        print("Running PD+FF baseline (zero RL residuals)...")
    else:
        actor_critic, normalizer_state, step = load_checkpoint(
            args_cli.model_path, str(device),
        )
        actor_critic = actor_critic.to(device)
        if normalizer_state:
            obs_normalizer.load_state_dict(normalizer_state)
            obs_normalizer.eval()
        print(f"Loaded checkpoint at step {step}")

    # Run rollout
    print("Running evaluation rollout...")
    positions, images = execute_rollout(env, actor_critic, obs_normalizer, render=True)

    # Compute tracking errors
    n_tracked = len(positions)
    mode = "BASELINE (PD+FF)" if args_cli.baseline else "RL POLICY"

    print("")
    print("=" * 50)
    print(f"ISAAC LAB TRACKING: {mode}")
    print("=" * 50)
    print(f"  Steps tracked: {n_tracked}/{max_phase}")

    if n_tracked > 0:
        # Phase is incremented before reward/done check, so the reference
        # position at phase k+1 is what we compare against after step k.
        ref_idx = min(n_tracked, max_phase)
        ref_positions = env._ref_body_pos[1:ref_idx + 1].cpu().numpy()
        actual_positions = np.array([p["root_pos"] for p in positions[:ref_idx]])
        pos_rms = np.sqrt(np.mean((ref_positions - actual_positions) ** 2))

        ref_joints = env._ref_joint_pos[1:ref_idx + 1].cpu().numpy()
        actual_joints = np.array([p["joint_pos"] for p in positions[:ref_idx]])
        joint_rms = np.sqrt(np.mean((ref_joints - actual_joints) ** 2))

        print(f"  Position RMS:  {pos_rms:.4f} m")
        print(f"  Joint RMS:     {joint_rms:.4f} rad")
    print("=" * 50)

    # Save video
    if images:
        import imageio

        # Hold last frame for 2s so the video rests at the end
        last_frame = images[-1]
        for _ in range(100):  # 2s at 50fps
            images.append(last_frame)

        output_path = Path(args_cli.output_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(output_path), fps=50, macro_block_size=1)
        for frame in images:
            writer.append_data(frame)
        writer.close()
        print(f"Video saved to {output_path}")
    else:
        print("No frames captured — video not saved.")
        print("  (Tip: try running without --headless, or with --enable_cameras)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
