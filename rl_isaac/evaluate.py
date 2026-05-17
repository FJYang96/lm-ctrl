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
parser.add_argument("--output-json", type=str, default="",
                    help="Optional JSON path: per-phase tracking errors, "
                         "termination cause/frame, contact log, RMS metrics.")
parser.add_argument("--baseline", action="store_true",
                    help="Run PD+FF baseline (zero RL residuals). Alias of --ff-only.")
parser.add_argument("--ff-only", dest="ff_only", action="store_true",
                    help="Run feedforward-only rollout (zero RL residuals). "
                         "Used as the binding gate for accepting MPC iterations.")
parser.add_argument("--enable-dr", dest="enable_dr", action="store_true",
                    help="Enable domain randomization in eval (joint_offset, "
                         "torque_scale, friction, restitution). Default off so "
                         "eval is deterministic. Use with --seed for reproducibility.")
parser.add_argument("--seed", type=int, default=0,
                    help="Torch seed for reproducible DR sampling (default 0 — "
                         "makes friction/restitution / DR draws fully reproducible "
                         "even in clean-eval mode). Pass an explicit seed for the "
                         "5-seed robustness sweep.")
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


def execute_rollout(env, actor_critic, obs_normalizer, render=True,
                    enable_dr=False, seed=None):
    """Run deterministic rollout from phase 0 in Isaac Lab.

    The env handles PD + feedforward + residual internally via _apply_action.
    We just provide actions (or zeros for FF-only) and collect results.

    Returns (positions, images, per_step) where per_step is a dict of lists
    capturing tracking errors, torque, contact, and termination cause for env 0
    — consumed by the eval JSON writer.
    """
    from rl_isaac.rewards import TORQUE_LIMITS, TERM_CAUSE_NAMES

    device = env.device
    max_phase = env._max_phase
    horizon = env._episode_horizon
    torque_limits = TORQUE_LIMITS.to(device)

    # Seed BEFORE reset so DR samples drawn inside _reset_idx are reproducible.
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    # Reset env. _reset_idx draws joint_offset/torque_scale/friction/restitution
    # samples from the training distributions; these stay if --enable-dr,
    # otherwise we zero the two per-step DR knobs below (joint_offset and
    # torque_scale). Friction/restitution come from the env's initial reset
    # either way; with --seed they are reproducible.
    env.reset()
    env._phase[:] = 0.0
    env._prev_action[:] = 0
    env._last_torque[:] = 0
    env._first_step[:] = True
    if not enable_dr:
        env._joint_offset[:] = 0
        env._torque_scale[:] = 1.0

    # Write reference state at phase 0
    all_ids = env._robot._ALL_INDICES
    ref0 = env._sample_reference(torch.zeros(env.num_envs, device=device))
    ref_pos = ref0["body_pos"].clone() + env._env_origins
    ref_quat = ref0["body_quat"].clone()
    ref_vel = ref0["body_vel"].clone()
    ref_ang_vel = ref0["body_ang_vel"].clone()
    ref_jpos = ref0["joint_pos"].clone()
    ref_jvel = ref0["joint_vel"].clone()

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
    per_step = {
        "pos_err": [], "ori_err": [], "joint_err": [],
        "max_torque": [], "torque_saturation_frac": [],
        "actual_contact": [], "expected_contact": [],
        "nonfoot_force": [], "nonfoot_body_id": [],
        "phase_at_step": [], "ref_root_pos": [], "ref_joint_pos": [],
    }
    use_policy = actor_critic is not None
    termination = {"cause": None, "frame": None}

    # Disable auto-reset so we can capture the final frame after termination.
    # Isaac Lab's step() calls _reset_idx on done envs, which teleports the robot.
    original_reset = env._reset_idx
    env._reset_idx = lambda env_ids: None

    for step_idx in range(horizon):
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
        phase_value = float(env._phase[0].clamp(0.0, float(max_phase)).item())
        ref_step = env._sample_reference(env._phase[0:1])
        positions.append({
            "root_pos": root_pos.copy(),
            "root_quat": root_quat.copy(),
            "joint_pos": joint_pos.copy(),
            "phase": phase_value,
        })

        # Per-step diagnostics for env 0 (Phase-0.4 instrumentation)
        te = env._tracking_errors
        per_step["pos_err"].append(float(te["pos_error"][0].item()))
        per_step["ori_err"].append(float(te["ori_error"][0].item()))
        per_step["joint_err"].append(float(te["joint_error"][0].item()))
        torque_abs = env._last_torque[0].abs()
        per_step["max_torque"].append(float(torque_abs.max().item()))
        per_step["torque_saturation_frac"].append(
            float((torque_abs > 0.9 * torque_limits).float().mean().item())
        )
        actual_contact = (env._last_actual_force_per_foot[0] > 1.0).cpu().numpy().astype(int).tolist()
        expected_contact = (ref_step["contact"][0] > 0.5).cpu().numpy().astype(int).tolist()
        per_step["actual_contact"].append(actual_contact)
        per_step["expected_contact"].append(expected_contact)
        per_step["nonfoot_force"].append(float(env._last_nonfoot_force[0].item()))
        per_step["nonfoot_body_id"].append(int(env._last_nonfoot_body_id[0].item()))
        per_step["phase_at_step"].append(phase_value)
        per_step["ref_root_pos"].append(ref_step["body_pos"][0].cpu().numpy().copy())
        per_step["ref_joint_pos"].append(ref_step["joint_pos"][0].cpu().numpy().copy())

        if render:
            frame = env.render()
            if frame is not None:
                images.append(frame)

        if terminated[0] or truncated[0]:
            # Attribute the termination to a single cause (priority order:
            # nan first — most diagnostic — then physical body contact,
            # contact-schedule mismatch, then threshold sub-causes by
            # severity, finally trunc).
            priority = (
                "nan", "body", "contact",
                "thresh_torque", "thresh_rate",
                "thresh_joint", "thresh_ori", "thresh_pos",
                "trunc",
            )
            cause_masks = env._last_cause_masks
            for name in priority:
                if name in cause_masks and bool(cause_masks[name][0].item()):
                    termination["cause"] = name
                    break
            termination["frame"] = step_idx + 1
            print(f"  Env 0 terminated at env step {step_idx + 1}/{horizon} "
                  f"(phase={phase_value:.3f}/{max_phase}) "
                  f"(cause={termination['cause']})")
            break

    env._reset_idx = original_reset
    return positions, images, per_step, termination


def main():
    ff_only_mode = bool(args_cli.baseline or args_cli.ff_only)
    if not ff_only_mode and not args_cli.model_path:
        print("ERROR: provide --model-path or --ff-only/--baseline")
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
    cfg.enable_domain_randomization = bool(args_cli.enable_dr)
    cfg.clean_phase0_reset_prob = 0.0
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
    horizon = env._episode_horizon

    # Load model or use baseline
    actor_critic = None
    obs_normalizer = EmpiricalNormalization(shape=[33], until=1e8).to(device)
    obs_normalizer.eval()

    if ff_only_mode:
        print("Running FF-only rollout (zero RL residuals)...")
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
    positions, images, per_step, termination = execute_rollout(
        env, actor_critic, obs_normalizer, render=True,
        enable_dr=bool(args_cli.enable_dr), seed=args_cli.seed,
    )

    # Compute tracking errors.  The rollout runs at the env/control rate
    # (100Hz), while frames_tracked reports completed 20Hz MPC reference
    # frames for compatibility with the smoke-test acceptance gate.
    n_env_steps = len(positions)
    final_phase = positions[-1]["phase"] if positions else 0.0
    complete = (
        termination["cause"] == "trunc"
        or n_env_steps >= horizon
        or final_phase >= max_phase - 0.5 * env._phase_inc
    )
    n_tracked = int(max_phase if complete else np.floor(final_phase + 1.0e-6))
    mode_label = "FF-ONLY (PD+FF, zero residual)" if ff_only_mode else "RL POLICY"
    mode_key = "ff_only" if ff_only_mode else "policy"

    print("")
    print("=" * 50)
    print(f"ISAAC LAB TRACKING: {mode_label}")
    print("=" * 50)
    print(f"  MPC frames tracked: {n_tracked}/{max_phase}")
    print(f"  Env steps tracked: {n_env_steps}/{horizon}  phase={final_phase:.3f}")
    print(f"  Termination cause: {termination['cause']}  frame: {termination['frame']}")

    pos_rms = float("nan")
    joint_rms = float("nan")
    ori_rms = float("nan")
    if n_env_steps > 0:
        ref_positions = np.array(per_step["ref_root_pos"][:n_env_steps])
        actual_positions = np.array([p["root_pos"] for p in positions[:n_env_steps]])
        pos_rms = float(np.sqrt(np.mean((ref_positions - actual_positions) ** 2)))

        ref_joints = np.array(per_step["ref_joint_pos"][:n_env_steps])
        actual_joints = np.array([p["joint_pos"] for p in positions[:n_env_steps]])
        joint_rms = float(np.sqrt(np.mean((ref_joints - actual_joints) ** 2)))

        ori_arr = np.asarray(per_step["ori_err"][:n_env_steps], dtype=np.float64)
        ori_rms = float(np.sqrt(np.mean(ori_arr ** 2)))

        print(f"  Position RMS:  {pos_rms:.4f} m")
        print(f"  Joint RMS:     {joint_rms:.4f} rad")
        print(f"  Orientation RMS: {ori_rms:.4f} rad")
    print("=" * 50)

    # ---- Phase-0.4: write eval JSON ----
    if args_cli.output_json:
        import json
        phases_np = np.asarray(per_step["phase_at_step"], dtype=np.float64)

        # Per-phase arrays aligned to reference frame index. Pad with NaN /
        # null past the termination frame so the JSON shape is always
        # max_phase regardless of how many frames the rollout completed.
        def pad(arr, fill):
            out = list(arr)
            while len(out) < horizon:
                out.append(fill)
            return out

        def per_ref_frame(arr, fill=None):
            out = []
            for ref_frame in range(1, max_phase + 1):
                hits = np.nonzero(phases_np >= ref_frame - 1.0e-6)[0]
                if len(hits) == 0:
                    out.append(fill)
                else:
                    out.append(arr[int(hits[0])])
            return out

        out_json = {
            "mode": mode_key,
            "enable_dr": bool(args_cli.enable_dr),
            "seed": args_cli.seed,
            "frames_tracked": n_tracked,
            "max_phase": int(max_phase),
            "env_steps_tracked": n_env_steps,
            "max_env_steps": int(horizon),
            "phase_inc": float(env._phase_inc),
            "termination_cause": termination["cause"],
            "termination_frame": termination["frame"],
            "per_phase_pos_err": per_ref_frame(per_step["pos_err"], None),
            "per_phase_ori_err": per_ref_frame(per_step["ori_err"], None),
            "per_phase_joint_err": per_ref_frame(per_step["joint_err"], None),
            "per_step_pos_err": pad(per_step["pos_err"], None),
            "per_step_ori_err": pad(per_step["ori_err"], None),
            "per_step_joint_err": pad(per_step["joint_err"], None),
            "per_step_max_torque": pad(per_step["max_torque"], None),
            "per_step_torque_saturation_frac": pad(per_step["torque_saturation_frac"], None),
            "per_step_contact_actual": pad(per_step["actual_contact"], None),
            "per_step_contact_expected": pad(per_step["expected_contact"], None),
            "per_step_nonfoot_force": pad(per_step["nonfoot_force"], None),
            "per_step_nonfoot_body_id": pad(per_step["nonfoot_body_id"], None),
            "per_step_phase": pad(per_step["phase_at_step"], None),
            "rms_pos": pos_rms,
            "rms_joint": joint_rms,
            "rms_ori": ori_rms,
        }
        out_path = Path(args_cli.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(out_json, f, indent=2)
        print(f"Eval JSON saved to {out_path}")

    # Save video
    if images:
        import imageio
        video_fps = int(round(1.0 / env._env_dt))

        # Hold last frame for 2s so the video rests at the end
        last_frame = images[-1]
        for _ in range(2 * video_fps):
            images.append(last_frame)

        output_path = Path(args_cli.output_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(output_path), fps=video_fps, macro_block_size=1)
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
