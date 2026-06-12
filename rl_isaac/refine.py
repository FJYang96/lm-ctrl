"""MPPI-style open-loop torque refinement for LLM-generated references.

This module inserts a trajectory refinement stage between planning output and
policy learning by optimizing an open-loop torque sequence against the same
tracking reward terms used by PPO training.
"""
# ruff: noqa: I001

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
import types
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Refine nominal trajectory with MPPI open-loop torques (Isaac Lab)"
)
parser.add_argument("--state-traj", type=str, default="")
parser.add_argument("--grf-traj", type=str, default="")
parser.add_argument("--joint-vel-traj", type=str, default="")
parser.add_argument("--contact-sequence", type=str, default="")
parser.add_argument("--traj-dir", type=str, default="")
parser.add_argument("--iter-num", type=int, default=-1)
parser.add_argument("--output-dir", type=str, default="rl_isaac/refine_output")
parser.add_argument("--run-tag", type=str, default="")
parser.add_argument("--mppi-iters", type=int, default=200)
parser.add_argument("--num-samples", type=int, default=512)
parser.add_argument("--noise-std", type=float, default=3.0)
parser.add_argument("--temperature", type=float, default=2.0)
parser.add_argument("--elite-fraction", type=float, default=0.25)
parser.add_argument("--momentum", type=float, default=0.0)
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--termination-penalty", type=float, default=-5.0)
parser.add_argument("--termination-weight-start", type=float, default=0.0)
parser.add_argument("--termination-weight-end", type=float, default=0.0)
parser.add_argument("--termination-anneal-iters", type=int, default=0)
parser.add_argument("--torque-limit-scale", type=float, default=1.0)
parser.add_argument(
    "--source-dt",
    type=float,
    default=None,
    help="MPC planning timestep in seconds (default: infer from go2_config / metadata).",
)
parser.add_argument(
    "--control-dt",
    type=float,
    default=None,
    help="MPPI / Isaac control timestep in seconds (overrides --ref-rate-hz).",
)
parser.add_argument(
    "--ref-rate-hz",
    type=int,
    default=100,
    choices=(100, 200),
    help="Reference control rate in Hz when --control-dt is not set (default: 100).",
)
parser.add_argument("--render-best-every", type=int, default=10)
parser.add_argument("--save-best-npy-every", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Ensure repo root is on sys.path (must be AFTER AppLauncher which resets sys.path)
import importlib.util  # noqa: E402

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
for mod_name, attrs in [
    ("mujoco.viewer", {"Handle": type("Handle", (), {})}),
    ("glfw", {"_glfw": True}),
]:
    if mod_name not in sys.modules:
        module = types.ModuleType(mod_name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[mod_name] = module
if "glfw.library" not in sys.modules:
    sys.modules["glfw.library"] = types.ModuleType("glfw.library")

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs.common import ViewerCfg  # noqa: E402
from mpc.dynamics.model import KinoDynamic_Model  # noqa: E402

from rl_isaac.env_cfg import Go2TrackingEnvCfg  # noqa: E402
from rl_isaac.feedforward import FeedforwardComputer  # noqa: E402
from rl_isaac.reference import ReferenceTrajectory  # noqa: E402
from rl_isaac.rewards import (  # noqa: E402
    ACTION_LIMIT,
    KD,
    KP,
    TORQUE_LIMITS,
    W_ORI_INC_MPPI,
    compute_rewards,
    compute_orientation_increment_reward,
    compute_tracking_errors,
)
from rl_isaac.tracking_env import Go2TrackingEnv  # noqa: E402
from rl_isaac.upsample_reference import (  # noqa: E402
    resolve_ref_control_dt,
    resolve_source_dt,
    save_reference_arrays,
    upsample_reference_arrays,
)


@dataclass
class RefPaths:
    state_traj: str
    grf_traj: str
    joint_vel_traj: str
    contact_sequence: str
    traj_dir: str
    iter_num: int


def _resolve_reference_paths(args: argparse.Namespace) -> RefPaths:
    if args.traj_dir:
        if args.iter_num < 0:
            raise ValueError("--iter-num must be set when --traj-dir is provided.")
        traj_dir = Path(args.traj_dir)
        state = str(traj_dir / f"state_traj_iter_{args.iter_num}.npy")
        grf = str(traj_dir / f"grf_traj_iter_{args.iter_num}.npy")
        jvel = str(traj_dir / f"joint_vel_traj_iter_{args.iter_num}.npy")
        contact = str(traj_dir / f"contact_sequence_iter_{args.iter_num}.npy")
    else:
        if not (args.state_traj and args.grf_traj and args.joint_vel_traj):
            raise ValueError(
                "Provide --traj-dir/--iter-num or explicit --state-traj/--grf-traj/--joint-vel-traj."
            )
        state = args.state_traj
        grf = args.grf_traj
        jvel = args.joint_vel_traj
        contact = args.contact_sequence

    for required_path in [state, grf, jvel]:
        if not Path(required_path).exists():
            raise FileNotFoundError(
                f"Required trajectory file not found: {required_path}"
            )
    if contact and not Path(contact).exists():
        contact = ""

    return RefPaths(
        state_traj=state,
        grf_traj=grf,
        joint_vel_traj=jvel,
        contact_sequence=contact,
        traj_dir=args.traj_dir,
        iter_num=args.iter_num,
    )


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_env(
    paths: RefPaths, num_envs: int, render: bool, headless: bool = False,
    control_dt: float | None = None,
) -> Go2TrackingEnv:
    import go2_config

    if control_dt is None:
        control_dt = go2_config.default_ref_control_dt
    cfg = Go2TrackingEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.state_traj_path = paths.state_traj
    cfg.grf_traj_path = paths.grf_traj
    cfg.joint_vel_traj_path = paths.joint_vel_traj
    cfg.contact_sequence_path = paths.contact_sequence
    cfg.control_dt = control_dt
    # In headless mode we avoid viewer/window setup; env.render() can still
    # produce RGB arrays when camera rendering is enabled.
    if render and not headless:
        cfg.viewer = ViewerCfg(
            eye=(2.5, 2.5, 1.5),
            lookat=(0.0, 0.0, 0.4),
            resolution=(1920, 1088),
            origin_type="world",
        )
    return Go2TrackingEnv(cfg, render_mode="rgb_array" if render else None)


def _set_nominal_materials(env: Go2TrackingEnv) -> None:
    mat = env._robot.root_physx_view.get_material_properties()
    env_ids_cpu = env._robot._ALL_INDICES.cpu()
    n_shapes = mat.shape[1]
    mat[env_ids_cpu, :, 0] = 0.8 * torch.ones(
        (len(env_ids_cpu), n_shapes), device=mat.device
    )
    mat[env_ids_cpu, :, 1] = 0.8 * torch.ones(
        (len(env_ids_cpu), n_shapes), device=mat.device
    )
    mat[env_ids_cpu, :, 2] = torch.zeros(
        (len(env_ids_cpu), n_shapes), device=mat.device
    )
    env._robot.root_physx_view.set_material_properties(mat, env_ids_cpu)


def _reset_env_to_reference_start(env: Go2TrackingEnv) -> None:
    env.reset()
    env._phase[:] = 0
    env._prev_action[:] = 0.0
    env._last_torque[:] = 0.0
    env._first_step[:] = True
    env._joint_offset[:] = 0.0
    env._torque_scale[:] = 1.0
    _set_nominal_materials(env)

    all_ids = env._robot._ALL_INDICES
    ref_pos = env._ref_body_pos[0:1].expand(env.num_envs, -1).clone() + env._env_origins
    ref_quat = env._ref_body_quat[0:1].expand(env.num_envs, -1).clone()
    ref_vel = env._ref_body_vel[0:1].expand(env.num_envs, -1).clone()
    ref_ang = env._ref_body_ang_vel[0:1].expand(env.num_envs, -1).clone()
    ref_jpos = env._ref_joint_pos[0:1].expand(env.num_envs, -1).clone()
    ref_jvel = env._ref_joint_vel[0:1].expand(env.num_envs, -1).clone()

    env._robot.write_root_pose_to_sim(torch.cat([ref_pos, ref_quat], dim=-1), all_ids)
    env._robot.write_root_velocity_to_sim(
        torch.cat([ref_vel, ref_ang], dim=-1), all_ids
    )
    env._robot.write_joint_state_to_sim(
        env._to_isaac_order(ref_jpos), env._to_isaac_order(ref_jvel), None, all_ids
    )


def _torque_to_action(
    env: Go2TrackingEnv, desired_torque: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    phase = env._phase.clamp(0, env._max_phase - 1).long()
    ref_jpos = env._ref_joint_pos[phase]
    ref_jvel = env._ref_joint_vel[phase]
    ff_torque = env._ref_ff_torques[phase]
    act_jpos = env._to_mpc_order(env._robot.data.joint_pos)
    act_jvel = env._to_mpc_order(env._robot.data.joint_vel)

    # Invert env PD+FF law to produce action residual that realizes desired torque.
    action_scaled = (
        (desired_torque - KD * (ref_jvel - act_jvel) - ff_torque) / KP
        - (ref_jpos - act_jpos)
        - env._joint_offset
    )
    action = torch.clamp(action_scaled / ACTION_LIMIT, -1.0, 1.0)
    return action, action * ACTION_LIMIT


def _score_population(
    env: Go2TrackingEnv,
    candidate_torques: torch.Tensor,
    gamma: float,
    termination_penalty: float,
    termination_weight: float,
) -> torch.Tensor:
    n_samples, horizon, _ = candidate_torques.shape
    if env.num_envs != n_samples:
        raise ValueError(
            f"env.num_envs={env.num_envs} must equal num_samples={n_samples}"
        )
    if env._max_phase != horizon:
        raise ValueError(
            f"Candidate horizon ({horizon}) must match reference horizon ({env._max_phase})"
        )

    _reset_env_to_reference_start(env)
    prev_action_scaled = torch.zeros(n_samples, 12, device=env.device)
    alive = torch.ones(n_samples, dtype=torch.bool, device=env.device)
    returns = torch.zeros(n_samples, device=env.device)
    gamma_acc = 1.0

    original_reset = env._reset_idx
    env._reset_idx = lambda env_ids: None
    try:
        for step in range(horizon):
            prev_phase = env._phase.clamp(0, env._max_phase - 1).long()
            prev_actual_quat = env._robot.data.root_quat_w.clone()
            actions, action_scaled = _torque_to_action(
                env, candidate_torques[:, step, :]
            )
            _, _, terminated, truncated, _ = env.step(actions)
            phase = env._phase.clamp(0, env._max_phase - 1).long()

            tracking_errors = compute_tracking_errors(
                env._ref_body_pos[phase],
                env._ref_body_quat[phase],
                env._ref_joint_pos[phase],
                env._robot.data.root_pos_w - env._env_origins,
                env._robot.data.root_quat_w,
                env._to_mpc_order(env._robot.data.joint_pos),
                action_scaled,
                prev_action_scaled,
                env._last_torque,
            )
            reward, _ = compute_rewards(tracking_errors)
            ori_inc_reward = compute_orientation_increment_reward(
                env._ref_body_quat[prev_phase],
                env._ref_body_quat[phase],
                prev_actual_quat,
                env._robot.data.root_quat_w,
            )
            reward = reward + W_ORI_INC_MPPI * ori_inc_reward
            # termination_weight = 1.0 -> strict RL-like termination behavior
            # termination_weight = 0.0 -> pure fixed-horizon accumulation
            effective_alive = (1.0 - termination_weight) + (
                termination_weight * alive.float()
            )
            returns += effective_alive * gamma_acc * reward

            done = terminated | truncated
            if termination_penalty != 0.0 and termination_weight > 0.0:
                returns += (
                    termination_weight
                    * alive.float()
                    * done.float()
                    * termination_penalty
                )
            alive &= ~done
            prev_action_scaled = action_scaled
            gamma_acc *= gamma
    finally:
        env._reset_idx = original_reset

    return returns


def _render_rollout_video(
    env: Go2TrackingEnv,
    torque_seq: torch.Tensor,
    out_path: Path,
) -> dict[str, float | int]:
    _reset_env_to_reference_start(env)
    prev_action_scaled = torch.zeros(1, 12, device=env.device)
    total_return = 0.0
    frames = []

    original_reset = env._reset_idx
    env._reset_idx = lambda env_ids: None
    try:
        for step in range(env._max_phase):
            prev_phase = env._phase.clamp(0, env._max_phase - 1).long()
            prev_actual_quat = env._robot.data.root_quat_w.clone()
            actions, action_scaled = _torque_to_action(env, torque_seq[step : step + 1])
            _, _, _, _, _ = env.step(actions)
            phase = env._phase.clamp(0, env._max_phase - 1).long()

            tracking_errors = compute_tracking_errors(
                env._ref_body_pos[phase],
                env._ref_body_quat[phase],
                env._ref_joint_pos[phase],
                env._robot.data.root_pos_w - env._env_origins,
                env._robot.data.root_quat_w,
                env._to_mpc_order(env._robot.data.joint_pos),
                action_scaled,
                prev_action_scaled,
                env._last_torque,
            )
            reward, _ = compute_rewards(tracking_errors)
            ori_inc_reward = compute_orientation_increment_reward(
                env._ref_body_quat[prev_phase],
                env._ref_body_quat[phase],
                prev_actual_quat,
                env._robot.data.root_quat_w,
            )
            reward = reward + W_ORI_INC_MPPI * ori_inc_reward
            total_return += float(reward[0].item())
            prev_action_scaled = action_scaled

            frame = env.render()
            if frame is not None:
                frames.append(frame)
    finally:
        env._reset_idx = original_reset

    if frames:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(out_path), frames, fps=50)
    return {"return": total_return, "frames": len(frames)}


def _write_metrics_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _effective_run_tag(args: argparse.Namespace) -> str:
    if args.run_tag:
        return args.run_tag
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    iter_tag = f"iter_{args.iter_num}" if args.iter_num >= 0 else "iter_unknown"
    return f"refine_{iter_tag}_{stamp}"


def _termination_weight_for_iter(args: argparse.Namespace, iter_idx: int) -> float:
    anneal_iters = (
        args.termination_anneal_iters
        if args.termination_anneal_iters > 0
        else args.mppi_iters
    )
    if anneal_iters <= 1:
        return float(args.termination_weight_end)
    t = min(max(iter_idx, 0), anneal_iters - 1) / float(anneal_iters - 1)
    return float(
        args.termination_weight_start
        + t * (args.termination_weight_end - args.termination_weight_start)
    )


def refine(args: argparse.Namespace) -> None:
    if args.num_samples < 2:
        raise ValueError("--num-samples must be >= 2.")
    if args.mppi_iters < 1:
        raise ValueError("--mppi-iters must be >= 1.")
    if not (0.0 < args.elite_fraction <= 1.0):
        raise ValueError("--elite-fraction must be in (0, 1].")
    if not (0.0 <= args.momentum < 1.0):
        raise ValueError("--momentum must be in [0, 1).")
    if args.temperature <= 0.0:
        raise ValueError("--temperature must be > 0.")
    if args.torque_limit_scale <= 0.0:
        raise ValueError("--torque-limit-scale must be > 0.")
    if not (0.0 <= args.termination_weight_start <= 1.0):
        raise ValueError("--termination-weight-start must be in [0, 1].")
    if not (0.0 <= args.termination_weight_end <= 1.0):
        raise ValueError("--termination-weight-end must be in [0, 1].")
    if args.termination_anneal_iters < 0:
        raise ValueError("--termination-anneal-iters must be >= 0.")

    _seed_everything(args.seed)
    paths = _resolve_reference_paths(args)
    run_dir = Path(args.output_dir) / _effective_run_tag(args)
    video_dir = run_dir / "videos"
    iter_best_dir = run_dir / "iter_best_torque"
    run_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    iter_best_dir.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("MPPI trajectory refinement")
    print("============================================================")
    print(f"state_traj:      {paths.state_traj}")
    print(f"grf_traj:        {paths.grf_traj}")
    print(f"joint_vel_traj:  {paths.joint_vel_traj}")
    print(f"contact_seq:     {paths.contact_sequence or '(derived from GRF)'}")
    print(f"output_dir:      {run_dir}")
    print(f"seed:            {args.seed}")
    print(
        "termination_w:   "
        f"{args.termination_weight_start:.2f} -> {args.termination_weight_end:.2f} "
        f"(anneal iters: {args.termination_anneal_iters or args.mppi_iters})"
    )
    print("============================================================")

    t0 = time.time()
    source_dt = resolve_source_dt(args.source_dt, paths.traj_dir or None)
    control_dt = resolve_ref_control_dt(args.control_dt, ref_rate_hz=args.ref_rate_hz)

    contact_seq = (
        np.load(paths.contact_sequence)
        if paths.contact_sequence
        else None
    )
    state_raw = np.load(paths.state_traj)
    jvel_raw = np.load(paths.joint_vel_traj)
    grf_raw = np.load(paths.grf_traj)
    state_up, jvel_up, grf_up, contact_up, upsample_meta = upsample_reference_arrays(
        state_raw,
        jvel_raw,
        grf_raw,
        contact_seq,
        source_dt=source_dt,
        target_dt=control_dt,
    )
    upsampled_dir = run_dir / "upsampled_reference"
    upsampled_paths = save_reference_arrays(
        upsampled_dir,
        state_up,
        jvel_up,
        grf_up,
        contact_up,
        upsample_meta,
    )
    env_paths = RefPaths(
        state_traj=upsampled_paths["state_traj"],
        grf_traj=upsampled_paths["grf_traj"],
        joint_vel_traj=upsampled_paths["joint_vel_traj"],
        contact_sequence=upsampled_paths.get("contact_sequence", ""),
        traj_dir=paths.traj_dir,
        iter_num=paths.iter_num,
    )

    print(
        "Reference upsampling: "
        f"source_dt={source_dt:.4f}s ({upsample_meta['source_horizon']} steps) -> "
        f"control_dt={control_dt:.4f}s ({upsample_meta['target_horizon']} steps)"
    )
    if upsample_meta["upsampled"]:
        print(f"Upsampled reference saved to: {upsampled_dir}")
    else:
        print("Reference already at control dt; skipping resample.")

    ref = ReferenceTrajectory(
        state_traj=state_up,
        joint_vel_traj=jvel_up,
        grf_traj=grf_up,
        contact_sequence=contact_up,
        control_dt=control_dt,
    )
    ff_seed = (
        FeedforwardComputer(KinoDynamic_Model())
        .precompute_trajectory(ref)
        .astype(np.float32)
    )
    ref.set_feedforward(ff_seed)
    horizon = ref.max_phase
    np.save(upsampled_dir / "upsampled_feedforward_torque_traj.npy", ff_seed)

    is_headless = bool(getattr(args, "headless", False))
    cameras_enabled = bool(getattr(args, "enable_cameras", False))
    rendering_enabled = cameras_enabled
    if not rendering_enabled:
        print("Rendering disabled: run with --enable_cameras to save rollout videos.")

    # Isaac Lab permits only one simulation context per process.
    # Use a single env for both batched scoring and optional video capture.
    env = _build_env(
        env_paths,
        num_envs=args.num_samples,
        render=rendering_enabled,
        headless=is_headless,
        control_dt=control_dt,
    )
    device = env.device

    torque_limits = TORQUE_LIMITS.to(device) * args.torque_limit_scale
    mean_torque = torch.tensor(ff_seed, dtype=torch.float32, device=device)
    global_best_torque = mean_torque.clone()
    global_best_score = -float("inf")
    metrics: list[dict[str, float | int]] = []

    seed_termination_weight = _termination_weight_for_iter(args, 0)
    seed_eval = _score_population(
        env,
        mean_torque.unsqueeze(0).repeat(args.num_samples, 1, 1),
        gamma=args.gamma,
        termination_penalty=args.termination_penalty,
        termination_weight=seed_termination_weight,
    )
    seed_score = float(seed_eval[0].item())
    global_best_score = seed_score
    print(f"Initial feedforward score: {seed_score:.4f}")

    for iter_idx in range(args.mppi_iters):
        termination_weight = _termination_weight_for_iter(args, iter_idx)
        noise = (
            torch.randn(args.num_samples, horizon, 12, device=device) * args.noise_std
        )
        candidates = mean_torque.unsqueeze(0) + noise
        candidates[0] = mean_torque  # include current nominal each iteration
        candidates = torch.clamp(
            candidates, -torque_limits.view(1, 1, 12), torque_limits.view(1, 1, 12)
        )

        scores = _score_population(
            env,
            candidates,
            gamma=args.gamma,
            termination_penalty=args.termination_penalty,
            termination_weight=termination_weight,
        )
        iter_best_idx = int(torch.argmax(scores).item())
        iter_best_score = float(scores[iter_best_idx].item())
        iter_mean_score = float(scores.mean().item())

        elite_count = max(1, int(args.num_samples * args.elite_fraction))
        elite_scores, elite_indices = torch.topk(scores, elite_count)
        elite_candidates = candidates[elite_indices]
        logits = (elite_scores - torch.max(elite_scores)) / args.temperature
        weights = torch.softmax(logits, dim=0)
        if elite_count > 1:
            weight_entropy = float(
                (-weights * torch.log(weights + 1e-8)).sum().item()
                / np.log(elite_count)
            )
        else:
            weight_entropy = 0.0
        updated_mean = torch.sum(weights.view(-1, 1, 1) * elite_candidates, dim=0)
        mean_torque = args.momentum * mean_torque + (1.0 - args.momentum) * updated_mean
        mean_torque = torch.clamp(
            mean_torque, -torque_limits.view(1, 12), torque_limits.view(1, 12)
        )

        iter_best_torque = candidates[iter_best_idx].clone()
        if iter_best_score > global_best_score:
            global_best_score = iter_best_score
            global_best_torque = iter_best_torque.clone()

        render_return = None
        render_frames = 0
        if (
            rendering_enabled
            and args.render_best_every > 0
            and ((iter_idx + 1) % args.render_best_every == 0)
        ):
            video_path = video_dir / f"best_iter_{iter_idx + 1:04d}.mp4"
            render_info = _render_rollout_video(env, iter_best_torque, video_path)
            render_return = float(render_info["return"])
            render_frames = int(render_info["frames"])

        if args.save_best_npy_every > 0 and (
            (iter_idx + 1) % args.save_best_npy_every == 0
        ):
            np.save(
                iter_best_dir / f"best_iter_{iter_idx + 1:04d}.npy",
                iter_best_torque.detach().cpu().numpy(),
            )

        metrics.append(
            {
                "iteration": iter_idx + 1,
                "score_best": iter_best_score,
                "score_mean": iter_mean_score,
                "score_global_best": global_best_score,
                "elite_count": elite_count,
                "weight_entropy": weight_entropy,
                "termination_weight": termination_weight,
                "render_return": float(render_return)
                if render_return is not None
                else float("nan"),
                "render_frames": render_frames,
            }
        )
        print(
            f"[iter {iter_idx + 1:04d}/{args.mppi_iters:04d}] "
            f"best={iter_best_score:.4f} mean={iter_mean_score:.4f} "
            f"global={global_best_score:.4f} term_w={termination_weight:.3f}"
        )

    if rendering_enabled:
        best_video_info = _render_rollout_video(
            env, global_best_torque, video_dir / "best_overall.mp4"
        )
    else:
        best_video_info = {"return": float("nan"), "frames": 0}

    refined = global_best_torque.detach().cpu().numpy()
    mean_np = mean_torque.detach().cpu().numpy()
    np.save(run_dir / "refined_torque_traj.npy", refined)
    np.save(run_dir / "refined_open_loop_torque_traj.npy", refined)
    np.save(run_dir / "final_mean_torque_traj.npy", mean_np)
    np.save(run_dir / "seed_feedforward_torque_traj.npy", ff_seed)
    np.save(
        run_dir / "iter_best_scores.npy",
        np.array([m["score_best"] for m in metrics], dtype=np.float32),
    )
    _write_metrics_csv(run_dir / "mppi_metrics.csv", metrics)

    summary = {
        "run_tag": run_dir.name,
        "created_at": datetime.now().isoformat(),
        "elapsed_sec": time.time() - t0,
        "paths": asdict(paths),
        "upsampled_reference": {
            "paths": upsampled_paths,
            "metadata": upsample_meta,
            "source_dt": source_dt,
            "control_dt": control_dt,
        },
        "settings": {
            "source_dt": source_dt,
            "control_dt": control_dt,
            "mppi_iters": args.mppi_iters,
            "num_samples": args.num_samples,
            "noise_std": args.noise_std,
            "temperature": args.temperature,
            "elite_fraction": args.elite_fraction,
            "momentum": args.momentum,
            "gamma": args.gamma,
            "termination_penalty": args.termination_penalty,
            "termination_weight_start": args.termination_weight_start,
            "termination_weight_end": args.termination_weight_end,
            "termination_anneal_iters": args.termination_anneal_iters,
            "torque_limit_scale": args.torque_limit_scale,
            "seed": args.seed,
            "render_best_every": args.render_best_every,
        },
        "horizon": horizon,
        "seed_score": seed_score,
        "best_score": global_best_score,
        "best_video": {
            "path": str(video_dir / "best_overall.mp4") if rendering_enabled else "",
            "return": float(best_video_info["return"]),
            "frames": int(best_video_info["frames"]),
        },
        "artifacts": {
            "refined_torque_traj": str(run_dir / "refined_torque_traj.npy"),
            "refined_open_loop_torque_traj": str(
                run_dir / "refined_open_loop_torque_traj.npy"
            ),
            "final_mean_torque_traj": str(run_dir / "final_mean_torque_traj.npy"),
            "seed_feedforward_torque_traj": str(
                run_dir / "seed_feedforward_torque_traj.npy"
            ),
            "upsampled_feedforward_torque_traj": str(
                upsampled_dir / "upsampled_feedforward_torque_traj.npy"
            ),
            "metrics_csv": str(run_dir / "mppi_metrics.csv"),
            "videos_dir": str(video_dir),
        },
    }
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("============================================================")
    print("Refinement complete")
    print(f"seed_score:  {seed_score:.4f}")
    print(f"best_score:  {global_best_score:.4f}")
    print(f"output_dir:  {run_dir}")
    print("============================================================")

    env.close()


def main() -> None:
    exit_code = 0
    try:
        refine(args_cli)
    except Exception:
        exit_code = 1
        # Isaac app shutdown can obscure Python exceptions in headless mode.
        # Print the traceback explicitly so run_refine.sh log captures it.
        traceback.print_exc()
    finally:
        simulation_app.close()
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
