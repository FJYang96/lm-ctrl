"""Fresh-env checkpoint selection for OPT-Mimic smoke tests."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Select best checkpoint by clean fresh-env rollout")
parser.add_argument("--run-dir", type=str, required=True)
parser.add_argument("--state-traj", type=str, required=True)
parser.add_argument("--grf-traj", type=str, required=True)
parser.add_argument("--joint-vel-traj", type=str, required=True)
parser.add_argument("--contact-sequence", type=str, default="")
parser.add_argument("--robust-seeds", type=int, default=5)
parser.add_argument("--robust-top-k", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys, importlib.util  # noqa: E402

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

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.modules import EmpiricalNormalization  # noqa: E402

from rl_isaac.env_cfg import Go2TrackingEnvCfg  # noqa: E402
from rl_isaac.network import OPTMimicActorCritic  # noqa: E402
from rl_isaac.tracking_env import Go2TrackingEnv  # noqa: E402
from rl_isaac.rewards import TERM_CAUSE_NAMES  # noqa: E402


def load_policy(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ac = OPTMimicActorCritic(num_obs=33, num_privileged_obs=0, num_actions=12).to(device)
    ac.load_state_dict(ckpt["model_state_dict"])
    ac.eval()
    norm = EmpiricalNormalization(shape=[33], until=1e8).to(device)
    if "normalizer_state_dict" in ckpt:
        norm.load_state_dict(ckpt["normalizer_state_dict"])
    norm.eval()
    return ac, norm, int(ckpt.get("step", 0))


def rollout(env, actor_critic, obs_normalizer, *, enable_dr: bool = False, seed: int | None = None):
    device = env.device
    max_phase = env._max_phase
    horizon = env._episode_horizon

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.device(device).type == "cuda":
            torch.cuda.manual_seed_all(seed)
    env.cfg.enable_domain_randomization = enable_dr
    env.cfg.phase0_reset_prob = 1.0
    env.cfg.clean_phase0_reset_prob = 0.0
    env.reset()
    env._phase[:] = 0.0
    env._prev_action[:] = 0
    env._last_torque[:] = 0
    env._first_step[:] = True
    if not enable_dr:
        env._joint_offset[:] = 0
        env._torque_scale[:] = 1.0

    all_ids = env._robot._ALL_INDICES
    ref0 = env._sample_reference(torch.zeros(env.num_envs, device=device))
    env._robot.write_root_pose_to_sim(
        torch.cat([ref0["body_pos"].clone() + env._env_origins, ref0["body_quat"].clone()], dim=-1),
        all_ids,
    )
    env._robot.write_root_velocity_to_sim(
        torch.cat([ref0["body_vel"].clone(), ref0["body_ang_vel"].clone()], dim=-1),
        all_ids,
    )
    env._robot.write_joint_state_to_sim(
        env._to_isaac_order(ref0["joint_pos"].clone()),
        env._to_isaac_order(ref0["joint_vel"].clone()),
        None,
        all_ids,
    )

    original_reset = env._reset_idx
    env._reset_idx = lambda env_ids: None
    positions = []
    pos_err = []
    ori_err = []
    termination = {"cause": None, "frame": None}
    try:
        for step_idx in range(horizon):
            obs = env._get_observations()["policy"]
            with torch.no_grad():
                actions = actor_critic.act_inference(obs_normalizer(obs))
            _, _, terminated, truncated, _ = env.step(actions)

            phase_value = float(env._phase[0].clamp(0.0, float(max_phase)).item())
            positions.append(phase_value)
            pos_err.append(float(env._tracking_errors["pos_error"][0].item()))
            ori_err.append(float(env._tracking_errors["ori_error"][0].item()))

            if terminated[0] or truncated[0]:
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
                break
    finally:
        env._reset_idx = original_reset

    n_steps = len(positions)
    final_phase = positions[-1] if positions else 0.0
    complete = (
        termination["cause"] == "trunc"
        or n_steps >= horizon
        or final_phase >= max_phase - 0.5 * env._phase_inc
    )
    frames = int(max_phase if complete else np.floor(final_phase + 1.0e-6))
    rms_pos = float(np.sqrt(np.mean(np.square(pos_err)))) if pos_err else float("nan")
    rms_ori = float(np.sqrt(np.mean(np.square(ori_err)))) if ori_err else float("nan")
    return {
        "frames_tracked": frames,
        "max_phase": int(max_phase),
        "env_steps_tracked": n_steps,
        "max_env_steps": int(horizon),
        "termination_cause": termination["cause"],
        "termination_frame": termination["frame"],
        "rms_pos_error_norm": rms_pos,
        "rms_ori_error_norm": rms_ori,
    }


def candidate_paths(run_dir: Path) -> list[Path]:
    out = []
    for rel in ["best_model/checkpoint.pt", "final_model/checkpoint.pt", "best_training_model/checkpoint.pt"]:
        p = run_dir / rel
        if p.exists():
            out.append(p)
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        out.extend(sorted(ckpt_dir.glob("*/checkpoint.pt")))
    video_ckpt_dir = run_dir / "video_checkpoints"
    if video_ckpt_dir.exists():
        out.extend(sorted(video_ckpt_dir.glob("*/checkpoint.pt")))
    seen = set()
    unique = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def main():
    cfg = Go2TrackingEnvCfg()
    cfg.scene.num_envs = 1
    cfg.state_traj_path = args_cli.state_traj
    cfg.grf_traj_path = args_cli.grf_traj
    cfg.joint_vel_traj_path = args_cli.joint_vel_traj
    cfg.contact_sequence_path = args_cli.contact_sequence if args_cli.contact_sequence else ""
    cfg.enable_domain_randomization = False
    cfg.clean_phase0_reset_prob = 0.0
    cfg.phase0_reset_prob = 1.0
    env = Go2TrackingEnv(cfg, render_mode=None)
    run_dir = Path(args_cli.run_dir)
    curriculum_switch_step = None
    curriculum_path = run_dir / "curriculum.json"
    if curriculum_path.exists():
        with curriculum_path.open() as f:
            curriculum = json.load(f)
        if curriculum.get("switched") and curriculum.get("switch_step") is not None:
            curriculum_switch_step = int(curriculum["switch_step"])
    results = []

    loaded = {}
    for ckpt_path in candidate_paths(run_dir):
        actor_critic, normalizer, step = load_policy(ckpt_path, env.device)
        loaded[str(ckpt_path)] = (actor_critic, normalizer)
        result = rollout(env, actor_critic, normalizer)
        result.update({
            "path": str(ckpt_path.parent),
            "checkpoint": str(ckpt_path),
            "step": step,
        })
        results.append(result)
        print(
            f"{ckpt_path.parent.name}: {result['frames_tracked']}/{result['max_phase']} "
            f"cause={result['termination_cause']} "
            f"pos={result['rms_pos_error_norm']:.4f} ori={result['rms_ori_error_norm']:.4f}"
        )

    if not results:
        raise RuntimeError(f"No checkpoints found in {run_dir}")

    def clean_key(r):
        clean_ok = int(r["termination_cause"] == "trunc" and r["frames_tracked"] == r["max_phase"])
        # evaluate.py reports component-wise position RMS, while this selector
        # stores RMS of the 3D error norm. Convert the selector value before
        # applying the smoke-test position gate so the chosen checkpoint matches
        # the official JSON criteria.
        pos_component_rms = float(r["rms_pos_error_norm"]) / float(np.sqrt(3.0))
        rms_ok = int(pos_component_rms < 0.10 and r["rms_ori_error_norm"] < 0.25)
        post_curriculum = int(
            curriculum_switch_step is not None
            and int(r["step"]) >= curriculum_switch_step
        )
        return (
            clean_ok,
            rms_ok,
            post_curriculum,
            int(r["frames_tracked"]),
            -float(r["rms_pos_error_norm"]),
            -float(r["rms_ori_error_norm"]),
            int(r["step"]),
        )

    robust_seeds = max(0, int(args_cli.robust_seeds))
    robust_top_k = max(0, int(args_cli.robust_top_k))
    if robust_seeds > 0 and robust_top_k > 0:
        clean_sorted = sorted(results, key=clean_key, reverse=True)
        robust_candidates = []
        for result in clean_sorted:
            pos_component_rms = float(result["rms_pos_error_norm"]) / float(np.sqrt(3.0))
            clean_ok = result["termination_cause"] == "trunc" and result["frames_tracked"] == result["max_phase"]
            rms_ok = pos_component_rms < 0.10 and result["rms_ori_error_norm"] < 0.25
            if clean_ok and rms_ok:
                robust_candidates.append(result)
            if len(robust_candidates) >= robust_top_k:
                break
        if not robust_candidates:
            robust_candidates = clean_sorted[:robust_top_k]

        for result in robust_candidates:
            actor_critic, normalizer = loaded[result["checkpoint"]]
            passes = 0
            min_frames = int(result["max_phase"])
            causes = {}
            for seed in range(robust_seeds):
                dr_result = rollout(env, actor_critic, normalizer, enable_dr=True, seed=seed)
                good = (
                    dr_result["termination_cause"] == "trunc"
                    and dr_result["frames_tracked"] == dr_result["max_phase"]
                )
                passes += int(good)
                min_frames = min(min_frames, int(dr_result["frames_tracked"]))
                cause = dr_result["termination_cause"]
                causes[cause] = causes.get(cause, 0) + 1
            result.update({
                "robust_passes": passes,
                "robust_trials": robust_seeds,
                "robust_min_frames": min_frames,
                "robust_causes": causes,
            })
            print(
                f"  DR mini {Path(result['path']).name}: "
                f"{passes}/{robust_seeds}, min_frames={min_frames}, causes={causes}"
            )

    def key(r):
        base = clean_key(r)
        robust_passes = int(r.get("robust_passes", -1))
        robust_min_frames = int(r.get("robust_min_frames", -1))
        return (
            base[0],
            base[1],
            robust_passes,
            robust_min_frames,
            base[2],
            base[3],
            -float(r["rms_ori_error_norm"]),
            -float(r["rms_pos_error_norm"]),
            base[6],
        )

    best = max(results, key=key)
    best_dir = run_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    dest = best_dir / "checkpoint.pt"
    if Path(best["checkpoint"]).resolve() != dest.resolve():
        shutil.copy2(best["checkpoint"], dest)
    with (run_dir / "checkpoint_selection.json").open("w") as f:
        json.dump({"selected": best, "candidates": results}, f, indent=2)
    print(
        f"Selected {best['path']} -> best_model "
        f"({best['frames_tracked']}/{best['max_phase']}, cause={best['termination_cause']})"
    )
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
