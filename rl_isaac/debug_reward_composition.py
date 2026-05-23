"""Visualize per-step reward composition for a refine output trajectory."""
# ruff: noqa: I001

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
import types
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Replay a refine torque trajectory and plot reward composition."
)
parser.add_argument("--run-dir", type=str, required=True, help="Refine output run dir.")
parser.add_argument(
    "--torque-path",
    type=str,
    default="",
    help="Explicit torque .npy path. Overrides --iter and auto selection.",
)
parser.add_argument(
    "--iter",
    type=int,
    default=-1,
    help="Specific best_iter_XXXX.npy to load from iter_best_torque.",
)
parser.add_argument(
    "--torque-kind",
    type=str,
    default="auto",
    choices=("auto", "refined", "final-mean", "seed-feedforward"),
    help=(
        "Fallback torque artifact to use when --torque-path/--iter are not set. "
        "'auto' selects the best iteration from mppi_metrics.csv first."
    ),
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="",
    help="Directory for reward_composition.{png,csv}. Defaults to --run-dir.",
)
parser.add_argument(
    "--output-prefix",
    type=str,
    default="reward_composition",
    help="Output filename prefix.",
)
parser.add_argument("--dpi", type=int, default=160, help="Saved plot DPI.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Ensure repo root is on sys.path (must be AFTER AppLauncher which resets sys.path).
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

# GLFW/mujoco stubs - gym_quadruped (via feedforward->model) needs these headless.
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_isaac.env_cfg import Go2TrackingEnvCfg  # noqa: E402
from rl_isaac.rewards import (  # noqa: E402
    ACTION_LIMIT,
    KD,
    KP,
    SIGMA_JOINT,
    SIGMA_ORI,
    SIGMA_POS,
    SIGMA_SMOOTH,
    SIGMA_TORQUE,
    W_JOINT,
    W_ORI,
    W_ORI_INC_MPPI,
    W_POS,
    W_SMOOTH,
    W_TORQUE,
    compute_orientation_increment_reward,
    compute_tracking_errors,
)
from rl_isaac.tracking_env import Go2TrackingEnv  # noqa: E402


@dataclass
class RefPaths:
    state_traj: str
    grf_traj: str
    joint_vel_traj: str
    contact_sequence: str
    traj_dir: str
    iter_num: int


@dataclass
class TorqueSelection:
    path: Path
    label: str
    selected_iter: int | None


TERM_COLUMNS = [
    "r_pos_weighted",
    "r_ori_weighted",
    "r_joint_weighted",
    "r_smooth_weighted",
    "r_torque_weighted",
    "r_ori_inc_weighted",
]


def _resolve_existing_path(path_str: str, *, context: str) -> Path:
    if not path_str:
        raise FileNotFoundError(f"{context} path is empty")
    path = Path(path_str).expanduser()
    if path.exists():
        return path
    repo_path = Path(_repo_root) / path_str
    if repo_path.exists():
        return repo_path
    raise FileNotFoundError(f"{context} not found: {path_str}")


def _load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _paths_from_summary(summary: dict) -> RefPaths:
    paths = summary.get("paths", {})
    state = _resolve_existing_path(paths.get("state_traj", ""), context="state_traj")
    grf = _resolve_existing_path(paths.get("grf_traj", ""), context="grf_traj")
    joint_vel = _resolve_existing_path(
        paths.get("joint_vel_traj", ""), context="joint_vel_traj"
    )
    contact_str = paths.get("contact_sequence", "")
    contact = ""
    if contact_str:
        try:
            contact = str(
                _resolve_existing_path(contact_str, context="contact_sequence")
            )
        except FileNotFoundError:
            contact = ""
    return RefPaths(
        state_traj=str(state),
        grf_traj=str(grf),
        joint_vel_traj=str(joint_vel),
        contact_sequence=contact,
        traj_dir=paths.get("traj_dir", ""),
        iter_num=int(paths.get("iter_num", -1)),
    )


def _best_iteration_from_metrics(run_dir: Path) -> int | None:
    metrics_path = run_dir / "mppi_metrics.csv"
    if not metrics_path.exists():
        return None
    best_iter = None
    best_score = -float("inf")
    with metrics_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                iteration = int(row["iteration"])
                score = float(row["score_best"])
            except (KeyError, TypeError, ValueError):
                continue
            if score > best_score:
                best_score = score
                best_iter = iteration
    return best_iter


def _artifact_path(summary: dict, key: str) -> Path | None:
    artifact = summary.get("artifacts", {}).get(key, "")
    if not artifact:
        return None
    path = Path(artifact).expanduser()
    if path.exists():
        return path
    repo_path = Path(_repo_root) / artifact
    if repo_path.exists():
        return repo_path
    return None


def _select_torque(
    args: argparse.Namespace, run_dir: Path, summary: dict
) -> TorqueSelection:
    if args.torque_path:
        torque_path = _resolve_existing_path(args.torque_path, context="torque_path")
        return TorqueSelection(torque_path, "explicit --torque-path", None)

    if args.iter >= 0:
        torque_path = run_dir / "iter_best_torque" / f"best_iter_{args.iter:04d}.npy"
        if not torque_path.exists():
            raise FileNotFoundError(f"Requested iter torque not found: {torque_path}")
        return TorqueSelection(torque_path, f"iter best {args.iter}", args.iter)

    if args.torque_kind == "auto":
        best_iter = _best_iteration_from_metrics(run_dir)
        if best_iter is not None:
            torque_path = run_dir / "iter_best_torque" / f"best_iter_{best_iter:04d}.npy"
            if torque_path.exists():
                return TorqueSelection(
                    torque_path,
                    f"auto best iteration {best_iter}",
                    best_iter,
                )
            print(
                "Best-iteration torque missing; falling back to refined_torque_traj.npy "
                f"(wanted {torque_path})"
            )
        artifact = _artifact_path(summary, "refined_torque_traj")
        if artifact is not None:
            return TorqueSelection(artifact, "refined_torque_traj fallback", None)
    else:
        artifact_key = {
            "refined": "refined_torque_traj",
            "final-mean": "final_mean_torque_traj",
            "seed-feedforward": "seed_feedforward_torque_traj",
        }[args.torque_kind]
        artifact = _artifact_path(summary, artifact_key)
        if artifact is not None:
            return TorqueSelection(artifact, artifact_key, None)

    fallback_names = (
        "refined_torque_traj.npy",
        "refined_open_loop_torque_traj.npy",
        "final_mean_torque_traj.npy",
        "seed_feedforward_torque_traj.npy",
    )
    for name in fallback_names:
        path = run_dir / name
        if path.exists():
            return TorqueSelection(path, f"{name} fallback", None)

    raise FileNotFoundError(
        "No usable torque trajectory found. Expected --torque-path, "
        "iter_best_torque/best_iter_XXXX.npy, or a saved run-level torque artifact."
    )


def _build_env(paths: RefPaths) -> Go2TrackingEnv:
    cfg = Go2TrackingEnvCfg()
    cfg.scene.num_envs = 1
    cfg.state_traj_path = paths.state_traj
    cfg.grf_traj_path = paths.grf_traj
    cfg.joint_vel_traj_path = paths.joint_vel_traj
    cfg.contact_sequence_path = paths.contact_sequence
    return Go2TrackingEnv(cfg, render_mode=None)


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

    action_scaled = (
        (desired_torque - KD * (ref_jvel - act_jvel) - ff_torque) / KP
        - (ref_jpos - act_jpos)
        - env._joint_offset
    )
    action = torch.clamp(action_scaled / ACTION_LIMIT, -1.0, 1.0)
    return action, action * ACTION_LIMIT


def _reward_terms(
    tracking_errors: dict[str, torch.Tensor], ori_inc_reward: torch.Tensor
) -> dict[str, float]:
    r_pos = torch.exp(-tracking_errors["pos_err_sq"] / (2.0 * SIGMA_POS**2))
    r_ori = torch.exp(-tracking_errors["ori_err_sq"] / (2.0 * SIGMA_ORI**2))
    r_joint = torch.exp(-tracking_errors["joint_err_sq"] / (2.0 * SIGMA_JOINT**2))
    r_smooth = torch.exp(-tracking_errors["rate_sq"] / (2.0 * SIGMA_SMOOTH**2))
    r_torque = torch.exp(
        -(tracking_errors["max_torque"] ** 2) / (2.0 * SIGMA_TORQUE**2)
    )
    terms = {
        "r_pos_weighted": W_POS * r_pos,
        "r_ori_weighted": W_ORI * r_ori,
        "r_joint_weighted": W_JOINT * r_joint,
        "r_smooth_weighted": W_SMOOTH * r_smooth,
        "r_torque_weighted": W_TORQUE * r_torque,
        "r_ori_inc_weighted": W_ORI_INC_MPPI * ori_inc_reward,
    }
    out = {
        key: float(torch.nan_to_num(value, nan=0.0)[0].item())
        for key, value in terms.items()
    }
    out["total"] = sum(out[key] for key in TERM_COLUMNS)
    out["pos_error"] = float(tracking_errors["pos_error"][0].item())
    out["ori_error"] = float(tracking_errors["ori_error"][0].item())
    out["joint_error"] = float(tracking_errors["joint_error"][0].item())
    out["action_rate"] = float(tracking_errors["action_rate"][0].item())
    out["max_torque"] = float(tracking_errors["max_torque"][0].item())
    return out


def _rollout_reward_terms(
    env: Go2TrackingEnv, torque_seq: torch.Tensor
) -> list[dict[str, float | int]]:
    if torque_seq.ndim != 2 or torque_seq.shape[1] != 12:
        raise ValueError(
            f"Expected torque sequence shape (T, 12), got {tuple(torque_seq.shape)}"
        )
    if torque_seq.shape[0] != env._max_phase:
        raise ValueError(
            f"Torque horizon ({torque_seq.shape[0]}) must match reference horizon ({env._max_phase})"
        )

    _reset_env_to_reference_start(env)
    prev_action_scaled = torch.zeros(1, 12, device=env.device)
    rows: list[dict[str, float | int]] = []

    original_reset = env._reset_idx
    env._reset_idx = lambda env_ids: None
    try:
        for step in range(env._max_phase):
            prev_phase = env._phase.clamp(0, env._max_phase - 1).long()
            prev_actual_quat = env._robot.data.root_quat_w.clone()
            actions, action_scaled = _torque_to_action(env, torque_seq[step : step + 1])
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
            ori_inc_reward = compute_orientation_increment_reward(
                env._ref_body_quat[prev_phase],
                env._ref_body_quat[phase],
                prev_actual_quat,
                env._robot.data.root_quat_w,
            )
            row = _reward_terms(tracking_errors, ori_inc_reward)
            row["step"] = step
            row["phase"] = int(phase[0].item())
            row["terminated"] = int(terminated[0].item())
            row["truncated"] = int(truncated[0].item())
            rows.append(row)
            prev_action_scaled = action_scaled
    finally:
        env._reset_idx = original_reset

    return rows


def _write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "phase",
        *TERM_COLUMNS,
        "total",
        "pos_error",
        "ori_error",
        "joint_error",
        "action_rate",
        "max_torque",
        "terminated",
        "truncated",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(
    path: Path, rows: list[dict[str, float | int]], title: str, dpi: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.asarray([row["step"] for row in rows], dtype=np.float32)
    term_arrays = [
        np.asarray([row[col] for row in rows], dtype=np.float32)
        for col in TERM_COLUMNS
    ]
    total = np.asarray([row["total"] for row in rows], dtype=np.float32)
    labels = [
        "position",
        "orientation",
        "joint",
        "smoothness",
        "torque",
        "orientation increment",
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(steps, term_arrays, labels=labels, alpha=0.85)
    ax.plot(steps, total, color="black", linewidth=1.5, label="total")
    ax.set_title(title)
    ax.set_xlabel("timestep")
    ax.set_ylabel("weighted reward")
    ax.set_xlim(
        float(steps[0]),
        float(steps[-1]) if len(steps) > 1 else float(steps[0] + 1),
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _write_ori_increment_plot(
    path: Path, rows: list[dict[str, float | int]], title: str, dpi: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = np.asarray([row["step"] for row in rows], dtype=np.float32)
    ori_inc_weighted = np.asarray(
        [row["r_ori_inc_weighted"] for row in rows], dtype=np.float32
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        steps,
        ori_inc_weighted,
        color="tab:purple",
        linewidth=1.6,
        label="orientation increment",
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("timestep")
    ax.set_ylabel("weighted reward")
    ax.set_xlim(
        float(steps[0]),
        float(steps[-1]) if len(steps) > 1 else float(steps[0] + 1),
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    run_dir = _resolve_existing_path(args.run_dir, context="run_dir")
    summary = _load_summary(run_dir)
    ref_paths = _paths_from_summary(summary)
    torque_selection = _select_torque(args, run_dir, summary)
    torque_np = np.load(torque_selection.path).astype(np.float32)

    env = _build_env(ref_paths)
    try:
        torque = torch.tensor(torque_np, dtype=torch.float32, device=env.device)
        rows = _rollout_reward_terms(env, torque)
    finally:
        env.close()

    for row in rows:
        row_total = sum(float(row[col]) for col in TERM_COLUMNS)
        if abs(row_total - float(row["total"])) > 1e-6:
            raise RuntimeError("Reward composition consistency check failed.")

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else run_dir
    csv_path = output_dir / f"{args.output_prefix}.csv"
    png_path = output_dir / f"{args.output_prefix}.png"
    ori_inc_png_path = output_dir / f"{args.output_prefix}_ori_increment.png"
    title = f"{run_dir.name}: {torque_selection.label}"
    _write_csv(csv_path, rows)
    _write_plot(png_path, rows, title, args.dpi)
    _write_ori_increment_plot(
        ori_inc_png_path,
        rows,
        f"{title} (orientation increment only)",
        args.dpi,
    )

    total_return = sum(float(row["total"]) for row in rows)
    print("============================================================")
    print("Reward composition debug complete")
    print("============================================================")
    print(f"run_dir:        {run_dir}")
    print(f"torque_source:  {torque_selection.label}")
    print(f"torque_path:    {torque_selection.path}")
    print(f"horizon:        {len(rows)}")
    print(f"total_return:   {total_return:.6f}")
    print(f"csv:            {csv_path}")
    print(f"plot:           {png_path}")
    print(f"ori_inc_plot:   {ori_inc_png_path}")
    print("============================================================")


if __name__ == "__main__":
    exit_code = 0
    try:
        main(args_cli)
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        simulation_app.close()
    if exit_code != 0:
        raise SystemExit(exit_code)
