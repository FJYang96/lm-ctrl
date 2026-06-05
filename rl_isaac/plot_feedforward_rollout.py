"""Visualize upsampled MPC reference vs Isaac rollout.

Supports two control modes:

  implicit_pd (default): 1 kHz PhysX implicit PD (Kp/Kd on actuators) tracking
      the 50 Hz reference joints, with optional feedforward torque.

  ff_invert: 50 Hz manual PD with inverted actions to realize open-loop FF torques.

Reference is plotted at control_dt (default 50 Hz). Sim is plotted at the actual
simulation rate (1 kHz when decimation=1).

Usage (inside Isaac Docker):

    python -m rl_isaac.plot_feedforward_rollout \\
        --traj-dir results/llm_iterations/... --iter-num 1

See rl_isaac/run_plot_feedforward_rollout.sh for a wrapper script.
"""
# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import types
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Plot upsampled reference vs feedforward Isaac rollout"
)
parser.add_argument("--traj-dir", type=str, default="")
parser.add_argument("--iter-num", type=int, default=-1)
parser.add_argument("--state-traj", type=str, default="")
parser.add_argument("--grf-traj", type=str, default="")
parser.add_argument("--joint-vel-traj", type=str, default="")
parser.add_argument("--contact-sequence", type=str, default="")
parser.add_argument(
    "--source-dt",
    type=float,
    default=None,
    help="MPC planning timestep in seconds (default: infer from metadata).",
)
parser.add_argument(
    "--control-dt",
    type=float,
    default=0.02,
    help="Control / plot timestep in seconds (50 Hz default).",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="rl_isaac/physics_diagnostics_output",
)
parser.add_argument("--run-tag", type=str, default="")
parser.add_argument("--dpi", type=int, default=150)
parser.add_argument(
    "--zoom-steps",
    type=int,
    default=10,
    help="Number of reference control steps for zoomed-in plots (default: 10).",
)
parser.add_argument(
    "--control-mode",
    type=str,
    default="implicit_pd",
    choices=("implicit_pd", "ff_invert"),
    help="implicit_pd: 1kHz actuator PD + ref targets; ff_invert: 50Hz open-loop FF.",
)
parser.add_argument("--decimation", type=int, default=1, help="Env decimation (1 = 1kHz).")
parser.add_argument("--sim-dt", type=float, default=0.001, help="Physics timestep (s).")
parser.add_argument("--actuator-kp", type=float, default=30.0, help="Implicit actuator Kp.")
parser.add_argument("--actuator-kd", type=float, default=5.0, help="Implicit actuator Kd.")
parser.add_argument(
    "--no-feedforward",
    action="store_true",
    help="Pure PD tracking (implicit_pd mode only); skip FF effort target.",
)
parser.add_argument(
    "--sim-base-z-offset",
    type=float,
    default=0.0,
    help=(
        "Raise sim reference base CoM z (m) for the full rollout (PD targets + init). "
        "Try 0.022 to compensate foot collision sphere radius vs MPC FK frame."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
from scipy.spatial.transform import Rotation  # noqa: E402

from mpc.dynamics.model import KinoDynamic_Model  # noqa: E402
from rl_isaac.env_cfg import Go2TrackingEnvCfg  # noqa: E402
from rl_isaac.feedforward import FeedforwardComputer  # noqa: E402
from rl_isaac.reference import ReferenceTrajectory  # noqa: E402
from rl_isaac.rewards import ACTION_LIMIT, KD, KP  # noqa: E402
from rl_isaac.tracking_env import Go2TrackingEnv  # noqa: E402
from rl_isaac.upsample_reference import (  # noqa: E402
    resolve_source_dt,
    upsample_reference_arrays,
)
from utils.conversion import (  # noqa: E402
    MPC_X_BASE_ANG,
    MPC_X_BASE_EUL,
    MPC_X_BASE_POS,
    MPC_X_BASE_VEL,
    MPC_X_Q_JOINTS,
    quaternion_to_euler,
)

FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
LEG_LABELS = ("FL", "FR", "RL", "RR")
LEG_JOINT_IDX = {
    "FL": (0, 1, 2),
    "FR": (3, 4, 5),
    "RL": (6, 7, 8),
    "RR": (9, 10, 11),
}
JOINT_NAMES = ("hip", "thigh", "calf")
REF_STYLE = {"color": "C0", "linestyle": "--", "linewidth": 1.4, "label": "reference"}
SIM_STYLE = {"color": "C1", "linestyle": "-", "linewidth": 1.4, "label": "sim"}


class Go2ImplicitPDEnv(Go2TrackingEnv):
    """Track reference joints via PhysX implicit PD (+ optional FF effort)."""

    def __init__(
        self,
        cfg,
        *,
        use_feedforward: bool = True,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._use_feedforward = use_feedforward

    def _apply_action(self):
        phase = self._phase.clamp(0, self._max_phase - 1)
        target = self._ref_joint_pos[phase] + self._joint_offset
        self._robot.set_joint_position_target(self._to_isaac_order(target))
        self._robot.set_joint_velocity_target(
            self._to_isaac_order(self._ref_joint_vel[phase])
        )
        if self._use_feedforward:
            torque = self._ref_ff_torques[phase]
        else:
            torque = torch.zeros_like(self._ref_ff_torques[phase])
        self._robot.set_joint_effort_target(self._to_isaac_order(torque))
        self._last_torque = torque.clone()


def _resolve_trajectory_paths(args: argparse.Namespace) -> dict[str, str]:
    if args.traj_dir:
        if args.iter_num < 0:
            raise ValueError("--iter-num must be set when --traj-dir is provided.")
        traj_dir = Path(args.traj_dir)
        paths = {
            "state_traj": str(traj_dir / f"state_traj_iter_{args.iter_num}.npy"),
            "grf_traj": str(traj_dir / f"grf_traj_iter_{args.iter_num}.npy"),
            "joint_vel_traj": str(traj_dir / f"joint_vel_traj_iter_{args.iter_num}.npy"),
            "contact_sequence": str(traj_dir / f"contact_sequence_iter_{args.iter_num}.npy"),
        }
    else:
        if not (args.state_traj and args.grf_traj and args.joint_vel_traj):
            raise ValueError(
                "Provide --traj-dir/--iter-num or explicit trajectory npy paths."
            )
        paths = {
            "state_traj": args.state_traj,
            "grf_traj": args.grf_traj,
            "joint_vel_traj": args.joint_vel_traj,
            "contact_sequence": args.contact_sequence,
        }
    for key in ("state_traj", "grf_traj", "joint_vel_traj"):
        if not Path(paths[key]).exists():
            raise FileNotFoundError(f"Missing trajectory file: {paths[key]}")
    if paths["contact_sequence"] and not Path(paths["contact_sequence"]).exists():
        paths["contact_sequence"] = ""
    return paths


def _build_env(
    state_path: str,
    grf_path: str,
    jvel_path: str,
    contact_path: str,
    control_dt: float,
    *,
    control_mode: str,
    decimation: int,
    sim_dt: float,
    actuator_kp: float,
    actuator_kd: float,
    use_feedforward: bool,
) -> Go2TrackingEnv:
    cfg = Go2TrackingEnvCfg()
    cfg.scene.num_envs = 1
    cfg.state_traj_path = state_path
    cfg.grf_traj_path = grf_path
    cfg.joint_vel_traj_path = jvel_path
    cfg.contact_sequence_path = contact_path
    cfg.control_dt = control_dt
    cfg.decimation = decimation
    cfg.sim.dt = sim_dt
    cfg.sim.render_interval = decimation

    legs = cfg.robot.actuators["legs"]
    if control_mode == "implicit_pd":
        legs.stiffness = actuator_kp
        legs.damping = actuator_kd
        return Go2ImplicitPDEnv(cfg, use_feedforward=use_feedforward)
    legs.stiffness = 0.0
    legs.damping = 0.0
    return Go2TrackingEnv(cfg)


def _sync_sim_state(env: Go2TrackingEnv) -> None:
    """Push written sim state into Isaac articulation buffers."""
    env.scene.write_data_to_sim()
    env.scene.update(0.0)


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
    ref_pos = env._ref_body_pos[0:1].clone() + env._env_origins
    ref_quat = env._ref_body_quat[0:1].clone()
    ref_vel = env._ref_body_vel[0:1].clone()
    ref_ang = env._ref_body_ang_vel[0:1].clone()
    ref_jpos = env._ref_joint_pos[0:1].clone()
    ref_jvel = env._ref_joint_vel[0:1].clone()

    env._robot.write_root_pose_to_sim(torch.cat([ref_pos, ref_quat], dim=-1), all_ids)
    env._robot.write_root_velocity_to_sim(
        torch.cat([ref_vel, ref_ang], dim=-1), all_ids
    )
    env._robot.write_joint_state_to_sim(
        env._to_isaac_order(ref_jpos), env._to_isaac_order(ref_jvel), None, all_ids
    )
    _sync_sim_state(env)


def _torque_to_action(
    env: Go2TrackingEnv, desired_torque: torch.Tensor
) -> torch.Tensor:
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
    return torch.clamp(action_scaled / ACTION_LIMIT, -1.0, 1.0)


def _foot_body_indices(env: Go2TrackingEnv) -> dict[str, int]:
    names = list(env._robot.body_names)
    return {foot: names.index(foot) for foot in FOOT_NAMES}


def _record_sim_state(env: Go2TrackingEnv, foot_body_idx: dict[str, int]) -> dict:
    root_pos = (env._robot.data.root_pos_w[0] - env._env_origins[0]).cpu().numpy()
    root_quat = env._robot.data.root_quat_w[0].cpu().numpy()
    root_euler = quaternion_to_euler(root_quat)
    root_vel = env._robot.data.root_lin_vel_w[0].cpu().numpy()
    root_ang = env._robot.data.root_ang_vel_w[0].cpu().numpy()
    joint_pos = env._to_mpc_order(env._robot.data.joint_pos)[0].cpu().numpy()
    joint_vel = env._to_mpc_order(env._robot.data.joint_vel)[0].cpu().numpy()
    body_pos = env._robot.data.body_pos_w[0].cpu().numpy()
    foot_heights = np.array(
        [body_pos[foot_body_idx[f], 2] - env._env_origins[0, 2].item() for f in FOOT_NAMES],
        dtype=np.float64,
    )
    net_forces = env._contact_sensor.data.net_forces_w[0].cpu().numpy()
    foot_sensor_ids = env._foot_body_ids_t.cpu().numpy()
    grf_fz = np.array([net_forces[i, 2] for i in foot_sensor_ids], dtype=np.float64)
    torque = env._last_torque[0].cpu().numpy()
    return {
        "com_pos": root_pos,
        "com_euler": root_euler,
        "com_vel": root_vel,
        "com_ang_vel": root_ang,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "foot_heights": foot_heights,
        "grf_fz": grf_fz,
        "torque": torque,
    }


def _reference_series(
    state: np.ndarray,
    jvel: np.ndarray,
    grf: np.ndarray,
    dt: float,
) -> dict[str, np.ndarray]:
    model = KinoDynamic_Model()
    fk_funs = [model.kindyn.forward_kinematics_fun(f) for f in FOOT_NAMES]
    n_steps = jvel.shape[0]
    n_states = state.shape[0]

    com_pos = state[:n_steps + 1, MPC_X_BASE_POS].copy()
    com_euler = state[:n_steps + 1, MPC_X_BASE_EUL].copy()
    com_vel = state[:n_steps + 1, MPC_X_BASE_VEL].copy()
    com_ang_vel = state[:n_steps + 1, MPC_X_BASE_ANG].copy()
    joint_pos = state[:n_steps + 1, MPC_X_Q_JOINTS].copy()
    joint_vel = np.zeros((n_steps + 1, 12), dtype=np.float64)
    joint_vel[:n_steps] = jvel
    joint_vel[n_steps] = jvel[-1]

    foot_heights = np.zeros((n_steps + 1, 4), dtype=np.float64)
    for k in range(n_steps + 1):
        euler = state[k, MPC_X_BASE_EUL]
        H = np.eye(4)
        H[:3, :3] = Rotation.from_euler("xyz", euler).as_matrix()
        H[:3, 3] = state[k, MPC_X_BASE_POS]
        q = state[k, MPC_X_Q_JOINTS]
        for fi, fk in enumerate(fk_funs):
            foot_heights[k, fi] = float(np.array(fk(H, q)).reshape(4, 4)[2, 3])

    grf_fz = np.zeros((n_steps + 1, 4), dtype=np.float64)
    for k in range(n_steps):
        for fi in range(4):
            grf_fz[k, fi] = grf[k, fi * 3 + 2]
    grf_fz[n_steps] = grf_fz[n_steps - 1]

    ff = FeedforwardComputer(model)
    ff_torques = np.zeros((n_steps + 1, 12), dtype=np.float64)
    for k in range(n_steps):
        q_ddot = (jvel[k] - jvel[k - 1]) / dt if k > 0 else np.zeros(12)
        ff_torques[k] = ff.compute(
            state[k, MPC_X_BASE_POS],
            state[k, MPC_X_BASE_EUL],
            state[k, MPC_X_BASE_VEL],
            state[k, MPC_X_BASE_ANG],
            state[k, MPC_X_Q_JOINTS],
            jvel[k],
            grf[k],
            q_ddot,
        )
    ff_torques[n_steps] = ff_torques[n_steps - 1]

    return {
        "com_pos": com_pos,
        "com_euler": com_euler,
        "com_vel": com_vel,
        "com_ang_vel": com_ang_vel,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "foot_heights": foot_heights,
        "grf_fz": grf_fz,
        "ff_torque": ff_torques,
    }


def _stack_records(records: list[dict]) -> dict[str, np.ndarray]:
    return {key: np.stack([row[key] for row in records], axis=0) for key in records[0]}


def _run_feedforward_rollout(env: Go2TrackingEnv) -> dict[str, np.ndarray]:
    _reset_env_to_reference_start(env)
    foot_body_idx = _foot_body_indices(env)
    horizon = env._max_phase
    records: list[dict] = [_record_sim_state(env, foot_body_idx)]

    original_reset = env._reset_idx
    env._reset_idx = lambda env_ids: None
    try:
        for _ in range(horizon):
            phase = env._phase.clamp(0, env._max_phase - 1).long()
            desired = env._ref_ff_torques[phase]
            actions = _torque_to_action(env, desired)
            env.step(actions)
            records.append(_record_sim_state(env, foot_body_idx))
    finally:
        env._reset_idx = original_reset

    return _stack_records(records)


def _run_implicit_pd_rollout(
    env: Go2TrackingEnv,
    control_dt: float,
    sim_dt: float,
) -> dict[str, np.ndarray]:
    """1 kHz sim with reference targets held at control_dt (ZOH)."""
    _reset_env_to_reference_start(env)
    foot_body_idx = _foot_body_indices(env)
    steps_per_ref = max(1, int(round(control_dt / sim_dt)))
    total_sim_steps = env._max_phase * steps_per_ref
    zero = torch.zeros(env.num_envs, 12, device=env.device)
    records: list[dict] = [_record_sim_state(env, foot_body_idx)]

    original_reset = env._reset_idx
    env._reset_idx = lambda env_ids: None
    try:
        for k in range(total_sim_steps):
            env._phase[:] = min(k // steps_per_ref, env._max_phase - 1)
            env.step(zero)
            records.append(_record_sim_state(env, foot_body_idx))
    finally:
        env._reset_idx = original_reset

    return _stack_records(records)


def _sim_at_ref_indices(sim: dict[str, np.ndarray], steps_per_ref: int) -> dict[str, np.ndarray]:
    """Sample sim series at reference frame boundaries."""
    n_sim = next(iter(sim.values())).shape[0]
    n_ref = (n_sim - 1) // steps_per_ref + 1
    idx = np.minimum(np.arange(n_ref) * steps_per_ref, n_sim - 1)
    return {k: v[idx] for k, v in sim.items()}


def _time_axis(n: int, dt: float) -> np.ndarray:
    return np.arange(n, dtype=np.float64) * dt


def _plot_ref_sim_pair(
    ax,
    t_ref,
    ref_y,
    t_sim,
    sim_y,
    ylabel: str,
    title: str,
    *,
    show_legend: bool = False,
) -> None:
    ax.plot(t_ref, ref_y, **REF_STYLE)
    ax.plot(t_sim, sim_y, **SIM_STYLE)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(loc="best", fontsize=7)


def _plot_ref_sim_ax(
    ax, t_ref, ref_y, t_sim, sim_y, ylabel: str, title: str
) -> None:
    _plot_ref_sim_pair(ax, t_ref, ref_y, t_sim, sim_y, ylabel, title)


def _plot_base_state(
    path: Path,
    t_ref,
    t_sim,
    ref: dict,
    sim: dict,
    dpi: int,
    subtitle: str = "",
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    labels = ("x", "y", "z")
    for i, lab in enumerate(labels):
        _plot_ref_sim_ax(
            axes[i, 0],
            t_ref,
            ref["com_pos"][:, i],
            t_sim,
            sim["com_pos"][:, i],
            "m",
            f"CoM {lab}",
        )
    rpy = ("roll", "pitch", "yaw")
    for i, lab in enumerate(rpy):
        _plot_ref_sim_ax(
            axes[i, 1],
            t_ref,
            np.rad2deg(ref["com_euler"][:, i]),
            t_sim,
            np.rad2deg(sim["com_euler"][:, i]),
            "deg",
            f"Base {lab}",
        )
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=2)
    suptitle = "Base position and orientation"
    if subtitle:
        suptitle = f"{suptitle} — {subtitle}"
    fig.suptitle(suptitle, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_base_velocity(
    path: Path,
    t_ref,
    t_sim,
    ref: dict,
    sim: dict,
    dpi: int,
    subtitle: str = "",
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for i in range(3):
        axes[0].plot(t_ref, ref["com_vel"][:, i], **REF_STYLE if i == 0 else {})
        axes[0].plot(t_sim, sim["com_vel"][:, i], **SIM_STYLE if i == 0 else {})
    axes[0].set_ylabel("m/s")
    axes[0].set_title("CoM linear velocity")
    for i in range(3):
        axes[1].plot(t_ref, ref["com_ang_vel"][:, i], **REF_STYLE if i == 0 else {})
        axes[1].plot(t_sim, sim["com_ang_vel"][:, i], **SIM_STYLE if i == 0 else {})
    axes[1].set_ylabel("rad/s")
    axes[1].set_title("Base angular velocity (world frame)")
    axes[1].set_xlabel("time (s)")
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=2)
    suptitle = "Base velocity"
    if subtitle:
        suptitle = f"{suptitle} — {subtitle}"
    fig.suptitle(suptitle, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_feet(
    path: Path,
    t_ref,
    t_sim,
    ref: dict,
    sim: dict,
    dpi: int,
    title: str = "",
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True)
    for fi, leg in enumerate(LEG_LABELS):
        _plot_ref_sim_pair(
            axes[0, fi],
            t_ref,
            ref["foot_heights"][:, fi],
            t_sim,
            sim["foot_heights"][:, fi],
            "m",
            f"{leg} height",
            show_legend=(fi == 0),
        )
        _plot_ref_sim_pair(
            axes[1, fi],
            t_ref,
            ref["grf_fz"][:, fi],
            t_sim,
            sim["grf_fz"][:, fi],
            "N",
            f"{leg} GRF Fz",
            show_legend=(fi == 0),
        )
    for ax in axes[1, :]:
        ax.set_xlabel("time (s)")
    suptitle = "Foot height and vertical GRF (per foot)"
    if title:
        suptitle = f"{suptitle} — {title}"
    fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_joints_grid(
    path: Path,
    t_ref,
    t_sim,
    ref_arr: np.ndarray,
    sim_arr: np.ndarray,
    ylabel: str,
    title: str,
    dpi: int,
    subtitle: str = "",
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True)
    for col, leg in enumerate(LEG_LABELS):
        idx = LEG_JOINT_IDX[leg]
        for row, jname in enumerate(JOINT_NAMES):
            ji = idx[row]
            _plot_ref_sim_pair(
                axes[row, col],
                t_ref,
                ref_arr[:, ji],
                t_sim,
                sim_arr[:, ji],
                ylabel if col == 0 else "",
                f"{leg} {jname}",
                show_legend=(row == 0 and col == 0),
            )
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    suptitle = title
    if subtitle:
        suptitle = f"{title} — {subtitle}"
    fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_tracking_errors(
    path: Path,
    t_ref,
    _t_sim_unused,
    ref: dict,
    sim: dict,
    dpi: int,
    subtitle: str = "",
) -> None:
    pos_err = np.linalg.norm(ref["com_pos"] - sim["com_pos"], axis=1)
    ori_err = np.linalg.norm(ref["com_euler"] - sim["com_euler"], axis=1)
    joint_err = np.linalg.norm(ref["joint_pos"] - sim["joint_pos"], axis=1)
    foot_err = np.linalg.norm(ref["foot_heights"] - sim["foot_heights"], axis=1)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t_ref, pos_err, label="CoM position (m)")
    ax.plot(t_ref, ori_err, label="orientation (rad)")
    ax.plot(t_ref, joint_err, label="joint angles (rad)")
    ax.plot(t_ref, foot_err, label="foot heights (m, L2 over feet)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("error magnitude")
    title = "Reference vs sim tracking errors (feedforward rollout)"
    if subtitle:
        title = f"{title} — {subtitle}"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_torque(
    path: Path,
    t_ref,
    t_sim,
    ref_ff: np.ndarray,
    sim_torque: np.ndarray,
    dpi: int,
    subtitle: str = "",
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(t_ref, np.linalg.norm(ref_ff, axis=1), **REF_STYLE)
    axes[0].plot(t_sim, np.linalg.norm(sim_torque, axis=1), **SIM_STYLE)
    axes[0].set_ylabel("Nm")
    axes[0].set_title("Total joint torque L2 norm")
    axes[0].grid(True, alpha=0.25)

    if len(t_ref) == len(t_sim):
        delta = np.linalg.norm(ref_ff - sim_torque, axis=1)
        t_delta = t_ref
    else:
        sim_sampled = sim_torque[
            np.minimum(np.arange(len(t_ref)) * max(1, (len(t_sim) - 1) // max(len(t_ref) - 1, 1)), len(t_sim) - 1)
        ]
        n = min(len(t_ref), len(sim_sampled))
        delta = np.linalg.norm(ref_ff[:n] - sim_sampled[:n], axis=1)
        t_delta = t_ref[:n]
    axes[1].plot(t_delta, delta, color="C3", linewidth=1.4)
    axes[1].set_ylabel("Nm")
    axes[1].set_title("Torque tracking error (planned FF vs applied)")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.25)
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=2)
    suptitle = "Feedforward torque"
    if subtitle:
        suptitle = f"{suptitle} — {subtitle}"
    fig.suptitle(suptitle, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _slice_series(ref: dict, sim: dict, n_ref: int, n_sim: int) -> tuple[dict, dict]:
    return (
        {k: v[:n_ref] for k, v in ref.items()},
        {k: v[:n_sim] for k, v in sim.items()},
    )


def _render_plot_suite(
    out_dir: Path,
    t_ref: np.ndarray,
    t_sim: np.ndarray,
    ref: dict,
    sim: dict,
    dpi: int,
    *,
    steps_per_ref: int = 1,
    suffix: str = "",
    subtitle: str = "",
) -> dict[str, Path]:
    tag = f"_{suffix}" if suffix else ""
    paths = {
        "base_state": out_dir / f"base_state{tag}.png",
        "base_velocity": out_dir / f"base_velocity{tag}.png",
        "feet_grf": out_dir / f"feet_grf{tag}.png",
        "joint_positions": out_dir / f"joint_positions{tag}.png",
        "joint_velocities": out_dir / f"joint_velocities{tag}.png",
        "tracking_errors": out_dir / f"tracking_errors{tag}.png",
        "torque": out_dir / f"torque{tag}.png",
    }
    _plot_base_state(paths["base_state"], t_ref, t_sim, ref, sim, dpi, subtitle)
    _plot_base_velocity(paths["base_velocity"], t_ref, t_sim, ref, sim, dpi, subtitle)
    _plot_feet(paths["feet_grf"], t_ref, t_sim, ref, sim, dpi, subtitle)
    _plot_joints_grid(
        paths["joint_positions"],
        t_ref,
        t_sim,
        ref["joint_pos"],
        sim["joint_pos"],
        "rad",
        "Joint positions q",
        dpi,
        subtitle,
    )
    _plot_joints_grid(
        paths["joint_velocities"],
        t_ref,
        t_sim,
        ref["joint_vel"],
        sim["joint_vel"],
        "rad/s",
        "Joint velocities q̇",
        dpi,
        subtitle,
    )
    sim_at_ref = sim if steps_per_ref <= 1 else _sim_at_ref_indices(sim, steps_per_ref)
    n_err = min(len(t_ref), len(sim_at_ref["com_pos"]))
    _plot_tracking_errors(
        paths["tracking_errors"],
        t_ref[:n_err],
        t_ref[:n_err],
        {k: v[:n_err] for k, v in ref.items()},
        {k: v[:n_err] for k, v in sim_at_ref.items()},
        dpi,
        subtitle,
    )
    _plot_torque(
        paths["torque"], t_ref, t_sim, ref["ff_torque"], sim["torque"], dpi, subtitle
    )
    return paths


def _close_sim(env: Go2TrackingEnv | None) -> None:
    if env is None:
        return
    try:
        env.close()
    except Exception:
        pass


def _save_rollout_npz(
    path: Path,
    t_ref: np.ndarray,
    t_sim: np.ndarray,
    ref: dict,
    sim: dict,
) -> None:
    np.savez(
        path,
        time_ref=t_ref,
        time_sim=t_sim,
        ref_com_pos=ref["com_pos"],
        ref_com_euler=ref["com_euler"],
        ref_joint_pos=ref["joint_pos"],
        ref_joint_vel=ref["joint_vel"],
        ref_foot_heights=ref["foot_heights"],
        ref_grf_fz=ref["grf_fz"],
        ref_ff_torque=ref["ff_torque"],
        sim_com_pos=sim["com_pos"],
        sim_com_euler=sim["com_euler"],
        sim_joint_pos=sim["joint_pos"],
        sim_joint_vel=sim["joint_vel"],
        sim_foot_heights=sim["foot_heights"],
        sim_grf_fz=sim["grf_fz"],
        sim_torque=sim["torque"],
    )


def main() -> None:
    run_tag = args_cli.run_tag or datetime.now().strftime("ff_rollout_%Y%m%d_%H%M%S")
    out_dir = Path(args_cli.output_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = _resolve_trajectory_paths(args_cli)
    source_dt = resolve_source_dt(args_cli.source_dt, args_cli.traj_dir or None)
    control_dt = float(args_cli.control_dt)
    if control_dt <= 0.0:
        raise ValueError("--control-dt must be > 0.")

    control_mode = args_cli.control_mode
    decimation = int(args_cli.decimation)
    sim_dt = float(args_cli.sim_dt)
    use_feedforward = not args_cli.no_feedforward
    base_z_offset = float(args_cli.sim_base_z_offset)
    if decimation < 1:
        raise ValueError("--decimation must be >= 1.")
    if sim_dt <= 0.0:
        raise ValueError("--sim-dt must be > 0.")

    steps_per_ref = max(1, int(round(control_dt / (sim_dt * decimation))))
    sim_label = "sim (PD+FF)" if use_feedforward else "sim (PD)"
    if control_mode == "ff_invert":
        sim_label = "sim (FF)"
    SIM_STYLE["label"] = sim_label

    state_raw = np.load(paths["state_traj"])
    jvel_raw = np.load(paths["joint_vel_traj"])
    grf_raw = np.load(paths["grf_traj"])
    contact_raw = (
        np.load(paths["contact_sequence"]) if paths["contact_sequence"] else None
    )
    state_up, jvel_up, grf_up, contact_up, upsample_meta = upsample_reference_arrays(
        state_raw,
        jvel_raw,
        grf_raw,
        contact_raw,
        source_dt=source_dt,
        target_dt=control_dt,
    )

    upsampled_dir = out_dir / "upsampled_reference"
    upsampled_dir.mkdir(parents=True, exist_ok=True)
    np.save(upsampled_dir / "upsampled_state_traj.npy", state_up)
    np.save(upsampled_dir / "upsampled_joint_vel_traj.npy", jvel_up)
    np.save(upsampled_dir / "upsampled_grf_traj.npy", grf_up)
    if contact_up is not None:
        np.save(upsampled_dir / "upsampled_contact_sequence.npy", contact_up)
    with (upsampled_dir / "upsampled_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(upsample_meta, f, indent=2)

    state_for_env = state_up.copy()
    if base_z_offset != 0.0:
        state_for_env[:, MPC_X_BASE_POS][:, 2] += base_z_offset
        np.save(upsampled_dir / "upsampled_state_traj_sim.npy", state_for_env)

    ref = ReferenceTrajectory(
        state_traj=state_up,
        joint_vel_traj=jvel_up,
        grf_traj=grf_up,
        contact_sequence=contact_up,
        control_dt=control_dt,
    )
    ff_seed = FeedforwardComputer(KinoDynamic_Model()).precompute_trajectory(ref)
    np.save(upsampled_dir / "upsampled_feedforward_torque_traj.npy", ff_seed)

    # Env loads upsampled arrays; sim state traj may include base-z offset.
    if base_z_offset != 0.0:
        state_path = str(upsampled_dir / "upsampled_state_traj_sim.npy")
    else:
        state_path = str(upsampled_dir / "upsampled_state_traj.npy")
    jvel_path = str(upsampled_dir / "upsampled_joint_vel_traj.npy")
    grf_path = str(upsampled_dir / "upsampled_grf_traj.npy")
    contact_path = (
        str(upsampled_dir / "upsampled_contact_sequence.npy")
        if contact_up is not None
        else ""
    )

    print("============================================================")
    print("Rollout visualization")
    print("============================================================")
    print(f"  control_mode: {control_mode}")
    print(f"  source_dt:    {source_dt:.4f} s")
    print(f"  control_dt:   {control_dt:.4f} s (reference rate)")
    print(f"  sim_dt:       {sim_dt:.4f} s, decimation={decimation}")
    print(f"  sim rate:     {1.0 / (sim_dt * decimation):.0f} Hz")
    print(f"  actuator:     Kp={args_cli.actuator_kp}, Kd={args_cli.actuator_kd}")
    print(f"  feedforward:  {use_feedforward}")
    print(f"  sim_base_z_offset: {base_z_offset:.4f} m")
    print(f"  horizon:      {ref.max_phase} ref steps ({ref.duration:.3f} s)")
    print(f"  upsampled:    {upsample_meta.get('upsampled', False)}")
    print(f"  output:       {out_dir}")
    print("============================================================")

    ref_series = _reference_series(state_up, jvel_up, grf_up, control_dt)
    if ff_seed.shape[0] + 1 == ref_series["com_pos"].shape[0]:
        ref_series["ff_torque"] = np.vstack([ff_seed, ff_seed[-1:]])
    else:
        ref_series["ff_torque"] = ff_seed

    if control_mode == "ff_invert" and decimation != 20:
        print("  note: ff_invert mode typically uses decimation=20 (50 Hz).")

    env = _build_env(
        state_path,
        grf_path,
        jvel_path,
        contact_path,
        control_dt,
        control_mode=control_mode,
        decimation=decimation,
        sim_dt=sim_dt,
        actuator_kp=args_cli.actuator_kp,
        actuator_kd=args_cli.actuator_kd,
        use_feedforward=use_feedforward,
    )
    try:
        if control_mode == "implicit_pd":
            sim_series = _run_implicit_pd_rollout(env, control_dt, sim_dt * decimation)
        else:
            sim_series = _run_feedforward_rollout(env)
    finally:
        _close_sim(env)

    n_ref = ref_series["com_pos"].shape[0]
    n_sim = sim_series["com_pos"].shape[0]
    t_ref = _time_axis(n_ref, control_dt)
    t_sim = _time_axis(n_sim, sim_dt * decimation)
    ref_trim = {k: v[:n_ref] for k, v in ref_series.items()}
    sim_trim = {k: v[:n_sim] for k, v in sim_series.items()}

    zoom_steps = max(1, int(args_cli.zoom_steps))
    n_ref_zoom = min(n_ref, zoom_steps + 1)
    n_sim_zoom = min(n_sim, zoom_steps * steps_per_ref + 1)
    t_ref_zoom = t_ref[:n_ref_zoom]
    t_sim_zoom = t_sim[:n_sim_zoom]
    ref_zoom, sim_zoom = _slice_series(ref_trim, sim_trim, n_ref_zoom, n_sim_zoom)
    zoom_label = f"first {zoom_steps} ref steps"

    spr = 1 if control_mode == "ff_invert" else steps_per_ref
    plot_paths = _render_plot_suite(
        out_dir,
        t_ref,
        t_sim,
        ref_trim,
        sim_trim,
        args_cli.dpi,
        steps_per_ref=spr,
        subtitle="full trajectory",
    )
    zoom_paths = _render_plot_suite(
        out_dir,
        t_ref_zoom,
        t_sim_zoom,
        ref_zoom,
        sim_zoom,
        args_cli.dpi,
        steps_per_ref=spr,
        suffix=f"zoom{zoom_steps}",
        subtitle=zoom_label,
    )
    _save_rollout_npz(out_dir / "rollout_data.npz", t_ref, t_sim, ref_trim, sim_trim)

    sim_at_ref = _sim_at_ref_indices(sim_trim, spr) if spr > 1 else sim_trim
    n_err = min(n_ref, len(sim_at_ref["com_pos"]))
    summary = {
        "created_at": datetime.now().isoformat(),
        "control_mode": control_mode,
        "source_dt": source_dt,
        "control_dt": control_dt,
        "sim_dt": sim_dt,
        "decimation": decimation,
        "steps_per_ref": spr,
        "actuator_kp": args_cli.actuator_kp,
        "actuator_kd": args_cli.actuator_kd,
        "use_feedforward": use_feedforward,
        "sim_base_z_offset_m": base_z_offset,
        "upsample_meta": upsample_meta,
        "trajectory_paths": paths,
        "n_ref_timesteps": int(n_ref),
        "n_sim_timesteps": int(n_sim),
        "duration_s": float(t_ref[-1]) if n_ref else 0.0,
        "zoom_steps": zoom_steps,
        "max_com_pos_err_m": float(
            np.linalg.norm(
                ref_trim["com_pos"][:n_err] - sim_at_ref["com_pos"][:n_err], axis=1
            ).max()
        ),
        "max_joint_err_rad": float(
            np.linalg.norm(
                ref_trim["joint_pos"][:n_err] - sim_at_ref["joint_pos"][:n_err], axis=1
            ).max()
        ),
        "max_foot_height_err_m": float(
            np.abs(ref_trim["foot_heights"][:n_err] - sim_at_ref["foot_heights"][:n_err]).max()
        ),
        "plot_paths": {k: str(v) for k, v in plot_paths.items()},
        "zoom_plot_paths": {k: str(v) for k, v in zoom_paths.items()},
        "rollout_npz": str(out_dir / "rollout_data.npz"),
    }
    summary_path = out_dir / "rollout_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPlots saved under {out_dir}")
    print("  Full trajectory:")
    for name, p in plot_paths.items():
        print(f"    {name}: {p.name}")
    print(f"  Zoom ({zoom_label}):")
    for name, p in zoom_paths.items():
        print(f"    {name}: {p.name}")
    print(f"  rollout_data.npz")
    print(f"  rollout_summary.json")
    print(
        f"\nMax errors: CoM {summary['max_com_pos_err_m']:.4f} m, "
        f"joints {summary['max_joint_err_rad']:.4f} rad, "
        f"foot height {summary['max_foot_height_err_m']:.4f} m"
    )


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
    # Isaac/Omniverse shutdown can hang indefinitely in headless mode.
    os._exit(exit_code)
