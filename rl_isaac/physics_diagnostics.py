"""Physics mismatch diagnostics for MPC (URDF/adam) vs Isaac Lab (USD/PhysX).

Runs four checks aligned with the ranked root-cause list:

  A. Model parameters — per-link mass, COM, inertia; total mass; foot FK at
     nominal pose (geometry proxy via foot positions and leg segment lengths).
  B. Contact consistency (reference-side) — GRF/contact schedule alignment,
     stance foot height, tangential foot velocity vs friction cone.
  C. Velocity-frame convention — sensitivity of inverse-dynamics torques to
     treating base angular velocity as body-frame vs world-frame.
  D. Reference self-consistency — finite-difference kinematics and one-step
     forward-dynamics residual against the stored MPC trajectory.

Section A requires Isaac Lab (USD side). B/C/D run offline when trajectory
arrays are provided.

Usage (inside Isaac Docker, with trajectory from traj_config.sh):

    python -m rl_isaac.physics_diagnostics \\
        --traj-dir results/llm_iterations/... --iter-num 1

Model-only (URDF parse still needs go2_config / gym_quadruped for the path):

    python -m rl_isaac.physics_diagnostics --sections A

See rl_isaac/run_physics_diagnostics.sh for a wrapper script.
"""
# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import types
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="MPC vs Isaac physics diagnostics")
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
    help="MPC timestep in seconds (default: infer from metadata / go2_config).",
)
parser.add_argument(
    "--sections",
    type=str,
    default="A,B,C,D",
    help="Comma-separated subset of A,B,C,D to run.",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="rl_isaac/physics_diagnostics_output",
)
parser.add_argument("--run-tag", type=str, default="")
parser.add_argument(
    "--urdf-path",
    type=str,
    default="",
    help="Override URDF path (default: go2_config.robot_data.urdf_filename).",
)
parser.add_argument(
    "--mass-rel-tol",
    type=float,
    default=0.01,
    help="Relative tolerance for per-link mass comparison.",
)
parser.add_argument(
    "--com-abs-tol-m",
    type=float,
    default=0.005,
    help="Absolute tolerance (m) for per-link COM comparison.",
)
parser.add_argument(
    "--foot-pos-abs-tol-m",
    type=float,
    default=0.01,
    help="Absolute tolerance (m) for foot position FK comparison.",
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

import casadi as cs  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from utils.conversion import (  # noqa: E402
    MPC_U_QVEL_JOINTS,
    MPC_X_BASE_ANG,
    MPC_X_BASE_EUL,
    MPC_X_BASE_POS,
    MPC_X_BASE_VEL,
    MPC_X_Q_JOINTS,
    quaternion_to_euler,
    sim_to_mpc,
)

FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
LEG_PREFIXES = ("FL", "FR", "RL", "RR")


@dataclass
class LinkInertial:
    name: str
    mass: float
    com: np.ndarray  # (3,) in link frame
    inertia: np.ndarray  # (3, 3) about COM in link frame


@dataclass
class JointGeometry:
    name: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray


@dataclass
class ModelCompareReport:
    urdf_path: str
    urdf_total_mass: float
    isaac_total_mass: float
    mass_delta: float
    mass_delta_pct: float
    isaac_body_names: list[str] = field(default_factory=list)
    matched_links: list[dict] = field(default_factory=list)
    unmatched_urdf_links: list[str] = field(default_factory=list)
    unmatched_isaac_bodies: list[str] = field(default_factory=list)
    foot_kinematics: dict = field(default_factory=dict)
    foot_position_errors_m: dict[str, float] = field(default_factory=dict)
    max_foot_position_error_m: float | None = None
    leg_segment_lengths: dict[str, dict[str, float]] = field(default_factory=dict)
    geometry_error: str | None = None
    flags: list[str] = field(default_factory=list)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _save_report(out_path: Path, report: dict) -> None:
    def _json_default(obj: object) -> object:
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_json_default)


def _homogeneous_translation(fk_result: object) -> np.ndarray:
    """Extract translation from adam FK output (4x4 or flat length-16)."""
    mat = np.array(fk_result, dtype=np.float64)
    if mat.shape == (4, 4):
        return mat[:3, 3].copy()
    flat = mat.reshape(-1)
    if flat.size == 16:
        return flat.reshape(4, 4)[:3, 3].copy()
    raise ValueError(f"Unexpected FK output shape: {mat.shape}")


def _resolve_isaac_body_index(body_names: list[str], target: str) -> int:
    if target in body_names:
        return body_names.index(target)
    target_norm = _normalize_body_name(target)
    for i, name in enumerate(body_names):
        if _normalize_body_name(name) == target_norm:
            return i
    for i, name in enumerate(body_names):
        if name.endswith(target) or target.endswith(name):
            return i
    raise KeyError(f"Isaac body '{target}' not found in body_names: {body_names}")


def _parse_xyz(text: str) -> np.ndarray:
    vals = [float(v) for v in text.split()]
    return np.array(vals[:3], dtype=np.float64)


def _parse_rpy(text: str) -> np.ndarray:
    vals = [float(v) for v in text.split()]
    return np.array(vals[:3], dtype=np.float64)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    return Rotation.from_euler("xyz", rpy).as_matrix()


def _normalize_body_name(name: str) -> str:
    n = name.strip().lower().replace("-", "_")
    for suffix in ("_link", "_body"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return n


def parse_urdf_model(urdf_path: str) -> tuple[dict[str, LinkInertial], list[JointGeometry]]:
    """Parse link inertial properties and joint geometry from URDF."""
    root = ET.parse(urdf_path).getroot()
    links: dict[str, LinkInertial] = {}
    for link_elem in root.iter("link"):
        name = link_elem.get("name")
        if not name:
            continue
        inertial = link_elem.find("inertial")
        if inertial is None:
            continue
        mass_elem = inertial.find("mass")
        origin_elem = inertial.find("origin")
        inertia_elem = inertial.find("inertia")
        if mass_elem is None or inertia_elem is None:
            continue
        mass = float(mass_elem.get("value", "0"))
        com = _parse_xyz(origin_elem.get("xyz", "0 0 0")) if origin_elem is not None else np.zeros(3)
        inertia = np.array([
            [float(inertia_elem.get("ixx", "0")), float(inertia_elem.get("ixy", "0")), float(inertia_elem.get("ixz", "0"))],
            [float(inertia_elem.get("ixy", "0")), float(inertia_elem.get("iyy", "0")), float(inertia_elem.get("iyz", "0"))],
            [float(inertia_elem.get("ixz", "0")), float(inertia_elem.get("iyz", "0")), float(inertia_elem.get("izz", "0"))],
        ], dtype=np.float64)
        links[name] = LinkInertial(name=name, mass=mass, com=com, inertia=inertia)

    joints: list[JointGeometry] = []
    for joint_elem in root.iter("joint"):
        jname = joint_elem.get("name")
        parent = joint_elem.find("parent")
        child = joint_elem.find("child")
        origin = joint_elem.find("origin")
        axis = joint_elem.find("axis")
        if not jname or parent is None or child is None:
            continue
        joints.append(
            JointGeometry(
                name=jname,
                parent=parent.get("link", ""),
                child=child.get("link", ""),
                origin_xyz=_parse_xyz(origin.get("xyz", "0 0 0")) if origin is not None else np.zeros(3),
                origin_rpy=_parse_rpy(origin.get("rpy", "0 0 0")) if origin is not None else np.zeros(3),
                axis=_parse_xyz(axis.get("xyz", "1 0 0")) if axis is not None else np.array([1.0, 0.0, 0.0]),
            )
        )
    return links, joints


def _resolve_urdf_path(explicit: str) -> str:
    if explicit:
        return explicit
    import go2_config

    return go2_config.robot_data.urdf_filename


def _resolve_trajectory_paths(args: argparse.Namespace) -> dict[str, str | None]:
    if args.traj_dir:
        if args.iter_num < 0:
            raise ValueError("--iter-num is required with --traj-dir")
        traj_dir = Path(args.traj_dir)
        paths = {
            "state_traj": str(traj_dir / f"state_traj_iter_{args.iter_num}.npy"),
            "grf_traj": str(traj_dir / f"grf_traj_iter_{args.iter_num}.npy"),
            "joint_vel_traj": str(traj_dir / f"joint_vel_traj_iter_{args.iter_num}.npy"),
            "contact_sequence": str(traj_dir / f"contact_sequence_iter_{args.iter_num}.npy"),
        }
    else:
        paths = {
            "state_traj": args.state_traj or None,
            "grf_traj": args.grf_traj or None,
            "joint_vel_traj": args.joint_vel_traj or None,
            "contact_sequence": args.contact_sequence or None,
        }
    for key in ("state_traj", "grf_traj", "joint_vel_traj"):
        if not paths[key] or not Path(paths[key]).exists():
            raise FileNotFoundError(f"Required trajectory file missing: {paths.get(key)}")
    if paths["contact_sequence"] and not Path(paths["contact_sequence"]).exists():
        paths["contact_sequence"] = None
    return paths


def _write_dummy_reference(tmpdir: Path, n_steps: int = 2) -> dict[str, str]:
    """Minimal reference quartet so Go2TrackingEnv can spawn for model readout."""
    import go2_config

    mpc_x, mpc_u = sim_to_mpc(
        go2_config.initial_crouch_qpos.copy(),
        go2_config.initial_crouch_qvel.copy(),
    )
    state = np.tile(mpc_x, (n_steps + 1, 1))
    jvel = np.tile(mpc_u[MPC_U_QVEL_JOINTS], (n_steps, 1))
    grf = np.zeros((n_steps, 12), dtype=np.float64)
    per_foot = go2_config.composite_mass * 9.81 / 4.0
    for foot in range(4):
        grf[:, foot * 3 + 2] = per_foot
    contact = np.ones((4, n_steps), dtype=np.float64)

    paths = {
        "state_traj": str(tmpdir / "dummy_state_traj.npy"),
        "grf_traj": str(tmpdir / "dummy_grf_traj.npy"),
        "joint_vel_traj": str(tmpdir / "dummy_joint_vel_traj.npy"),
        "contact_sequence": str(tmpdir / "dummy_contact_sequence.npy"),
    }
    np.save(paths["state_traj"], state)
    np.save(paths["grf_traj"], grf)
    np.save(paths["joint_vel_traj"], jvel)
    np.save(paths["contact_sequence"], contact)
    return paths


def _spawn_isaac_env(ref_paths: dict[str, str]):
    from rl_isaac.env_cfg import Go2TrackingEnvCfg
    from rl_isaac.tracking_env import Go2TrackingEnv

    cfg = Go2TrackingEnvCfg()
    cfg.scene.num_envs = 1
    cfg.state_traj_path = ref_paths["state_traj"]
    cfg.grf_traj_path = ref_paths["grf_traj"]
    cfg.joint_vel_traj_path = ref_paths["joint_vel_traj"]
    cfg.contact_sequence_path = ref_paths.get("contact_sequence") or ""
    import go2_config

    cfg.control_dt = go2_config.default_ref_control_dt
    env = Go2TrackingEnv(cfg)
    env.reset()
    return env


def _set_env_to_crouch(env, settle_steps: int = 0) -> None:
    """Write nominal crouch pose to sim. No physics settling by default."""
    import go2_config

    all_ids = env._robot._ALL_INDICES
    qpos = go2_config.initial_crouch_qpos.copy()
    qvel = go2_config.initial_crouch_qvel.copy()
    root_pos = (
        torch.tensor(qpos[:3], device=env.device, dtype=torch.float32).unsqueeze(0)
        + env._env_origins[:1]
    )
    root_quat = torch.tensor(qpos[3:7], device=env.device, dtype=torch.float32).unsqueeze(0)
    root_vel = torch.zeros(1, 6, device=env.device, dtype=torch.float32)
    jpos_mpc = torch.tensor(qpos[7:19], device=env.device, dtype=torch.float32).unsqueeze(0)
    jvel_mpc = torch.zeros(1, 12, device=env.device, dtype=torch.float32)
    env._robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), all_ids)
    env._robot.write_root_velocity_to_sim(root_vel, all_ids)
    env._robot.write_joint_state_to_sim(
        env._to_isaac_order(jpos_mpc),
        env._to_isaac_order(jvel_mpc),
        None,
        all_ids,
    )
    _refresh_isaac_kinematics(env, settle_steps=settle_steps)


def _refresh_isaac_kinematics(env, settle_steps: int = 0) -> None:
    """Update Isaac body/joint buffers after writing state."""
    env.scene.write_data_to_sim()
    if settle_steps > 0:
        for _ in range(settle_steps):
            env.sim.step()
            env.scene.update(env.physics_dt)
    else:
        env.scene.update(0.0)


def _parse_isaac_com_pose(com_buffer: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Extract link-frame COM position (and optional quat) from PhysX get_coms().

    Isaac returns shape (num_bodies, 7): [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w].
    URDF inertial origins are position-only, so we compare against the first three components.
    """
    if com_buffer.ndim == 1:
        com_buffer = com_buffer.reshape(1, -1)
    if com_buffer.shape[-1] == 7:
        com_pos = com_buffer[:, :3]
        com_quat_xyzw = com_buffer[:, 3:7]
        return com_pos, com_quat_xyzw
    if com_buffer.shape[-1] == 3:
        return com_buffer, None
    raise ValueError(
        f"Unexpected COM buffer shape {com_buffer.shape}; expected (*, 7) or (*, 3)"
    )


def _read_isaac_inertial(
    env,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    masses = env._robot.root_physx_view.get_masses()[0].detach().cpu().numpy()
    com_pose_raw = env._robot.root_physx_view.get_coms()[0].detach().cpu().numpy()
    coms, com_quats = _parse_isaac_com_pose(com_pose_raw)
    inertia_raw = env._robot.root_physx_view.get_inertias()[0].detach().cpu().numpy()
    n_bodies = len(env._robot.body_names)
    if inertia_raw.size == n_bodies * 9:
        inertias = inertia_raw.reshape(n_bodies, 3, 3)
    elif inertia_raw.ndim == 3 and inertia_raw.shape[-2:] == (3, 3):
        inertias = inertia_raw
    else:
        raise ValueError(
            f"Unexpected inertia buffer shape {inertia_raw.shape} "
            f"for {n_bodies} bodies"
        )
    return list(env._robot.body_names), masses, coms, inertias, com_quats


def _match_links(
    urdf_links: dict[str, LinkInertial],
    isaac_names: list[str],
) -> tuple[dict[str, tuple[str, str]], list[str], list[str]]:
    urdf_norm = {_normalize_body_name(k): k for k in urdf_links}
    isaac_norm = {_normalize_body_name(n): n for n in isaac_names}
    matched: dict[str, tuple[str, str]] = {}
    for norm, urdf_name in urdf_norm.items():
        if norm in isaac_norm:
            matched[urdf_name] = (urdf_name, isaac_norm[norm])
    unmatched_urdf = sorted(set(urdf_links) - {u for u, _ in matched.values()})
    unmatched_isaac = sorted(set(isaac_names) - {i for _, i in matched.values()})
    return matched, unmatched_urdf, unmatched_isaac


def _read_isaac_configuration(env) -> dict[str, object]:
    """Read the actual root pose and MPC-ordered joint angles from Isaac."""
    origin = env._env_origins[0].detach().cpu().numpy()
    root_pos = env._robot.data.root_pos_w[0].detach().cpu().numpy() - origin
    root_quat_wxyz = env._robot.data.root_quat_w[0].detach().cpu().numpy()
    root_euler = quaternion_to_euler(root_quat_wxyz)
    q_isaac = env._robot.data.joint_pos[0].detach().cpu().numpy()
    q_mpc = env._to_mpc_order(env._robot.data.joint_pos)[0].detach().cpu().numpy()
    return {
        "root_pos_w": root_pos,
        "root_euler_xyz": root_euler,
        "joint_pos_mpc": q_mpc,
        "joint_pos_isaac": q_isaac,
        "isaac_joint_names": list(env._robot.joint_names),
        "joint_reorder_needed": env._joint_reorder is not None,
    }


def _adam_foot_positions(
    model,
    base_pos: np.ndarray,
    base_euler: np.ndarray,
    q_joints: np.ndarray,
) -> dict[str, np.ndarray]:
    """Foot sphere centers in world frame (same convention as MPC constraints)."""
    w_R_b = Rotation.from_euler("xyz", base_euler).as_matrix()
    H = np.eye(4)
    H[:3, :3] = w_R_b
    H[:3, 3] = base_pos
    center_funs = [
        model.foot_center_position_fl_fun,
        model.foot_center_position_fr_fun,
        model.foot_center_position_rl_fun,
        model.foot_center_position_rr_fun,
    ]
    return {
        foot: np.array(center_fun(H, q_joints)).flatten()
        for foot, center_fun in zip(FOOT_NAMES, center_funs)
    }


def _isaac_foot_sphere_centers(
    env,
    model,
    base_pos: np.ndarray,
    base_euler: np.ndarray,
    q_joints: np.ndarray,
) -> dict[str, np.ndarray]:
    """Isaac foot body origin + URDF collision offset rotated by adam FK."""
    import go2_config

    w_R_b = Rotation.from_euler("xyz", base_euler).as_matrix()
    H = np.eye(4)
    H[:3, :3] = w_R_b
    H[:3, 3] = base_pos
    offset = go2_config.foot_sphere_center_offset
    body_names = list(env._robot.body_names)
    origin = env._env_origins[0].detach().cpu().numpy()
    out: dict[str, np.ndarray] = {}
    for foot in FOOT_NAMES:
        idx = _resolve_isaac_body_index(body_names, foot)
        p_foot = env._robot.data.body_pos_w[0, idx].detach().cpu().numpy() - origin
        fk = model.kindyn.forward_kinematics_fun(foot)
        H_foot = np.array(fk(H, q_joints)).reshape(4, 4)
        R_foot = H_foot[:3, :3]
        out[foot] = p_foot + R_foot @ offset
    return out


def _foot_errors(adam_feet: dict[str, np.ndarray], isaac_feet: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        foot: float(np.linalg.norm(adam_feet[foot] - isaac_feet[foot]))
        for foot in FOOT_NAMES
    }


def _foot_errors_base_relative(
    adam_feet: dict[str, np.ndarray],
    isaac_feet: dict[str, np.ndarray],
    adam_base: np.ndarray,
    isaac_base: np.ndarray,
) -> dict[str, float]:
    return {
        foot: float(
            np.linalg.norm((adam_feet[foot] - adam_base) - (isaac_feet[foot] - isaac_base))
        )
        for foot in FOOT_NAMES
    }


def _isaac_foot_positions(env) -> dict[str, np.ndarray]:
    body_names = list(env._robot.body_names)
    origin = env._env_origins[0].detach().cpu().numpy()
    out: dict[str, np.ndarray] = {}
    for foot in FOOT_NAMES:
        idx = _resolve_isaac_body_index(body_names, foot)
        pos_w = env._robot.data.body_pos_w[0, idx].detach().cpu().numpy() - origin
        out[foot] = pos_w
    return out


def _isaac_body_position(env, body_name: str) -> np.ndarray:
    body_names = list(env._robot.body_names)
    origin = env._env_origins[0].detach().cpu().numpy()
    idx = _resolve_isaac_body_index(body_names, body_name)
    return env._robot.data.body_pos_w[0, idx].detach().cpu().numpy() - origin


def _isaac_base_position(env) -> np.ndarray:
    origin = env._env_origins[0].detach().cpu().numpy()
    return env._robot.data.root_pos_w[0].detach().cpu().numpy() - origin


def _urdf_leg_segment_lengths(joints: list[JointGeometry]) -> dict[str, dict[str, float]]:
    """Fixed URDF joint-origin segment lengths (independent of pose)."""
    by_child = {j.child: j for j in joints}
    out: dict[str, dict[str, float]] = {}
    for prefix in LEG_PREFIXES:
        chain = [
            ("hip_offset", "base", f"{prefix}_hip"),
            ("thigh", f"{prefix}_hip", f"{prefix}_thigh"),
            ("calf", f"{prefix}_thigh", f"{prefix}_calf"),
            ("foot", f"{prefix}_calf", f"{prefix}_foot"),
        ]
        segments: dict[str, float] = {}
        for seg_name, _parent, child in chain:
            if child not in by_child:
                continue
            segments[seg_name] = float(np.linalg.norm(by_child[child].origin_xyz))
        out[f"{prefix}_URDF_static"] = segments
    return out


def _isaac_leg_segment_lengths(env) -> dict[str, dict[str, float]]:
    """Runtime inter-body distances in Isaac at the current pose."""
    out: dict[str, dict[str, float]] = {}
    for prefix in LEG_PREFIXES:
        chain = [
            ("hip_offset", "base", f"{prefix}_hip"),
            ("thigh", f"{prefix}_hip", f"{prefix}_thigh"),
            ("calf", f"{prefix}_thigh", f"{prefix}_calf"),
            ("foot", f"{prefix}_calf", f"{prefix}_foot"),
        ]
        segments: dict[str, float] = {}
        for seg_name, parent, child in chain:
            try:
                segments[seg_name] = float(
                    np.linalg.norm(
                        _isaac_body_position(env, child) - _isaac_body_position(env, parent)
                    )
                )
            except KeyError:
                continue
        out[f"{prefix}_ISAAC_runtime"] = segments
    return out


def _compare_foot_kinematics(env, urdf_joints: list[JointGeometry]) -> dict:
    from mpc.dynamics.model import KinoDynamic_Model
    import go2_config

    model = KinoDynamic_Model()
    cfg = _read_isaac_configuration(env)
    q_nominal = go2_config.initial_crouch_qpos[7:19].astype(np.float64)
    q_isaac = np.asarray(cfg["joint_pos_mpc"], dtype=np.float64)
    isaac_base = _isaac_base_position(env)
    nominal_base = go2_config.initial_crouch_qpos[:3].copy()
    nominal_euler = np.zeros(3)
    readback_base = np.asarray(cfg["root_pos_w"], dtype=np.float64)
    readback_euler = np.asarray(cfg["root_euler_xyz"], dtype=np.float64)
    isaac_feet_nominal = _isaac_foot_sphere_centers(
        env, model, nominal_base, nominal_euler, q_nominal
    )
    isaac_feet_readback = _isaac_foot_sphere_centers(
        env, model, readback_base, readback_euler, q_isaac
    )

    adam_nominal = _adam_foot_positions(model, nominal_base, nominal_euler, q_nominal)
    err_nominal_world = _foot_errors(adam_nominal, isaac_feet_nominal)

    adam_readback = _adam_foot_positions(model, readback_base, readback_euler, q_isaac)
    err_readback_world = _foot_errors(adam_readback, isaac_feet_readback)
    adam_base = readback_base
    err_readback_base = _foot_errors_base_relative(
        adam_readback, isaac_feet, adam_base, isaac_base
    )

    joint_delta = q_isaac - q_nominal
    return {
        "notes": {
            "adam_fk_convention": (
                "adam FK uses world-frame base position + xyz euler + MPC joint order, "
                "matching mpc/dynamics/model.py foot sphere-center constraints."
            ),
            "isaac_foot_frame": (
                "body_pos_w of *_foot bodies + URDF collision sphere offset rotated by adam FK"
            ),
            "leg_segment_urdf": "fixed |joint origin xyz| from URDF (pose-independent)",
            "leg_segment_isaac": "runtime 3D distance between body origins at current pose",
        },
        "joint_order": {
            "isaac_joint_names": cfg["isaac_joint_names"],
            "joint_reorder_needed": cfg["joint_reorder_needed"],
        },
        "joint_delta_nominal_vs_isaac": {
            "max_abs_rad": float(np.max(np.abs(joint_delta))),
            "per_joint": joint_delta.tolist(),
        },
        "foot_error_world_m": {
            "nominal_hardcoded_base": err_nominal_world,
            "isaac_readback_state": err_readback_world,
        },
        "foot_error_base_relative_m": {
            "nominal_hardcoded_base": _foot_errors_base_relative(
                adam_nominal, isaac_feet_nominal, nominal_base, isaac_base
            ),
            "isaac_readback_state": err_readback_base,
        },
        "max_foot_error_world_m": {
            "nominal_hardcoded_base": max(err_nominal_world.values()),
            "isaac_readback_state": max(err_readback_world.values()),
        },
        "leg_segment_lengths": {
            **_urdf_leg_segment_lengths(urdf_joints),
            **_isaac_leg_segment_lengths(env),
        },
    }


def run_model_compare(
    urdf_path: str,
    mass_rel_tol: float,
    com_abs_tol_m: float,
    foot_pos_abs_tol_m: float,
) -> ModelCompareReport:
    urdf_links, urdf_joints = parse_urdf_model(urdf_path)
    urdf_total = float(sum(link.mass for link in urdf_links.values()))

    import tempfile

    with tempfile.TemporaryDirectory(prefix="physics_diag_") as tmp:
        ref_paths = _write_dummy_reference(Path(tmp))
        _log("[A] Spawning Isaac env for model readout...")
        env = _spawn_isaac_env(ref_paths)
        foot_errs: dict[str, float] = {}
        foot_kinematics: dict = {}
        leg_segments: dict[str, dict[str, float]] = {}
        geometry_error: str | None = None
        try:
            _log("[A] Setting nominal crouch pose...")
            _set_env_to_crouch(env)
            _log("[A] Reading Isaac inertial properties...")
            isaac_names, masses, coms, inertias, _com_quats = _read_isaac_inertial(env)
            isaac_total = float(masses.sum())
            matched, unmatched_urdf, unmatched_isaac = _match_links(urdf_links, isaac_names)

            rows: list[dict] = []
            for urdf_name, (_, isaac_name) in sorted(matched.items(), key=lambda x: x[0]):
                u = urdf_links[urdf_name]
                idx = isaac_names.index(isaac_name)
                m_isaac = float(masses[idx])
                com_isaac = coms[idx]
                inertia_isaac = inertias[idx]
                dm = m_isaac - u.mass
                dm_rel = abs(dm) / max(u.mass, 1e-9)
                dcom = com_isaac - u.com
                dinertia = inertia_isaac - u.inertia
                rows.append({
                    "urdf_link": urdf_name,
                    "isaac_body": isaac_name,
                    "mass_urdf": u.mass,
                    "mass_isaac": m_isaac,
                    "mass_delta": dm,
                    "mass_delta_pct": 100.0 * dm_rel,
                    "com_urdf": u.com.tolist(),
                    "com_isaac": com_isaac.tolist(),
                    "com_delta_m": dcom.tolist(),
                    "com_delta_norm_m": float(np.linalg.norm(dcom)),
                    "inertia_delta_fro": float(np.linalg.norm(dinertia)),
                })

            _log("[A] Comparing foot FK / leg geometry (adam URDF vs Isaac bodies)...")
            try:
                foot_kinematics = _compare_foot_kinematics(env, urdf_joints)
                foot_errs = foot_kinematics["foot_error_world_m"]["isaac_readback_state"]
                leg_segments = foot_kinematics["leg_segment_lengths"]
            except Exception as exc:
                geometry_error = f"{type(exc).__name__}: {exc}"
                foot_kinematics = {}
                _log(f"[A] Geometry comparison failed: {geometry_error}")

            report = ModelCompareReport(
                urdf_path=urdf_path,
                urdf_total_mass=urdf_total,
                isaac_total_mass=isaac_total,
                mass_delta=isaac_total - urdf_total,
                mass_delta_pct=100.0 * abs(isaac_total - urdf_total) / max(urdf_total, 1e-9),
                isaac_body_names=isaac_names,
                matched_links=rows,
                unmatched_urdf_links=unmatched_urdf,
                unmatched_isaac_bodies=unmatched_isaac,
                foot_kinematics=foot_kinematics,
                foot_position_errors_m=foot_errs,
                max_foot_position_error_m=max(foot_errs.values()) if foot_errs else None,
                leg_segment_lengths=leg_segments,
                geometry_error=geometry_error,
            )
        finally:
            env.close()

    if report.mass_delta_pct > 100.0 * mass_rel_tol:
        report.flags.append(
            f"total mass differs by {report.mass_delta_pct:.2f}% "
            f"(URDF {report.urdf_total_mass:.4f} kg vs Isaac {report.isaac_total_mass:.4f} kg)"
        )
    for row in report.matched_links:
        if row["mass_delta_pct"] > 100.0 * mass_rel_tol:
            report.flags.append(
                f"mass mismatch {row['urdf_link']}: "
                f"{row['mass_urdf']:.4f} vs {row['mass_isaac']:.4f} kg "
                f"({row['mass_delta_pct']:.1f}%)"
            )
        if row["com_delta_norm_m"] > com_abs_tol_m:
            report.flags.append(
                f"COM mismatch {row['urdf_link']}: |Δ|={row['com_delta_norm_m']*1000:.1f} mm"
            )
    if report.max_foot_position_error_m is not None and report.max_foot_position_error_m > foot_pos_abs_tol_m:
        report.flags.append(
            f"foot FK mismatch (isaac readback): max error "
            f"{report.max_foot_position_error_m*1000:.1f} mm "
            f"(>{foot_pos_abs_tol_m*1000:.1f} mm tol)"
        )
    fk = report.foot_kinematics
    if fk:
        legacy_max = fk.get("max_foot_error_world_m", {}).get("nominal_hardcoded_base")
        readback_max = fk.get("max_foot_error_world_m", {}).get("isaac_readback_state")
        if legacy_max is not None and readback_max is not None and legacy_max > 10 * max(readback_max, 1e-6):
            report.flags.append(
                "large foot error under legacy hardcoded FK but small under Isaac readback "
                "— earlier mismatch was likely a script convention bug, not model geometry"
            )
        j_delta = fk.get("joint_delta_nominal_vs_isaac", {}).get("max_abs_rad", 0.0)
        if j_delta > 1e-3:
            report.flags.append(
                f"Isaac joint angles differ from nominal crouch by up to {j_delta:.4f} rad"
            )
    if report.geometry_error:
        report.flags.append(f"geometry comparison failed: {report.geometry_error}")
    return report


def run_reference_contact_checks(
    state: np.ndarray,
    grf: np.ndarray,
    jvel: np.ndarray,
    contact: np.ndarray | None,
    dt: float,
) -> dict:
    from mpc.dynamics.model import KinoDynamic_Model

    model = KinoDynamic_Model()
    N = grf.shape[0]
    import go2_config

    R = float(go2_config.foot_sphere_radius)
    fk_funs = [
        model.foot_center_position_fl_fun,
        model.foot_center_position_fr_fun,
        model.foot_center_position_rl_fun,
        model.foot_center_position_rr_fun,
    ]
    jac_funs = [
        model.foot_center_jacobian_fl_fun,
        model.foot_center_jacobian_fr_fun,
        model.foot_center_jacobian_rl_fun,
        model.foot_center_jacobian_rr_fun,
    ]

    stance_grf_without_contact = 0
    flight_grf_with_contact = 0
    low_stance_feet = 0
    mdp_violations = 0
    foot_heights_stance: list[float] = []

    for k in range(N):
        euler = state[k, MPC_X_BASE_EUL]
        H = np.eye(4)
        H[:3, :3] = Rotation.from_euler("xyz", euler).as_matrix()
        H[:3, 3] = state[k, MPC_X_BASE_POS]
        q = state[k, MPC_X_Q_JOINTS]
        v_gen = np.concatenate([state[k, MPC_X_BASE_VEL], state[k, MPC_X_BASE_ANG], jvel[k]])

        for fi in range(4):
            contact_on = True if contact is None else contact[fi, k] > 0.5
            fz = grf[k, fi * 3 + 2]
            f_tang = grf[k, fi * 3: fi * 3 + 2]
            foot_height = float(np.array(fk_funs[fi](H, q)).flatten()[2])

            if contact_on:
                foot_heights_stance.append(foot_height)
                if foot_height < R - 0.005:
                    low_stance_feet += 1
                if fz < 1.0:
                    stance_grf_without_contact += 1
            elif fz > 1.0:
                flight_grf_with_contact += 1

            if contact_on and fz > 5.0:
                J = np.array(jac_funs[fi](H, q)).reshape(3, -1)
                v_foot = J @ v_gen
                v_tang = v_foot[:2]
                if np.linalg.norm(v_tang) > 0.01 and float(np.dot(f_tang, v_tang)) > 0.0:
                    mdp_violations += 1

    return {
        "n_steps": N,
        "stance_grf_without_contact": stance_grf_without_contact,
        "flight_grf_with_contact": flight_grf_with_contact,
        "stance_feet_below_minus_5mm": low_stance_feet,
        "mdp_violations_f_dot_v_pos": mdp_violations,
        "stance_foot_height_mean_m": float(np.mean(foot_heights_stance)) if foot_heights_stance else None,
        "stance_foot_height_max_m": float(np.max(foot_heights_stance)) if foot_heights_stance else None,
        "stance_foot_height_min_m": float(np.min(foot_heights_stance)) if foot_heights_stance else None,
        "flags": [
            *(["GRF present during scheduled flight"] if flight_grf_with_contact else []),
            *(["missing GRF during scheduled stance"] if stance_grf_without_contact else []),
            *(["stance sphere centers below R-5mm in reference FK"] if low_stance_feet else []),
            *(["MDP violations (f_t·v_t > 0) in reference"] if mdp_violations else []),
        ],
    }


def run_velocity_frame_check(
    state: np.ndarray,
    jvel: np.ndarray,
    grf: np.ndarray,
    dt: float,
    sample_stride: int = 1,
) -> dict:
    from mpc.dynamics.model import KinoDynamic_Model
    from rl_isaac.feedforward import FeedforwardComputer

    ff = FeedforwardComputer(KinoDynamic_Model())
    N = jvel.shape[0]
    rel_errors: list[float] = []
    max_abs_delta = 0.0
    worst_step = -1

    for k in range(0, N, sample_stride):
        if k > 0:
            q_ddot = (jvel[k] - jvel[k - 1]) / dt
        else:
            q_ddot = np.zeros(12)
        base_pos = state[k, MPC_X_BASE_POS]
        base_rpy = state[k, MPC_X_BASE_EUL]
        base_lin_vel = state[k, MPC_X_BASE_VEL]
        omega_world = state[k, MPC_X_BASE_ANG]
        R = Rotation.from_euler("xyz", base_rpy).as_matrix()
        omega_body = R.T @ omega_world
        joint_pos = state[k, MPC_X_Q_JOINTS]
        joint_vel = jvel[k]

        common = (base_pos, base_rpy, base_lin_vel, joint_pos, joint_vel, grf[k], q_ddot)
        tau_world = ff.compute(*common[:3], omega_world, *common[3:])
        tau_body = ff.compute(*common[:3], omega_body, *common[3:])
        delta = np.abs(tau_world - tau_body)
        max_abs_delta = max(max_abs_delta, float(delta.max()))
        denom = np.maximum(np.abs(tau_world), 1.0)
        rel_errors.append(float((delta / denom).max()))
        if float(delta.max()) >= max_abs_delta:
            worst_step = k

    rel_arr = np.array(rel_errors) if rel_errors else np.zeros(0)
    return {
        "n_samples": int(len(rel_errors)),
        "max_abs_torque_delta_nm": max_abs_delta,
        "max_rel_error": float(rel_arr.max()) if rel_arr.size else 0.0,
        "mean_rel_error": float(rel_arr.mean()) if rel_arr.size else 0.0,
        "worst_step": worst_step,
        "interpretation": (
            "Angular velocity is likely world-frame (MIXED rep): body-frame "
            "reinterpretation barely changes ID torques."
            if (rel_arr.size and rel_arr.max() < 0.05)
            else "Large ID sensitivity to omega frame — verify MPC state stores "
            "world-frame angular velocity as expected by adam MIXED representation."
        ),
        "flags": [
            "velocity-frame sensitivity > 5% on inverse dynamics"
        ] if (rel_arr.size and rel_arr.max() > 0.05) else [],
    }


def _build_forward_dynamics_fun():
    from mpc.dynamics.model import KinoDynamic_Model

    model = KinoDynamic_Model()
    param = cs.vertcat(model.stance_param, model.q_ddot_j)
    f_expl = model.forward_dynamics(model.states, model.inputs, param)
    return cs.Function(
        "forward_dynamics",
        [model.states, model.inputs, param],
        [f_expl],
    )


def run_reference_consistency(
    state: np.ndarray,
    jvel: np.ndarray,
    grf: np.ndarray,
    contact: np.ndarray | None,
    dt: float,
) -> dict:
    fd_fun = _build_forward_dynamics_fun()
    N = jvel.shape[0]

    pos_fd_err = []
    vel_fd_err = []
    joint_vel_err = []
    dyn_residual_norm = []

    for k in range(N):
        qdot_from_jvel = jvel[k]
        qdot_from_fd = (state[k + 1, MPC_X_Q_JOINTS] - state[k, MPC_X_Q_JOINTS]) / dt
        joint_vel_err.append(float(np.linalg.norm(qdot_from_jvel - qdot_from_fd)))

        com_vel_fd = (state[k + 1, MPC_X_BASE_POS] - state[k, MPC_X_BASE_POS]) / dt
        vel_fd_err.append(float(np.linalg.norm(state[k, MPC_X_BASE_VEL] - com_vel_fd)))

        if k > 0:
            q_ddot = (jvel[k] - jvel[k - 1]) / dt
        else:
            q_ddot = np.zeros(12)
        stance = np.ones(4) if contact is None else contact[:, k]
        p = np.concatenate([stance, q_ddot])
        u = np.concatenate([jvel[k], grf[k]])
        xdot = np.array(fd_fun(state[k], u, p)).flatten()
        x_pred = state[k] + dt * xdot
        # Compare primary mechanical states (exclude integral slots).
        residual = state[k + 1, :24] - x_pred[:24]
        dyn_residual_norm.append(float(np.linalg.norm(residual)))

    jve = np.array(joint_vel_err)
    dve = np.array(vel_fd_err)
    dyn = np.array(dyn_residual_norm)
    flags = []
    if jve.max() > 0.5:
        flags.append(f"joint_vel inconsistent with diff(joint_pos): max err {jve.max():.3f} rad/s")
    if dve.max() > 0.05:
        flags.append(f"com_vel inconsistent with diff(com_pos): max err {dve.max():.3f} m/s")
    if dyn.max() > 0.1:
        flags.append(f"forward-dynamics one-step residual: max {dyn.max():.3f} (state units)")

    return {
        "source_dt": dt,
        "n_steps": N,
        "joint_vel_vs_diff_q_max": float(jve.max()),
        "com_vel_vs_diff_pos_max": float(dve.max()),
        "forward_dynamics_residual_max": float(dyn.max()),
        "forward_dynamics_residual_mean": float(dyn.mean()),
        "forward_dynamics_residual_rms": float(np.sqrt(np.mean(dyn ** 2))),
        "flags": flags,
    }


def _print_model_compare(report: ModelCompareReport) -> None:
    print("\n" + "=" * 72)
    print("[A] URDF vs Isaac model parameters")
    print("=" * 72)
    print(f"URDF path:     {report.urdf_path}")
    print(
        f"Total mass:    URDF {report.urdf_total_mass:.4f} kg | "
        f"Isaac {report.isaac_total_mass:.4f} kg | "
        f"Δ {report.mass_delta:+.4f} kg ({report.mass_delta_pct:.2f}%)"
    )
    print(f"Matched links: {len(report.matched_links)}")
    if report.isaac_body_names:
        preview = ", ".join(report.isaac_body_names[:12])
        suffix = "..." if len(report.isaac_body_names) > 12 else ""
        print(f"Isaac bodies:  {preview}{suffix}")
    if report.unmatched_urdf_links:
        print(f"Unmatched URDF links ({len(report.unmatched_urdf_links)}): "
              f"{', '.join(report.unmatched_urdf_links[:8])}"
              f"{'...' if len(report.unmatched_urdf_links) > 8 else ''}")
    if report.unmatched_isaac_bodies:
        print(f"Unmatched Isaac bodies ({len(report.unmatched_isaac_bodies)}): "
              f"{', '.join(report.unmatched_isaac_bodies[:8])}"
              f"{'...' if len(report.unmatched_isaac_bodies) > 8 else ''}")

    print("\nPer-link mass / COM (matched):")
    print(f"  {'link':<16} {'m_urdf':>8} {'m_isaac':>8} {'Δm%':>7} {'|Δcom| mm':>10}")
    for row in report.matched_links:
        print(
            f"  {row['urdf_link']:<16} "
            f"{row['mass_urdf']:8.4f} {row['mass_isaac']:8.4f} "
            f"{row['mass_delta_pct']:7.2f} {row['com_delta_norm_m']*1000:10.2f}"
        )

    print("\nFoot FK comparison (world frame):")
    fk = report.foot_kinematics
    if fk:
        for mode in ("nominal_hardcoded_base", "isaac_readback_state"):
            errs = fk.get("foot_error_world_m", {}).get(mode, {})
            if errs:
                label = "legacy hardcoded" if "nominal" in mode else "adam FK with Isaac readback"
                print(f"  [{label}]")
                for foot, err in errs.items():
                    print(f"    {foot}: |Δ| = {err*1000:.2f} mm")
        base_errs = fk.get("foot_error_base_relative_m", {}).get("isaac_readback_state", {})
        if base_errs:
            print("  [base-relative, isaac readback]")
            for foot, err in base_errs.items():
                print(f"    {foot}: |Δ| = {err*1000:.2f} mm")
        j_delta = fk.get("joint_delta_nominal_vs_isaac", {})
        if j_delta:
            print(
                f"  Joint delta nominal vs Isaac: max "
                f"{j_delta.get('max_abs_rad', 0.0):.4f} rad"
            )
    elif report.geometry_error:
        print(f"  (skipped — {report.geometry_error})")
    elif report.foot_position_errors_m:
        for foot, err in report.foot_position_errors_m.items():
            print(f"  {foot}: |Δ| = {err*1000:.2f} mm")
    else:
        print("  (not computed)")

    print("\nLeg segment lengths (URDF static joint origins vs Isaac runtime body distances):")
    for source, segs in report.leg_segment_lengths.items():
        parts = ", ".join(f"{k}={v*1000:.1f}mm" for k, v in segs.items())
        print(f"  {source}: {parts}")

    if report.flags:
        print("\nFlags:")
        for flag in report.flags:
            print(f"  - {flag}")
    else:
        print("\nNo model-parameter flags above tolerance.")


def _print_section(title: str, payload: dict) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    for key, val in payload.items():
        if key == "flags":
            continue
        print(f"  {key}: {val}")
    flags = payload.get("flags") or []
    if flags:
        print("  Flags:")
        for flag in flags:
            print(f"    - {flag}")
    else:
        print("  Flags: none")


def main() -> None:
    sections = {s.strip().upper() for s in args_cli.sections.split(",") if s.strip()}
    unknown = sections - {"A", "B", "C", "D"}
    if unknown:
        raise ValueError(f"Unknown sections: {sorted(unknown)}")

    run_tag = args_cli.run_tag or datetime.now().strftime("physics_diag_%Y%m%d_%H%M%S")
    out_dir = Path(args_cli.output_dir) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "physics_diagnostics.json"

    report: dict = {
        "created_at": datetime.now().isoformat(),
        "sections_requested": sorted(sections),
        "settings": {
            "mass_rel_tol": args_cli.mass_rel_tol,
            "com_abs_tol_m": args_cli.com_abs_tol_m,
            "foot_pos_abs_tol_m": args_cli.foot_pos_abs_tol_m,
        },
    }

    try:
        if "A" in sections:
            urdf_path = _resolve_urdf_path(args_cli.urdf_path)
            model_report = run_model_compare(
                urdf_path,
                args_cli.mass_rel_tol,
                args_cli.com_abs_tol_m,
                args_cli.foot_pos_abs_tol_m,
            )
            report["A_model_compare"] = asdict(model_report)
            _print_model_compare(model_report)

        needs_traj = sections & {"B", "C", "D"}
        if needs_traj:
            paths = _resolve_trajectory_paths(args_cli)
            state = np.load(paths["state_traj"])
            jvel = np.load(paths["joint_vel_traj"])
            grf = np.load(paths["grf_traj"])
            contact = np.load(paths["contact_sequence"]) if paths["contact_sequence"] else None

            if args_cli.source_dt is not None and args_cli.source_dt > 0:
                dt = float(args_cli.source_dt)
            else:
                from rl_isaac.upsample_reference import resolve_source_dt

                dt = resolve_source_dt(None, args_cli.traj_dir or None)

            report["trajectory_paths"] = paths
            report["source_dt"] = dt

            if "B" in sections:
                _log("[B] Checking reference contact / GRF consistency...")
                b_report = run_reference_contact_checks(state, grf, jvel, contact, dt)
                report["B_contact_reference"] = b_report
                _print_section("[B] Reference contact / GRF consistency", b_report)

            if "C" in sections:
                _log("[C] Checking velocity-frame ID sensitivity...")
                c_report = run_velocity_frame_check(state, jvel, grf, dt)
                report["C_velocity_frame"] = c_report
                _print_section("[C] Velocity-frame ID sensitivity", c_report)

            if "D" in sections:
                _log("[D] Checking reference self-consistency...")
                d_report = run_reference_consistency(state, jvel, grf, contact, dt)
                report["D_reference_consistency"] = d_report
                _print_section("[D] Reference self-consistency", d_report)

        _save_report(out_path, report)
        _log(f"\nReport written to {out_path}")
    except Exception as exc:
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        _save_report(out_path, report)
        _log(f"\nPartial report written to {out_path} (section failed)")
        raise


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        simulation_app.close()
    if exit_code != 0:
        raise SystemExit(exit_code)
