"""Single source of truth for Unitree Go2 robot configuration.

Parses the URDF for hardware specs, hardcodes what the URDF cannot provide
(operational limits, composite inertia, capability limits), and exports
everything as module-level variables.  Replaces both ``config.py`` and
``configs/robots/go2.py``.

LLM data flow — which fields are sent to which LLM call:
 ┌───────────────────────────────┬─────────┬─────────┬─────────┐
 │ Field                         │ Codegen │ Scoring │ Summary │
 ├───────────────────────────────┼─────────┼─────────┼─────────┤
 │ composite_mass                │    ✓    │    ✓    │         │
 │ initial_crouch_qpos[2] (h)   │    ✓    │    ✓    │         │
 │ urdf_joint_limits_lower/upper │    ✓    │    ✓    │         │
 │ urdf_joint_velocities         │    ✓    │    ✓    │         │
 │ urdf_joint_efforts            │         │         │         │
 │ grf_limits                    │    ✓    │    ✓    │         │
 │ body_half_extents             │    ✓    │         │         │
 │ leg_lengths                   │    ✓    │         │         │
 │ hip_spacing                   │    ✓    │         │         │
 │ capability_limits             │    ✓    │    ✓    │         │
 │ experiment.mu_ground          │    ✓    │    ✓    │         │
 │ mpc_config.mpc_dt / duration  │    ✓    │         │         │
 ├───────────────────────────────┼─────────┴─────────┴─────────┤
 │ composite_inertia             │ internal (dynamics + energy  │
 │                               │ analysis only)               │
 │ analysis_thresholds           │ internal (motion analysis    │
 │                               │ code only, not sent to LLMs) │
 │ experiment.gravity_constant   │ internal (dynamics + energy  │
 │                               │ analysis only)               │
 │ initial_crouch_qpos (full)    │ runtime arg to LLM's         │
 │                               │ generate_reference_trajectory │
 │ solver_config                 │ internal (IPOPT only)        │
 └───────────────────────────────┴──────────────────────────────┘

 Codegen  = get_system_prompt()          → prompts.py:get_robot_details()
 Scoring  = evaluate_iteration_unified() → llm_calls.py:_format_robot_context()
 Summary  = generate_iteration_summary() → no robot context (metrics only)
"""

from __future__ import annotations

import os
import types
import xml.etree.ElementTree as ET
from typing import Any

import gym_quadruped
import numpy as np
from gym_quadruped.robot_cfgs import get_robot_config

# ────────────────────────────────────────────────────────────────────────────
# Section 1 — URDF parsing
# ────────────────────────────────────────────────────────────────────────────

_URDF_PATH = os.path.dirname(gym_quadruped.__file__) + "/robot_model/go2/go2.urdf"
_XML_PATH = os.path.dirname(gym_quadruped.__file__) + "/robot_model/go2/go2.xml"

_tree = ET.parse(_URDF_PATH)
_root = _tree.getroot()

# Joint order expected by the dynamics model
JOINT_ORDER = _JOINT_ORDER = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

# Build lookup tables from URDF
_joints: dict[str, ET.Element] = {}
for _j in _root.iter("joint"):
    _name = _j.get("name", "")
    if _name:
        _joints[_name] = _j

_links: dict[str, ET.Element] = {}
for _l in _root.iter("link"):
    _name = _l.get("name", "")
    if _name:
        _links[_name] = _l


def _parse_limit(joint_elem: ET.Element) -> dict[str, float]:
    lim = joint_elem.find("limit")
    if lim is None:
        return {"lower": 0.0, "upper": 0.0, "effort": 0.0, "velocity": 0.0}
    return {
        "lower": float(lim.get("lower", "0")),
        "upper": float(lim.get("upper", "0")),
        "effort": float(lim.get("effort", "0")),
        "velocity": float(lim.get("velocity", "0")),
    }


def _parse_origin_xyz(joint_elem: ET.Element) -> tuple[float, float, float]:
    origin = joint_elem.find("origin")
    if origin is None:
        return (0.0, 0.0, 0.0)
    xyz = origin.get("xyz", "0 0 0").split()
    return (float(xyz[0]), float(xyz[1]), float(xyz[2]))


# Per-joint URDF limits (in _JOINT_ORDER)
_urdf_lower: list[float] = []
_urdf_upper: list[float] = []
_urdf_vel: list[float] = []
_urdf_eff: list[float] = []
for _jn in _JOINT_ORDER:
    _lim = _parse_limit(_joints[_jn])
    _urdf_lower.append(_lim["lower"])
    _urdf_upper.append(_lim["upper"])
    _urdf_vel.append(_lim["velocity"])
    _urdf_eff.append(_lim["effort"])

urdf_joint_limits_lower: np.ndarray = np.array(_urdf_lower)
urdf_joint_limits_upper: np.ndarray = np.array(_urdf_upper)
urdf_joint_velocities: np.ndarray = np.array(_urdf_vel)
urdf_joint_efforts: np.ndarray = np.array(_urdf_eff)

# Base link collision box → body half-extents
_base_collision = _links["base"].find("collision")
assert _base_collision is not None
_base_box = _base_collision.find("geometry/box")
assert _base_box is not None
_box_size = [float(v) for v in _base_box.get("size", "0 0 0").split()]
body_half_extents: tuple[float, float, float] = (
    _box_size[0] / 2.0,
    _box_size[1] / 2.0,
    _box_size[2] / 2.0,
)

# Joint origins → leg segment lengths and hip offsets
_fl_hip_xyz = _parse_origin_xyz(_joints["FL_hip_joint"])
_fr_hip_xyz = _parse_origin_xyz(_joints["FR_hip_joint"])
_rl_hip_xyz = _parse_origin_xyz(_joints["RL_hip_joint"])
_rr_hip_xyz = _parse_origin_xyz(_joints["RR_hip_joint"])

_fl_thigh_xyz = _parse_origin_xyz(_joints["FL_thigh_joint"])
_fl_calf_xyz = _parse_origin_xyz(_joints["FL_calf_joint"])
_fl_foot_xyz = _parse_origin_xyz(_joints["FL_foot_joint"])

_thigh_length = abs(_fl_calf_xyz[2])  # z-offset from thigh to calf joint
_calf_length = abs(_fl_foot_xyz[2])  # z-offset from calf to foot joint

leg_lengths: dict[str, float] = {
    "thigh": _thigh_length,
    "calf": _calf_length,
    "total": _thigh_length + _calf_length,
}

hip_spacing: dict[str, float] = {
    "front_rear_from_com": abs(_fl_hip_xyz[0]),  # x-offset of front hips
    "left_right_from_com": abs(_fl_hip_xyz[1]) + abs(_fl_thigh_xyz[1]),  # y total
}

# Sum all link masses
_total_mass = 0.0
for _link in _root.iter("link"):
    _inertial = _link.find("inertial")
    if _inertial is not None:
        _mass_elem = _inertial.find("mass")
        if _mass_elem is not None:
            _total_mass += float(_mass_elem.get("value", "0"))
urdf_total_mass: float = _total_mass


# ────────────────────────────────────────────────────────────────────────────
# Section 3 — Values that can't come from URDF
# ────────────────────────────────────────────────────────────────────────────

# Joint limits and velocity limits are read directly from the URDF (Section 1).
# The variables used by robot_data are the urdf_joint_limits_* arrays.

grf_limits: float = 500.0

# Mass comes from URDF (sum of all link masses)
composite_mass: float = urdf_total_mass

# Composite inertia (whole-body, about COM — cannot be derived from URDF)
composite_inertia: np.ndarray = np.array(
    [
        [1.58460467e-01, 1.21660000e-04, -1.55444692e-02],
        [1.21660000e-04, 4.68645637e-01, -3.12000000e-05],
        [-1.55444692e-02, -3.12000000e-05, 5.24474661e-01],
    ]
)

# Initial crouch pose (pre-loaded jump stance, height ~0.21m)
initial_crouch_qpos: np.ndarray = np.zeros(19)
initial_crouch_qpos[0:3] = [0.0, 0.0, 0.2117]
initial_crouch_qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
initial_crouch_qpos[7:19] = [
    0.0,
    1.0,
    -2.1,
    0.0,
    1.0,
    -2.1,
    0.0,
    1.0,
    -2.1,
    0.0,
    1.0,
    -2.1,
]
initial_crouch_qvel: np.ndarray = np.zeros(18)

# Physical capability limits (for LLM prompts and scoring)
capability_limits: dict[str, Any] = {
    "min_height_gain_normal": 0.15,
    "max_height_gain_normal": 0.25,
    "max_height_gain_aggressive": 0.4,
    "min_takeoff_vz": 1.8,
    "max_takeoff_vz": 3.0,
    "min_flight_duration": 0.3,
    "max_flight_duration": 0.55,
    "min_peak_grf_total": 900.0,
    "max_peak_grf_total": 1200.0,
    "min_peak_grf_bodyweight_multiple": 6.0,
    "max_peak_grf_bodyweight_multiple": 8.0,
    "min_com_accel_typical_g": 4.0,
    "max_com_accel_typical_g": 6.0,
    "max_com_accel_feasible_g": 13.0,
    "min_peak_angular_velocity": 8.0,
    "peak_angular_velocity": 18.0,
}

# Analysis thresholds (used by motion quality reports and trajectory classification)
analysis_thresholds: dict[str, float] = {
    "ground_penetration_tolerance": 0.005,  # 5mm — feet/body below ground
    "swing_clearance_min": 0.005,  # 5mm — min foot height during swing
    "landing_foot_height_tolerance": 0.01,  # 1cm — foot considered "on ground"
    "flight_height_offset": 0.05,  # 5cm above initial → classified as airborne
    "phantom_force_threshold": 1.0,  # 1N — GRF during flight = phantom
    "missing_force_threshold": 0.1,  # 0.1N — no GRF_z during stance = missing
    "chattering_transition_limit": 4,  # >4 contact transitions = chattering
    "joint_limit_proximity": 0.05,  # 5% of range → near limit warning
    "manipulability_warning": 0.001,  # below this → singularity warning
    "foot_spread_ratio_min": 0.7,  # landing foot spread < 70% nominal → narrow
    "foot_spread_ratio_max": 1.3,  # landing foot spread > 130% nominal → wide
    "negligible_force_threshold": 1.0,  # 1N — skip negligible forces in analysis
}


# ────────────────────────────────────────────────────────────────────────────
# Section 4 — Namespace instances + settings (same interface as old config.py)
# ────────────────────────────────────────────────────────────────────────────

robot: str = "go2"

robot_data = types.SimpleNamespace(
    name="go2",
    mass=composite_mass,
    inertia=composite_inertia,
    urdf_filename=_URDF_PATH,
    xml_filename=_XML_PATH,
    joint_limits_lower=urdf_joint_limits_lower,
    joint_limits_upper=urdf_joint_limits_upper,
    robot_cfg=get_robot_config(robot_name="go2"),
    joint_velocity_limits=urdf_joint_velocities,
    joint_efforts=urdf_joint_efforts,
    grf_limits=grf_limits,
    initial_qpos=initial_crouch_qpos,
    initial_qvel=initial_crouch_qvel,
)

# Structural constants derived from URDF/joint order
N_JOINTS: int = len(JOINT_ORDER)
N_LEGS: int = N_JOINTS // 3
STATES_DIM: int = 30  # 12 base + 12 joint + 6 integral
INPUTS_DIM: int = 24  # 12 joint velocities + 12 GRFs (3 per foot)

duration: float = 1.0

# Default gait timing (used by MPC configs — single source of truth)
default_pre_flight_stance_duration: float = 0.3
default_flight_duration: float = 0.4

# Default MPC time steps per constraint mode
default_mpc_dt_complementarity: float = 0.02
default_mpc_dt_standard: float = 0.1

# Select constraint mode: "standard" or "complementarity"
CONSTRAINT_MODE: str = "complementarity"

experiment = types.SimpleNamespace(
    mu_ground=0.8,
    gravity_constant=9.81,
    duration=1.0,
    sim_dt=0.01,
    terrain="flat",
    initial_qpos=robot_data.initial_qpos,
    initial_qvel=robot_data.initial_qvel,
    render=True,
)


def __getattr__(name: str) -> Any:
    if name == "mpc_config":
        from mpc.defaults import mpc_config as _mpc_config

        globals()["mpc_config"] = _mpc_config
        return _mpc_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


solver_config: dict[str, Any] = {
    "ipopt.print_level": 5,
    "print_time": True,
    "ipopt.max_iter": 5000,
    "ipopt.tol": 1e-4,
    "ipopt.acceptable_tol": 1e-3,
    "ipopt.mu_init": 1e-2,
    "ipopt.mu_strategy": "adaptive",
    "ipopt.alpha_for_y": "primal",
    "ipopt.recalc_y": "yes",
    "ipopt.max_wall_time": 3600.0,
}

plot_quantities = [
    "base_position",
    "base_linear_velocity",
    "base_orientation",
    "base_angular_velocity",
    "ground_reaction_forces",
]
