"""Motion analysis: motion quality report.

Public API:
    compute_motion_quality_report() — formatted text report with raw metrics per section
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...logging_config import logger
from ._physics_stability import (
    _section_angular_momentum,
    _section_energy_continuity,
    _section_friction_cone,
    _section_terminal_stability,
)
from ._quality_kinematics import (
    _section_contact_quality,
    _section_joint_quality,
    _section_manipulability,
)
from ._surface_contact import (
    _section_grf_contact,
    _section_ground_penetration,
    _section_smoothness,
)

# Foot names in order matching the 4x3 GRF layout and contact_sequence rows
_FOOT_NAMES = ("FL", "FR", "RL", "RR")

__all__ = ["compute_motion_quality_report"]


def compute_motion_quality_report(
    state_traj: np.ndarray,
    grf_traj: np.ndarray,
    joint_vel_traj: np.ndarray,
    mpc_dt: float,
    contact_sequence: np.ndarray | None,
    kindyn_model: Any,
    joint_limits_lower: np.ndarray,
    joint_limits_upper: np.ndarray,
    robot_mass: float,
    mu_friction: float,
    torque_limits: np.ndarray | None = None,
) -> str:
    """Compute a comprehensive motion quality report from trajectory data.

    Args:
        state_traj: (N+1, 24) state trajectory — pos(3), vel(3), euler(3), omega(3), joints(12).
        grf_traj: (N, 12) ground reaction forces — 4 feet x 3 forces.
        joint_vel_traj: (N, 12) joint velocities.
        mpc_dt: Timestep duration in seconds.
        contact_sequence: (4, N) binary contact sequence, or None.
        kindyn_model: KinoDynamic_Model instance with FK/Jacobian functions.
        joint_limits_lower: (12,) per-joint lower limits.
        joint_limits_upper: (12,) per-joint upper limits.
        robot_mass: Robot mass in kg.
        mu_friction: Ground friction coefficient.
        torque_limits: (12,) per-joint torque limits, or None.

    Returns:
        Formatted text report with raw metrics per section.
    """
    sections: list[tuple[str, list[str]]] = []

    try:
        lns = _section_smoothness(state_traj, mpc_dt, contact_sequence)
        sections.append(("SMOOTHNESS", lns))
    except Exception as e:
        logger.warning(f"Motion quality: smoothness computation failed: {e}")
        sections.append(("SMOOTHNESS", [f"  Computation failed: {e}"]))

    try:
        lns = _section_ground_penetration(state_traj, contact_sequence, kindyn_model)
        sections.append(("GROUND PENETRATION", lns))
    except Exception as e:
        logger.warning(f"Motion quality: ground penetration computation failed: {e}")
        sections.append(("GROUND PENETRATION", [f"  Computation failed: {e}"]))

    try:
        lns = _section_grf_contact(grf_traj, contact_sequence)
        sections.append(("GRF-CONTACT CONSISTENCY", lns))
    except Exception as e:
        logger.warning(f"Motion quality: GRF-contact computation failed: {e}")
        sections.append(("GRF-CONTACT CONSISTENCY", [f"  Computation failed: {e}"]))

    try:
        lns = _section_friction_cone(grf_traj, contact_sequence, mu_friction)
        sections.append(("FRICTION CONE", lns))
    except Exception as e:
        logger.warning(f"Motion quality: friction cone computation failed: {e}")
        sections.append(("FRICTION CONE", [f"  Computation failed: {e}"]))

    try:
        lns = _section_angular_momentum(state_traj, contact_sequence, mpc_dt)
        sections.append(("ANGULAR MOMENTUM (FLIGHT)", lns))
    except Exception as e:
        logger.warning(f"Motion quality: angular momentum computation failed: {e}")
        sections.append(("ANGULAR MOMENTUM (FLIGHT)", [f"  Computation failed: {e}"]))

    try:
        lns = _section_energy_continuity(state_traj, robot_mass, mpc_dt)
        sections.append(("ENERGY CONTINUITY", lns))
    except Exception as e:
        logger.warning(f"Motion quality: energy continuity computation failed: {e}")
        sections.append(("ENERGY CONTINUITY", [f"  Computation failed: {e}"]))

    try:
        lns = _section_terminal_stability(state_traj)
        sections.append(("TERMINAL STABILITY", lns))
    except Exception as e:
        logger.warning(f"Motion quality: terminal stability computation failed: {e}")
        sections.append(("TERMINAL STABILITY", [f"  Computation failed: {e}"]))

    try:
        lns = _section_contact_quality(
            state_traj,
            contact_sequence,
            kindyn_model,
            mpc_dt,
            joint_limits_lower,
            joint_limits_upper,
        )
        sections.append(("CONTACT QUALITY", lns))
    except Exception as e:
        logger.warning(f"Motion quality: contact quality computation failed: {e}")
        sections.append(("CONTACT QUALITY", [f"  Computation failed: {e}"]))

    try:
        lns = _section_joint_quality(
            state_traj,
            grf_traj,
            joint_limits_lower,
            joint_limits_upper,
            kindyn_model,
            torque_limits,
        )
        sections.append(("JOINT QUALITY", lns))
    except Exception as e:
        logger.warning(f"Motion quality: joint quality computation failed: {e}")
        sections.append(("JOINT QUALITY", [f"  Computation failed: {e}"]))

    try:
        lns = _section_manipulability(state_traj, kindyn_model)
        sections.append(("MANIPULABILITY", lns))
    except Exception as e:
        logger.warning(f"Motion quality: manipulability computation failed: {e}")
        sections.append(("MANIPULABILITY", [f"  Computation failed: {e}"]))

    # Assemble report
    report_lines: list[str] = []
    for name, lns in sections:
        report_lines.append(f"{name}:")
        report_lines.extend(lns)
        report_lines.append("")

    return "\n".join(report_lines).strip()
