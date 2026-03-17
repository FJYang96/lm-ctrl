"""Shared geometry helpers for motion analysis sections."""

from __future__ import annotations

from typing import Any

import numpy as np


def _euler_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX intrinsic euler angles to 3x3 rotation matrix (world <- body)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    return R


def _build_H(com_pos: np.ndarray, euler: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transform matching model.py convention."""
    H = np.eye(4)
    H[0:3, 0:3] = _euler_to_rotation(euler[0], euler[1], euler[2])
    H[0:3, 3] = com_pos
    return H


def _eval_fk(fk_fun: Any, H: np.ndarray, joint_pos: np.ndarray) -> np.ndarray:
    """Evaluate a CasADi FK function and return foot position as (3,) ndarray."""
    result = fk_fun(H, joint_pos)
    return np.array(result[0:3, 3]).flatten()


def _eval_jacobian(jac_fun: Any, H: np.ndarray, joint_pos: np.ndarray) -> np.ndarray:
    """Evaluate a CasADi Jacobian function and return (3, 18) translational Jacobian."""
    result = jac_fun(H, joint_pos)
    return np.array(result[0:3, :]).reshape(3, -1)
