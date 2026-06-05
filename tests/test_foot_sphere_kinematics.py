"""Unit tests for foot sphere-center FK and Jacobian."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import go2_config
from mpc.dynamics.model import KinoDynamic_Model


@pytest.fixture(scope="module")
def model() -> KinoDynamic_Model:
    return KinoDynamic_Model()


def _build_H(base_pos: np.ndarray, euler: np.ndarray) -> np.ndarray:
    H = np.eye(4)
    H[:3, :3] = Rotation.from_euler("xyz", euler).as_matrix()
    H[:3, 3] = base_pos
    return H


def test_foot_sphere_params_from_urdf() -> None:
    assert go2_config.foot_sphere_radius == pytest.approx(0.022)
    assert np.allclose(
        go2_config.foot_sphere_center_offset,
        np.array([-0.002, 0.0, 0.0]),
    )


def test_foot_center_offset_from_foot_frame(model: KinoDynamic_Model) -> None:
    base_pos = go2_config.initial_crouch_qpos[:3].copy()
    euler = np.zeros(3)
    q = go2_config.initial_crouch_qpos[7:19].copy()
    H = _build_H(base_pos, euler)

    offset = go2_config.foot_sphere_center_offset
    for prefix in ("FL", "FR", "RL", "RR"):
        fk = getattr(model, f"forward_kinematics_{prefix}_fun")
        center_fun = getattr(model, f"foot_center_position_{prefix.lower()}_fun")
        H_foot = np.array(fk(H, q)).reshape(4, 4)
        p_foot = H_foot[:3, 3]
        R_foot = H_foot[:3, :3]
        p_center = np.array(center_fun(H, q)).flatten()
        np.testing.assert_allclose(p_center, p_foot + R_foot @ offset, atol=1e-9)


def test_crouch_stance_foot_center_height(model: KinoDynamic_Model) -> None:
    """At nominal crouch, sphere centers should be near radius above ground."""
    base_pos = go2_config.initial_crouch_qpos[:3].copy()
    euler = np.zeros(3)
    q = go2_config.initial_crouch_qpos[7:19].copy()
    H = _build_H(base_pos, euler)
    R = go2_config.foot_sphere_radius

    for prefix in ("FL", "FR", "RL", "RR"):
        center_fun = getattr(model, f"foot_center_position_{prefix.lower()}_fun")
        z_center = float(np.array(center_fun(H, q)).flatten()[2])
        assert z_center == pytest.approx(R, abs=0.01), (
            f"{prefix} center z={z_center:.4f}, expected ~{R:.4f}"
        )


def test_foot_center_jacobian_finite_difference(model: KinoDynamic_Model) -> None:
    base_pos = np.array([0.0, 0.0, 0.25])
    euler = np.zeros(3)
    q = go2_config.initial_crouch_qpos[7:19].copy()
    H = _build_H(base_pos, euler)

    v_gen = np.zeros(18)
    v_gen[8] = 0.5  # FL thigh joint velocity
    dt = 1e-7

    center_fun = model.foot_center_position_fl_fun
    jac_fun = model.foot_center_jacobian_fl_fun
    p0 = np.array(center_fun(H, q)).flatten()
    J = np.array(jac_fun(H, q)).reshape(3, -1)
    v_analytic = J @ v_gen

    q_p = q + v_gen[6:18] * dt
    p1 = np.array(center_fun(H, q_p)).flatten()
    v_numeric = (p1 - p0) / dt

    np.testing.assert_allclose(v_analytic, v_numeric, rtol=1e-5, atol=1e-8)
