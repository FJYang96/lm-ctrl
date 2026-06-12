"""Tests for MPC → control-rate reference upsampling."""

from __future__ import annotations

import numpy as np

from rl_isaac.kinematic_upsample import (
    KinematicUpsampleParams,
    interpolate_stance_aware_grf,
    kinematic_upsample_joints,
)
from rl_isaac.upsample_reference import upsample_reference_arrays
from utils.conversion import (
    MPC_X_BASE_POS,
    MPC_X_BASE_VEL,
    MPC_X_Q_JOINTS,
)


def _make_synthetic_reference(n_mpc: int = 20, source_dt: float = 0.05) -> tuple:
    t = np.arange(n_mpc + 1) * source_dt
    state = np.zeros((n_mpc + 1, 30))
    state[:, 0] = 0.1 * t
    state[:, 2] = 0.21 + 0.05 * np.sin(2 * np.pi * t)
    state[:, 6:9] = np.stack(
        [0.1 * t, 0.05 * np.sin(t), 0.02 * t], axis=-1
    )
    state[:, 12:24] = 0.5 * np.sin(2 * np.pi * t)[:, None]
    jvel = np.diff(state[:, 12:24], axis=0) / source_dt
    grf = np.zeros((n_mpc, 12))
    grf[:, 2::3] = 50.0
    contact = np.ones((4, n_mpc))
    return state, jvel, grf, contact


def test_upsample_horizon_and_duration():
    state, jvel, grf, contact = _make_synthetic_reference()
    state_up, jvel_up, grf_up, contact_up, ff_up, meta = upsample_reference_arrays(
        state, jvel, grf, contact, source_dt=0.05, target_dt=0.01, method="hermite"
    )
    assert meta["source_horizon"] == 20
    assert meta["target_horizon"] == 100
    assert meta["joint_interp"] == "cubic_hermite"
    assert ff_up is None
    assert state_up.shape == (101, 30)
    assert jvel_up.shape == (100, 12)
    assert grf_up.shape == (100, 12)
    assert contact_up.shape == (4, 100)
    assert abs(meta["target_duration_s"] - 1.0) < 1e-6
    np.testing.assert_allclose(state_up[0], state[0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(state_up[-1], state[-1], rtol=0, atol=1e-3)


def test_noop_when_dts_match():
    state, jvel, grf, contact = _make_synthetic_reference(n_mpc=10, source_dt=0.01)
    state_up, jvel_up, grf_up, contact_up, ff_up, meta = upsample_reference_arrays(
        state, jvel, grf, contact, source_dt=0.01, target_dt=0.01
    )
    assert not meta["upsampled"]
    assert meta["target_horizon"] == 10
    assert ff_up is None
    np.testing.assert_array_equal(state_up, state)
    np.testing.assert_array_equal(jvel_up, jvel)


def test_grf_zero_order_hold():
    state, jvel, grf, contact = _make_synthetic_reference(n_mpc=4, source_dt=0.05)
    grf[2, 2] = 999.0
    _, _, grf_up, _, _, meta = upsample_reference_arrays(
        state, jvel, grf, contact, source_dt=0.05, target_dt=0.01, method="hermite"
    )
    assert meta["target_horizon"] == 20
    assert np.all(grf_up[0:5, 2] == grf[0, 2])
    assert np.all(grf_up[5:10, 2] == grf[1, 2])
    assert np.any(grf_up[:, 2] == 999.0)


def test_joint_hermite_endpoint_constraints():
    state, jvel, grf, contact = _make_synthetic_reference(n_mpc=8, source_dt=0.05)
    source_dt = 0.05
    target_dt = 0.01
    state_up, jvel_up, _, _, _, meta = upsample_reference_arrays(
        state,
        jvel,
        grf,
        contact,
        source_dt=source_dt,
        target_dt=target_dt,
        method="hermite",
    )
    assert meta["joint_interp"] == "cubic_hermite"

    n_mpc = jvel.shape[0]
    knot_times = np.linspace(0.0, n_mpc * source_dt, n_mpc + 1)
    knot_idx = np.round(knot_times / target_dt).astype(int)

    from utils.conversion import MPC_X_Q_JOINTS

    q_knots = state[:, MPC_X_Q_JOINTS]
    m_knots = np.zeros((n_mpc + 1, 12), dtype=np.float64)
    m_knots[:n_mpc] = jvel
    m_knots[n_mpc] = jvel[-1]

    np.testing.assert_allclose(
        state_up[knot_idx, MPC_X_Q_JOINTS], q_knots, rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(jvel_up[knot_idx[:n_mpc]], m_knots[:n_mpc], rtol=0, atol=1e-8)


def test_joint_hermite_tangents_at_source_knots():
    state, jvel, grf, contact = _make_synthetic_reference(n_mpc=12, source_dt=0.05)
    source_dt = 0.05
    target_dt = 0.01
    _, jvel_up, _, _, _, _ = upsample_reference_arrays(
        state,
        jvel,
        grf,
        contact,
        source_dt=source_dt,
        target_dt=target_dt,
        method="hermite",
    )

    n_mpc = jvel.shape[0]
    knot_idx = np.round(np.linspace(0.0, n_mpc * source_dt, n_mpc + 1) / target_dt).astype(
        int
    )
    m_knots = np.zeros((n_mpc + 1, 12), dtype=np.float64)
    m_knots[:n_mpc] = jvel
    m_knots[n_mpc] = jvel[-1]

    np.testing.assert_allclose(
        jvel_up[knot_idx[:n_mpc]], m_knots[:n_mpc], rtol=0, atol=1e-8
    )


def test_upsample_200hz():
    state, jvel, grf, contact = _make_synthetic_reference(n_mpc=10, source_dt=0.05)
    _, _, _, _, _, meta = upsample_reference_arrays(
        state, jvel, grf, contact, source_dt=0.05, target_dt=0.005, method="hermite"
    )
    assert meta["target_horizon"] == 100
    assert abs(meta["target_duration_s"] - 0.5) < 1e-6


class _FakeKinematicsModel:
    def __init__(self) -> None:
        for leg in ("fl", "fr", "rl", "rr"):
            foot = {"fl": 0, "fr": 1, "rl": 2, "rr": 3}[leg]
            setattr(self, f"foot_center_position_{leg}_fun", self._fk_fun(foot))
            setattr(self, f"foot_center_jacobian_{leg}_fun", self._jac_fun(foot))

    @staticmethod
    def _fk_fun(foot: int):
        def fk(H, q):
            q_arr = np.asarray(q, dtype=np.float64)
            return np.asarray(H, dtype=np.float64)[:3, 3] + q_arr[foot * 3 : foot * 3 + 3]

        return fk

    @staticmethod
    def _jac_fun(foot: int):
        def jac(_H, _q):
            J = np.zeros((3, 18), dtype=np.float64)
            J[:, :3] = np.eye(3)
            J[:, 6 + foot * 3 : 9 + foot * 3] = np.eye(3)
            return J

        return jac


def _make_fake_fk_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_mpc = 2
    state = np.zeros((n_mpc + 1, 30), dtype=np.float64)
    state[:, 0] = [0.0, 0.05, 0.1]
    state[:, MPC_X_BASE_VEL] = np.array([1.0, 0.0, 0.0])
    for k in range(n_mpc + 1):
        for foot in range(4):
            state[k, 12 + foot * 3 : 15 + foot * 3] = -state[k, MPC_X_BASE_POS]
    jvel = np.zeros((n_mpc, 12), dtype=np.float64)
    jvel[:, 0::3] = -1.0
    grf = np.zeros((n_mpc, 12), dtype=np.float64)
    grf[:, 2::3] = 50.0
    contact = np.ones((4, n_mpc), dtype=np.float64)
    return state, jvel, grf, contact


def test_fk_ik_stance_feet_remain_stationary():
    state, jvel, grf, contact = _make_fake_fk_reference()
    source_dt = 0.05
    target_dt = 0.025
    t_src = np.linspace(0.0, 0.1, 3)
    t_state = np.linspace(0.0, 0.1, 5)
    t_u = np.arange(4, dtype=np.float64) * target_dt
    state_out = np.zeros((5, 30), dtype=np.float64)
    state_out[:, 0] = np.linspace(0.0, 0.1, 5)
    state_out[:, MPC_X_BASE_VEL] = np.array([1.0, 0.0, 0.0])
    q_interp_state = np.zeros((5, 12), dtype=np.float64)
    q_interp_u = np.zeros((4, 12), dtype=np.float64)
    params = KinematicUpsampleParams(compute_feedforward=False, alpha=1e-6, beta=1e-6)

    state_fk, jvel_fk, grf_fk, contact_fk, ff_fk, meta = kinematic_upsample_joints(
        state,
        jvel,
        grf,
        contact,
        state_out,
        q_interp_state,
        q_interp_u,
        t_src,
        t_state,
        t_u,
        source_dt,
        target_dt,
        _FakeKinematicsModel(),
        params,
    )

    assert ff_fk is None
    assert contact_fk.shape == (4, 4)
    assert meta["ik_max_final_error"] < 1e-4
    for foot in range(4):
        q_leg = state_fk[:, 12 + foot * 3 : 15 + foot * 3]
        foot_world = state_fk[:, MPC_X_BASE_POS] + q_leg
        np.testing.assert_allclose(
            foot_world, np.broadcast_to(foot_world[0], foot_world.shape), atol=1e-4
        )
    np.testing.assert_allclose(jvel_fk[:, 0::3], -1.0, atol=1e-4)


def test_stance_grf_interpolates_linearly_and_zoh_before_swing():
    grf = np.zeros((3, 12), dtype=np.float64)
    grf[:, 2] = [10.0, 20.0, 0.0]
    contact = np.ones((4, 3), dtype=np.float64)
    contact[0] = [1.0, 1.0, 0.0]
    t_u = np.array([0.0, 0.025, 0.05, 0.075, 0.1])

    out = interpolate_stance_aware_grf(grf, contact, t_u, source_dt=0.05)

    np.testing.assert_allclose(out[:3, 2], [10.0, 15.0, 20.0], atol=1e-9)
    assert out[3, 2] == 20.0
    assert out[4, 2] == 0.0


if __name__ == "__main__":
    test_upsample_horizon_and_duration()
    test_noop_when_dts_match()
    test_grf_zero_order_hold()
    test_joint_hermite_endpoint_constraints()
    test_joint_hermite_tangents_at_source_knots()
    test_upsample_200hz()
    test_fk_ik_stance_feet_remain_stationary()
    test_stance_grf_interpolates_linearly_and_zoh_before_swing()
    print("upsample_reference tests passed")
