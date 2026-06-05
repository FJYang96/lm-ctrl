"""Tests for MPC → control-rate reference upsampling."""

from __future__ import annotations

import numpy as np

from rl_isaac.upsample_reference import upsample_reference_arrays


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
    state_up, jvel_up, grf_up, contact_up, meta = upsample_reference_arrays(
        state, jvel, grf, contact, source_dt=0.05, target_dt=0.02
    )
    assert meta["source_horizon"] == 20
    assert meta["target_horizon"] == 50
    assert state_up.shape == (51, 30)
    assert jvel_up.shape == (50, 12)
    assert grf_up.shape == (50, 12)
    assert contact_up.shape == (4, 50)
    assert abs(meta["target_duration_s"] - 1.0) < 1e-6
    np.testing.assert_allclose(state_up[0], state[0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(state_up[-1], state[-1], rtol=0, atol=1e-3)


def test_noop_when_dts_match():
    state, jvel, grf, contact = _make_synthetic_reference(n_mpc=10, source_dt=0.02)
    state_up, jvel_up, grf_up, contact_up, meta = upsample_reference_arrays(
        state, jvel, grf, contact, source_dt=0.02, target_dt=0.02
    )
    assert not meta["upsampled"]
    assert meta["target_horizon"] == 10
    np.testing.assert_array_equal(state_up, state)
    np.testing.assert_array_equal(jvel_up, jvel)


def test_grf_zero_order_hold():
    state, jvel, grf, contact = _make_synthetic_reference(n_mpc=4, source_dt=0.05)
    grf[2, 2] = 999.0
    _, _, grf_up, _, meta = upsample_reference_arrays(
        state, jvel, grf, contact, source_dt=0.05, target_dt=0.02
    )
    assert meta["target_horizon"] == 10
    assert np.all(grf_up[0:2, 2] == grf[0, 2])
    assert np.all(grf_up[2:5, 2] == grf[1, 2])
    assert np.any(grf_up[:, 2] == 999.0)


if __name__ == "__main__":
    test_upsample_horizon_and_duration()
    test_noop_when_dts_match()
    test_grf_zero_order_hold()
    print("upsample_reference tests passed")
