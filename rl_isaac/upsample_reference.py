"""Upsample MPC reference trajectories to the RL/MPPI control timestep.

MPC plans at coarse dt (e.g. 0.05 s / 20 Hz) while Isaac Lab control runs at
100 Hz by default (0.01 s). Joint positions and velocities are resampled via
cubic Hermite splines; base state uses linear / SLERP; GRF uses zero-order hold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from utils.conversion import (
    MPC_X_BASE_ANG,
    MPC_X_BASE_EUL,
    MPC_X_BASE_POS,
    MPC_X_BASE_VEL,
    MPC_X_INTEGRAL,
    MPC_X_Q_JOINTS,
    euler_to_quaternion,
    quaternion_to_euler,
)

from .reference import ReferenceTrajectory


def _linear_interp(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Interpolate ``y_src`` (T, D) from ``t_src`` onto ``t_dst``."""
    out = np.zeros((len(t_dst), y_src.shape[1]), dtype=np.float64)
    for col in range(y_src.shape[1]):
        out[:, col] = np.interp(t_dst, t_src, y_src[:, col])
    return out


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """SLERP between unit quaternions ``q0`` and ``q1`` at fraction ``t`` in [0, 1]."""
    q0 = _normalize_quat(q0.reshape(1, 4))[0]
    q1 = _normalize_quat(q1.reshape(1, 4))[0]
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return _normalize_quat(out.reshape(1, 4))[0]
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    w0 = np.sin((1.0 - t) * theta) / sin_theta
    w1 = np.sin(t * theta) / sin_theta
    return w0 * q0 + w1 * q1


def _interp_euler(
    t_src: np.ndarray, euler_src: np.ndarray, t_dst: np.ndarray
) -> np.ndarray:
    """Interpolate roll/pitch/yaw via quaternion SLERP."""
    quats_src = np.array(
        [euler_to_quaternion(euler_src[k]) for k in range(euler_src.shape[0])],
        dtype=np.float64,
    )
    out = np.zeros((len(t_dst), 3), dtype=np.float64)
    for i, t in enumerate(t_dst):
        if t <= t_src[0]:
            out[i] = euler_src[0]
            continue
        if t >= t_src[-1]:
            out[i] = euler_src[-1]
            continue
        j = int(np.searchsorted(t_src, t, side="right") - 1)
        j = min(j, len(t_src) - 2)
        local_t = (t - t_src[j]) / max(t_src[j + 1] - t_src[j], 1e-12)
        q = _slerp_quat(quats_src[j], quats_src[j + 1], float(local_t))
        out[i] = quaternion_to_euler(q)
    return out


def _zoh_index(t_dst: np.ndarray, source_dt: float, n_src: int) -> np.ndarray:
    """Map destination times to source interval indices (zero-order hold)."""
    idx = np.floor(t_dst / source_dt).astype(np.int64)
    return np.clip(idx, 0, n_src - 1)


def _joint_hermite_knots(
    state_traj: np.ndarray,
    joint_vel_traj: np.ndarray,
    n_mpc: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build joint position and tangent arrays at MPC state knot times."""
    q_knots = state_traj[:, MPC_X_Q_JOINTS].astype(np.float64)
    m_knots = np.zeros((n_mpc + 1, q_knots.shape[1]), dtype=np.float64)
    m_knots[:n_mpc] = joint_vel_traj
    m_knots[n_mpc] = joint_vel_traj[-1]
    return q_knots, m_knots


def _hermite_interp_joints(
    t_knots: np.ndarray,
    q_knots: np.ndarray,
    m_knots: np.ndarray,
    t_dst: np.ndarray,
    *,
    derivative_order: int = 0,
) -> np.ndarray:
    """Evaluate cubic Hermite splines for all joints at ``t_dst``."""
    n_joints = q_knots.shape[1]
    out = np.zeros((len(t_dst), n_joints), dtype=np.float64)
    for j in range(n_joints):
        spline = CubicHermiteSpline(t_knots, q_knots[:, j], m_knots[:, j])
        if derivative_order == 0:
            out[:, j] = spline(t_dst)
        else:
            out[:, j] = spline.derivative(derivative_order)(t_dst)
    return out


def upsample_reference_arrays(
    state_traj: np.ndarray,
    joint_vel_traj: np.ndarray,
    grf_traj: np.ndarray,
    contact_sequence: np.ndarray | None,
    source_dt: float,
    target_dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict]:
    """Resample MPC arrays from ``source_dt`` onto ``target_dt``.

    Returns:
        (state_traj, joint_vel_traj, grf_traj, contact_sequence, metadata)
    """
    if source_dt <= 0.0 or target_dt <= 0.0:
        raise ValueError("source_dt and target_dt must be positive.")
    n_mpc = joint_vel_traj.shape[0]
    if state_traj.shape[0] != n_mpc + 1:
        raise ValueError(
            f"state_traj must have length N+1 ({n_mpc + 1}), got {state_traj.shape[0]}"
        )
    if grf_traj.shape[0] != n_mpc:
        raise ValueError(
            f"grf_traj must have length N ({n_mpc}), got {grf_traj.shape[0]}"
        )

    duration = n_mpc * source_dt
    n_sim = max(1, int(round(duration / target_dt)))
    sim_duration = n_sim * target_dt

    metadata = {
        "source_dt": float(source_dt),
        "target_dt": float(target_dt),
        "source_horizon": int(n_mpc),
        "target_horizon": int(n_sim),
        "source_duration_s": float(duration),
        "target_duration_s": float(sim_duration),
        "upsampled": abs(source_dt - target_dt) > 1e-9,
        "joint_interp": "cubic_hermite",
    }

    if not metadata["upsampled"]:
        return (
            state_traj.copy(),
            joint_vel_traj.copy(),
            grf_traj.copy(),
            None if contact_sequence is None else contact_sequence.copy(),
            metadata,
        )

    t_src_state = np.linspace(0.0, duration, n_mpc + 1)
    t_dst_state = np.linspace(0.0, duration, n_sim + 1)
    t_dst_u = np.arange(n_sim, dtype=np.float64) * target_dt

    state_out = np.zeros((n_sim + 1, state_traj.shape[1]), dtype=np.float64)
    state_out[:, MPC_X_BASE_POS] = _linear_interp(
        t_src_state, state_traj[:, MPC_X_BASE_POS], t_dst_state
    )
    state_out[:, MPC_X_BASE_VEL] = _linear_interp(
        t_src_state, state_traj[:, MPC_X_BASE_VEL], t_dst_state
    )
    state_out[:, MPC_X_BASE_EUL] = _interp_euler(
        t_src_state, state_traj[:, MPC_X_BASE_EUL], t_dst_state
    )
    state_out[:, MPC_X_BASE_ANG] = _linear_interp(
        t_src_state, state_traj[:, MPC_X_BASE_ANG], t_dst_state
    )
    state_out[:, MPC_X_INTEGRAL] = _linear_interp(
        t_src_state, state_traj[:, MPC_X_INTEGRAL], t_dst_state
    )

    q_knots, m_knots = _joint_hermite_knots(state_traj, joint_vel_traj, n_mpc)
    state_out[:, MPC_X_Q_JOINTS] = _hermite_interp_joints(
        t_src_state, q_knots, m_knots, t_dst_state
    )
    joint_vel_out = _hermite_interp_joints(
        t_src_state, q_knots, m_knots, t_dst_u, derivative_order=1
    )

    zoh_idx = _zoh_index(t_dst_u, source_dt, n_mpc)
    grf_out = grf_traj[zoh_idx].copy()

    contact_out = None
    if contact_sequence is not None:
        if contact_sequence.shape[1] != n_mpc:
            raise ValueError(
                "contact_sequence must have shape (4, N) matching joint_vel_traj."
            )
        contact_out = contact_sequence[:, zoh_idx].copy()

    return state_out, joint_vel_out, grf_out, contact_out, metadata


def upsample_reference_trajectory(
    ref: ReferenceTrajectory,
    source_dt: float,
    target_dt: float,
) -> ReferenceTrajectory:
    """Return a new ``ReferenceTrajectory`` resampled to ``target_dt``."""
    state, jvel, grf, contact, _ = upsample_reference_arrays(
        ref.state_traj,
        ref.joint_vel_traj,
        ref.grf_traj,
        ref.contact_sequence,
        source_dt,
        target_dt,
    )
    return ReferenceTrajectory(
        state_traj=state,
        joint_vel_traj=jvel,
        grf_traj=grf,
        contact_sequence=contact,
        control_dt=target_dt,
    )


def save_reference_arrays(
    output_dir: Path,
    state_traj: np.ndarray,
    joint_vel_traj: np.ndarray,
    grf_traj: np.ndarray,
    contact_sequence: np.ndarray | None,
    metadata: dict,
    prefix: str = "upsampled",
) -> dict[str, str]:
    """Save resampled trajectory arrays and return path mapping."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "state_traj": str(output_dir / f"{prefix}_state_traj.npy"),
        "joint_vel_traj": str(output_dir / f"{prefix}_joint_vel_traj.npy"),
        "grf_traj": str(output_dir / f"{prefix}_grf_traj.npy"),
    }
    np.save(paths["state_traj"], state_traj)
    np.save(paths["joint_vel_traj"], joint_vel_traj)
    np.save(paths["grf_traj"], grf_traj)
    if contact_sequence is not None:
        paths["contact_sequence"] = str(
            output_dir / f"{prefix}_contact_sequence.npy"
        )
        np.save(paths["contact_sequence"], contact_sequence)
    meta_path = output_dir / f"{prefix}_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    paths["metadata"] = str(meta_path)
    return paths


def resolve_source_dt(
    explicit_dt: float | None,
    traj_dir: str | None,
) -> float:
    """Resolve MPC source dt from CLI override or project defaults."""
    if explicit_dt is not None and explicit_dt > 0.0:
        return float(explicit_dt)

    if traj_dir:
        meta_candidates = sorted(Path(traj_dir).glob("*metadata*.json"))
        for meta_path in meta_candidates:
            try:
                with meta_path.open(encoding="utf-8") as f:
                    meta = json.load(f)
                for key in ("source_dt", "mpc_dt", "time_step", "control_dt"):
                    if key in meta and float(meta[key]) > 0.0:
                        return float(meta[key])
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue

    import go2_config

    return float(go2_config.default_mpc_dt_complementarity)


def resolve_ref_control_dt(
    explicit_dt: float | None,
    ref_rate_hz: int = 100,
) -> float:
    """Resolve reference control dt from explicit override or ``ref_rate_hz``."""
    if explicit_dt is not None and explicit_dt > 0.0:
        return float(explicit_dt)

    import go2_config

    if ref_rate_hz == 200:
        return float(go2_config.high_ref_control_dt)
    if ref_rate_hz != 100:
        raise ValueError(f"Unsupported ref_rate_hz={ref_rate_hz}; use 100 or 200.")
    return float(go2_config.default_ref_control_dt)
