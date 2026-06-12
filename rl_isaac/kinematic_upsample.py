"""FK/IK-based upsampling helpers for MPC reference trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation

from utils.conversion import (
    MPC_X_BASE_ANG,
    MPC_X_BASE_EUL,
    MPC_X_BASE_POS,
    MPC_X_BASE_VEL,
    MPC_X_Q_JOINTS,
)

LEG_LABELS = ("fl", "fr", "rl", "rr")
N_LEGS = 4
JOINTS_PER_LEG = 3


@dataclass
class KinematicUpsampleParams:
    """Tuning parameters for FK/IK reference upsampling."""

    alpha: float = 1e-2
    beta: float = 1e-3
    ik_max_iters: int = 40
    ik_tol: float = 1e-8
    pinv_damping: float = 1e-4
    stance_anchor: str = "mean"
    contact_force_threshold: float = 1.0
    compute_feedforward: bool = True

    def as_metadata(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


def derive_contact_from_grf(
    grf_traj: np.ndarray,
    threshold: float = 1.0,
) -> np.ndarray:
    """Derive a (4, N) contact schedule from vertical GRFs."""
    contact = np.zeros((N_LEGS, grf_traj.shape[0]), dtype=np.float64)
    for foot in range(N_LEGS):
        contact[foot] = (grf_traj[:, foot * 3 + 2] > threshold).astype(np.float64)
    return contact


def contact_segments(contact_leg: np.ndarray) -> list[tuple[int, int, int]]:
    """Return interval segments as (start, end_exclusive, state)."""
    if contact_leg.ndim != 1:
        raise ValueError("contact_leg must be a 1D array.")
    if contact_leg.size == 0:
        return []
    binary = (contact_leg > 0.5).astype(np.int64)
    segments: list[tuple[int, int, int]] = []
    start = 0
    current = int(binary[0])
    for k in range(1, len(binary)):
        state = int(binary[k])
        if state != current:
            segments.append((start, k, current))
            start = k
            current = state
    segments.append((start, len(binary), current))
    return segments


def interpolate_stance_aware_grf(
    grf_traj: np.ndarray,
    contact_sequence: np.ndarray,
    t_dst_u: np.ndarray,
    source_dt: float,
) -> np.ndarray:
    """Interpolate GRFs linearly within stance and ZOH elsewhere."""
    n_mpc = grf_traj.shape[0]
    grf_out = np.zeros((len(t_dst_u), grf_traj.shape[1]), dtype=np.float64)
    if n_mpc == 0:
        return grf_out

    for i, t in enumerate(t_dst_u):
        src_idx = int(np.floor(t / source_dt))
        src_idx = int(np.clip(src_idx, 0, n_mpc - 1))
        if src_idx < n_mpc - 1:
            frac = float((t - src_idx * source_dt) / max(source_dt, 1e-12))
            frac = float(np.clip(frac, 0.0, 1.0))
        else:
            frac = 0.0

        for foot in range(N_LEGS):
            cols = slice(foot * 3, foot * 3 + 3)
            now_stance = contact_sequence[foot, src_idx] > 0.5
            next_stance = (
                src_idx < n_mpc - 1 and contact_sequence[foot, src_idx + 1] > 0.5
            )
            if now_stance and next_stance:
                grf_out[i, cols] = (
                    (1.0 - frac) * grf_traj[src_idx, cols]
                    + frac * grf_traj[src_idx + 1, cols]
                )
            else:
                grf_out[i, cols] = grf_traj[src_idx, cols]
    return grf_out


def compute_feedforward_torques(
    state_traj: np.ndarray,
    joint_vel_traj: np.ndarray,
    grf_traj: np.ndarray,
    target_dt: float,
    kindyn_model: Any,
) -> np.ndarray:
    """Compute dense full-ID feedforward torques with finite-difference qddot."""
    from .feedforward import FeedforwardComputer

    computer = FeedforwardComputer(kindyn_model)
    n_steps = joint_vel_traj.shape[0]
    ff = np.zeros((n_steps, 12), dtype=np.float64)
    for k in range(n_steps):
        if k > 0:
            q_ddot_j = (joint_vel_traj[k] - joint_vel_traj[k - 1]) / target_dt
        else:
            q_ddot_j = np.zeros(12, dtype=np.float64)
        ff[k] = computer.compute(
            state_traj[k, MPC_X_BASE_POS],
            state_traj[k, MPC_X_BASE_EUL],
            state_traj[k, MPC_X_BASE_VEL],
            state_traj[k, MPC_X_BASE_ANG],
            state_traj[k, MPC_X_Q_JOINTS],
            joint_vel_traj[k],
            grf_traj[k],
            q_ddot_j,
        )
    return ff


def kinematic_upsample_joints(
    state_traj: np.ndarray,
    joint_vel_traj: np.ndarray,
    grf_traj: np.ndarray,
    contact_sequence: np.ndarray | None,
    state_out: np.ndarray,
    q_interp_state: np.ndarray,
    q_interp_u: np.ndarray,
    t_src_state: np.ndarray,
    t_dst_state: np.ndarray,
    t_dst_u: np.ndarray,
    source_dt: float,
    target_dt: float,
    kindyn_model: Any,
    params: KinematicUpsampleParams | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict]:
    """Upsample joints/GRF/contact with FK/IK consistency."""
    params = params or KinematicUpsampleParams()
    if params.stance_anchor not in {"mean", "first"}:
        raise ValueError("stance_anchor must be 'mean' or 'first'.")
    if params.ik_max_iters < 1:
        raise ValueError("ik_max_iters must be >= 1.")

    n_mpc = joint_vel_traj.shape[0]
    contact = (
        derive_contact_from_grf(grf_traj, params.contact_force_threshold)
        if contact_sequence is None
        else contact_sequence.astype(np.float64, copy=True)
    )
    if contact.shape != (N_LEGS, n_mpc):
        raise ValueError(f"contact_sequence must have shape (4, {n_mpc}).")

    q_knots = state_traj[:, MPC_X_Q_JOINTS].astype(np.float64)
    qdot_knots = np.zeros((n_mpc + 1, 12), dtype=np.float64)
    qdot_knots[:n_mpc] = joint_vel_traj
    qdot_knots[n_mpc] = joint_vel_traj[-1]

    fk_knots, v_knots = _foot_kinematics_at_knots(
        kindyn_model, state_traj, q_knots, qdot_knots
    )
    task_state, task_vel_state = _build_foot_tasks(
        fk_knots, v_knots, contact, t_src_state, t_dst_state, params
    )
    task_u, task_vel_u = _build_foot_tasks(
        fk_knots, v_knots, contact, t_src_state, t_dst_u, params
    )

    q_state, ik_stats = _solve_dense_ik(
        kindyn_model, state_out, q_interp_state, task_state, params
    )
    q_u = q_state[: len(t_dst_u)]
    if len(q_u) != len(q_interp_u):
        q_u = q_interp_u.copy()
        q_u[: min(len(q_state), len(q_u))] = q_state[: min(len(q_state), len(q_u))]

    joint_vel_out = _solve_joint_velocities(
        kindyn_model, state_out[: len(t_dst_u)], q_u, task_vel_u, params
    )
    grf_out = interpolate_stance_aware_grf(grf_traj, contact, t_dst_u, source_dt)
    contact_out = _zoh_contact(contact, t_dst_u, source_dt)

    state_out = state_out.copy()
    state_out[:, MPC_X_Q_JOINTS] = q_state
    ff_torques = None
    if params.compute_feedforward:
        ff_torques = compute_feedforward_torques(
            state_out, joint_vel_out, grf_out, target_dt, kindyn_model
        )

    metadata = {
        "ik_mean_final_error": float(np.mean(ik_stats["final_errors"]))
        if ik_stats["final_errors"]
        else 0.0,
        "ik_max_final_error": float(np.max(ik_stats["final_errors"]))
        if ik_stats["final_errors"]
        else 0.0,
        "ik_mean_iters": float(np.mean(ik_stats["iterations"]))
        if ik_stats["iterations"]
        else 0.0,
    }
    return state_out, joint_vel_out, grf_out, contact_out, ff_torques, metadata


def _build_H(base_pos: np.ndarray, base_euler: np.ndarray) -> np.ndarray:
    H = np.eye(4, dtype=np.float64)
    H[:3, :3] = Rotation.from_euler("xyz", base_euler).as_matrix()
    H[:3, 3] = base_pos
    return H


def _as_vec(value: Any, size: int | None = None) -> np.ndarray:
    out = np.array(value, dtype=np.float64).reshape(-1)
    if size is not None:
        out = out[:size]
    return out


def _as_mat(value: Any) -> np.ndarray:
    return np.array(value, dtype=np.float64)


def _center_funs(kindyn_model: Any) -> list[Any]:
    return [
        kindyn_model.foot_center_position_fl_fun,
        kindyn_model.foot_center_position_fr_fun,
        kindyn_model.foot_center_position_rl_fun,
        kindyn_model.foot_center_position_rr_fun,
    ]


def _center_jac_funs(kindyn_model: Any) -> list[Any]:
    return [
        kindyn_model.foot_center_jacobian_fl_fun,
        kindyn_model.foot_center_jacobian_fr_fun,
        kindyn_model.foot_center_jacobian_rl_fun,
        kindyn_model.foot_center_jacobian_rr_fun,
    ]


def _foot_kinematics_at_knots(
    kindyn_model: Any,
    state_traj: np.ndarray,
    q_knots: np.ndarray,
    qdot_knots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_knots = q_knots.shape[0]
    fk = np.zeros((N_LEGS, n_knots, 3), dtype=np.float64)
    vel = np.zeros_like(fk)
    fk_funs = _center_funs(kindyn_model)
    jac_funs = _center_jac_funs(kindyn_model)
    for k in range(n_knots):
        H = _build_H(state_traj[k, MPC_X_BASE_POS], state_traj[k, MPC_X_BASE_EUL])
        base_vel = np.concatenate(
            [state_traj[k, MPC_X_BASE_VEL], state_traj[k, MPC_X_BASE_ANG]]
        )
        gen_vel = np.concatenate([base_vel, qdot_knots[k]])
        for foot in range(N_LEGS):
            fk[foot, k] = _as_vec(fk_funs[foot](H, q_knots[k]), 3)
            vel[foot, k] = _as_mat(jac_funs[foot](H, q_knots[k])) @ gen_vel
    return fk, vel


def _segment_anchor(
    fk_leg: np.ndarray,
    start: int,
    end: int,
    mode: str,
) -> np.ndarray:
    knot_idx = np.arange(start, end + 1, dtype=np.int64)
    knot_idx = np.clip(knot_idx, 0, fk_leg.shape[0] - 1)
    if mode == "first":
        return fk_leg[knot_idx[0]].copy()
    return np.mean(fk_leg[knot_idx], axis=0)


def _stance_anchor_map(
    fk_leg: np.ndarray,
    segments: list[tuple[int, int, int]],
    mode: str,
) -> dict[tuple[int, int], np.ndarray]:
    anchors: dict[tuple[int, int], np.ndarray] = {}
    for start, end, state in segments:
        if state == 1:
            anchors[(start, end)] = _segment_anchor(fk_leg, start, end, mode)
    return anchors


def _adjacent_stance_anchor(
    anchors: dict[tuple[int, int], np.ndarray],
    segments: list[tuple[int, int, int]],
    seg_idx: int,
    side: str,
) -> np.ndarray | None:
    other_idx = seg_idx - 1 if side == "prev" else seg_idx + 1
    if other_idx < 0 or other_idx >= len(segments):
        return None
    start, end, state = segments[other_idx]
    if state != 1:
        return None
    return anchors[(start, end)]


def _build_foot_tasks(
    fk_knots: np.ndarray,
    v_knots: np.ndarray,
    contact: np.ndarray,
    t_src_state: np.ndarray,
    t_eval: np.ndarray,
    params: KinematicUpsampleParams,
) -> tuple[np.ndarray, np.ndarray]:
    task_pos = np.zeros((len(t_eval), N_LEGS, 3), dtype=np.float64)
    task_vel = np.zeros_like(task_pos)
    if len(t_eval) == 0:
        return task_pos, task_vel

    for foot in range(N_LEGS):
        segments = contact_segments(contact[foot])
        anchors = _stance_anchor_map(fk_knots[foot], segments, params.stance_anchor)
        for seg_idx, (start, end, state) in enumerate(segments):
            t0 = t_src_state[start]
            t1 = t_src_state[end]
            is_last = seg_idx == len(segments) - 1
            mask = (t_eval >= t0 - 1e-12) & (
                t_eval <= t1 + 1e-12 if is_last else t_eval < t1 - 1e-12
            )
            if not np.any(mask):
                continue
            if state == 1:
                anchor = anchors[(start, end)]
                task_pos[mask, foot] = anchor
                task_vel[mask, foot] = 0.0
                continue

            knot_idx = np.arange(start, end + 1, dtype=np.int64)
            times = t_src_state[knot_idx]
            pos = fk_knots[foot, knot_idx].copy()
            vel = v_knots[foot, knot_idx].copy()
            prev_anchor = _adjacent_stance_anchor(anchors, segments, seg_idx, "prev")
            next_anchor = _adjacent_stance_anchor(anchors, segments, seg_idx, "next")
            if prev_anchor is not None:
                pos[0] = prev_anchor
                vel[0] = 0.0
            if next_anchor is not None:
                pos[-1] = next_anchor
                vel[-1] = 0.0
            if len(times) == 1:
                task_pos[mask, foot] = pos[0]
                task_vel[mask, foot] = vel[0]
                continue
            for axis in range(3):
                spline = CubicHermiteSpline(times, pos[:, axis], vel[:, axis])
                task_pos[mask, foot, axis] = spline(t_eval[mask])
                task_vel[mask, foot, axis] = spline.derivative()(t_eval[mask])
    return task_pos, task_vel


def _solve_dense_ik(
    kindyn_model: Any,
    state_out: np.ndarray,
    q_interp_state: np.ndarray,
    task_pos: np.ndarray,
    params: KinematicUpsampleParams,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    n_states = state_out.shape[0]
    q_out = np.zeros((n_states, 12), dtype=np.float64)
    q_prev = q_interp_state[0].copy()
    fk_funs = _center_funs(kindyn_model)
    jac_funs = _center_jac_funs(kindyn_model)
    stats: dict[str, list[float]] = {"final_errors": [], "iterations": []}

    for k in range(n_states):
        q_solution = q_interp_state[k].copy()
        H = _build_H(state_out[k, MPC_X_BASE_POS], state_out[k, MPC_X_BASE_EUL])
        for foot in range(N_LEGS):
            leg_slice = slice(foot * JOINTS_PER_LEG, (foot + 1) * JOINTS_PER_LEG)
            q_leg = q_prev[leg_slice].copy()
            q_ref = q_interp_state[k, leg_slice]
            last_err = 0.0
            iterations = 0
            for iterations in range(1, params.ik_max_iters + 1):
                q_full = q_solution.copy()
                q_full[leg_slice] = q_leg
                p = _as_vec(fk_funs[foot](H, q_full), 3)
                J = _as_mat(jac_funs[foot](H, q_full))[:, 6 + foot * 3 : 9 + foot * 3]
                residual = task_pos[k, foot] - p
                lhs = J.T @ J + (params.alpha + params.beta) * np.eye(JOINTS_PER_LEG)
                rhs = J.T @ residual + params.alpha * (
                    q_prev[leg_slice] - q_leg
                ) + params.beta * (q_ref - q_leg)
                dq = np.linalg.solve(lhs, rhs)
                q_leg = q_leg + dq
                last_err = float(np.linalg.norm(residual))
                if np.linalg.norm(dq) < params.ik_tol:
                    break
            q_solution[leg_slice] = q_leg
            stats["final_errors"].append(last_err)
            stats["iterations"].append(float(iterations))
        q_out[k] = q_solution
        q_prev = q_solution
    return q_out, stats


def _solve_joint_velocities(
    kindyn_model: Any,
    state_u: np.ndarray,
    q_u: np.ndarray,
    task_vel_u: np.ndarray,
    params: KinematicUpsampleParams,
) -> np.ndarray:
    n_steps = q_u.shape[0]
    qdot_out = np.zeros((n_steps, 12), dtype=np.float64)
    jac_funs = _center_jac_funs(kindyn_model)
    eye3 = np.eye(JOINTS_PER_LEG)
    damping2 = params.pinv_damping**2
    for k in range(n_steps):
        H = _build_H(state_u[k, MPC_X_BASE_POS], state_u[k, MPC_X_BASE_EUL])
        base_vel = np.concatenate([state_u[k, MPC_X_BASE_VEL], state_u[k, MPC_X_BASE_ANG]])
        for foot in range(N_LEGS):
            leg_slice = slice(foot * JOINTS_PER_LEG, (foot + 1) * JOINTS_PER_LEG)
            J_full = _as_mat(jac_funs[foot](H, q_u[k]))
            J_base = J_full[:, :6]
            J_leg = J_full[:, 6 + foot * 3 : 9 + foot * 3]
            desired_rel_vel = task_vel_u[k, foot] - J_base @ base_vel
            lhs = J_leg.T @ J_leg + damping2 * eye3
            rhs = J_leg.T @ desired_rel_vel
            qdot_out[k, leg_slice] = np.linalg.solve(lhs, rhs)
    return qdot_out


def _zoh_contact(
    contact: np.ndarray,
    t_dst_u: np.ndarray,
    source_dt: float,
) -> np.ndarray:
    idx = np.floor(t_dst_u / source_dt).astype(np.int64)
    idx = np.clip(idx, 0, contact.shape[1] - 1)
    return contact[:, idx].copy()
