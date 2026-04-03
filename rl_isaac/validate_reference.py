#!/usr/bin/env python3
"""Static validator for RL reference trajectories.

This script validates generated reference trajectories before they are used by
`rl_isaac.train` or `rl_isaac.evaluate`. It performs shape/integrity checks,
physics-oriented feasibility checks, and optionally saves debug plots.

Example:
    python rl_isaac/validate_reference.py \
        --traj-dir llm_integration/backflip --iter 20 --plot-dir results/validation
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import go2_config
from mpc.dynamics.model import KinoDynamic_Model
from rl_isaac.feedforward import FeedforwardComputer
from rl_isaac.reference import ReferenceTrajectory
from utils.conversion import (
    MPC_X_BASE_ANG,
    MPC_X_BASE_EUL,
    MPC_X_BASE_POS,
    MPC_X_BASE_VEL,
    MPC_X_Q_JOINTS,
)

FOOT_NAMES = ["FL", "FR", "RL", "RR"]
JOINT_NAMES = [
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
]

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


@dataclass
class CheckResult:
    name: str
    status: str
    details: list[str]


def _worst_status(statuses: list[str]) -> str:
    if FAIL in statuses:
        return FAIL
    if WARN in statuses:
        return WARN
    return PASS


def _fmt_bool(flag: bool) -> str:
    return "yes" if flag else "no"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate generated reference trajectories for RL preflight."
    )
    parser.add_argument(
        "--traj-dir",
        type=str,
        default="llm_integration/backflip",
        help="Directory containing *_traj_iter_*.npy files",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=20,
        help="Iteration number used in filename suffixes",
    )
    parser.add_argument(
        "--state-traj",
        type=str,
        default="",
        help="Optional explicit state trajectory path",
    )
    parser.add_argument(
        "--joint-vel-traj",
        type=str,
        default="",
        help="Optional explicit joint velocity trajectory path",
    )
    parser.add_argument(
        "--grf-traj",
        type=str,
        default="",
        help="Optional explicit GRF trajectory path",
    )
    parser.add_argument(
        "--contact-sequence",
        type=str,
        default="",
        help="Optional explicit contact sequence path",
    )
    parser.add_argument(
        "--control-dt",
        type=float,
        default=float(go2_config.mpc_config.mpc_dt),
        help="Control timestep in seconds",
    )
    parser.add_argument(
        "--grf-contact-threshold",
        type=float,
        default=1.0,
        help="Fz threshold for contact inference from GRF",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="",
        help="If provided, save debug plots to this directory",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code on FAIL",
    )
    return parser


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    traj_dir = Path(args.traj_dir)
    state_path = (
        Path(args.state_traj)
        if args.state_traj
        else traj_dir / f"state_traj_iter_{args.iter}.npy"
    )
    jvel_path = (
        Path(args.joint_vel_traj)
        if args.joint_vel_traj
        else traj_dir / f"joint_vel_traj_iter_{args.iter}.npy"
    )
    grf_path = (
        Path(args.grf_traj)
        if args.grf_traj
        else traj_dir / f"grf_traj_iter_{args.iter}.npy"
    )
    if args.contact_sequence:
        contact_path = Path(args.contact_sequence)
    else:
        candidate = traj_dir / f"contact_sequence_iter_{args.iter}.npy"
        contact_path = candidate if candidate.exists() else None
    return {
        "state": state_path,
        "joint_vel": jvel_path,
        "grf": grf_path,
        "contact": contact_path,
    }


def _check_files(paths: dict[str, Path | None]) -> CheckResult:
    details: list[str] = []
    missing: list[str] = []

    for key in ["state", "joint_vel", "grf"]:
        path = paths[key]
        assert path is not None
        if not path.exists():
            missing.append(str(path))
        else:
            details.append(f"found {key}: {path}")

    contact_path = paths["contact"]
    if contact_path is None:
        details.append("contact sequence: not provided (will derive from GRF)")
    elif contact_path.exists():
        details.append(f"found contact: {contact_path}")
    else:
        details.append(f"contact path provided but missing: {contact_path}")
        missing.append(str(contact_path))

    if missing:
        for item in missing:
            details.append(f"missing file: {item}")
        return CheckResult("files", FAIL, details)
    return CheckResult("files", PASS, details)


def _check_shapes_and_numeric(
    state_traj: np.ndarray,
    joint_vel_traj: np.ndarray,
    grf_traj: np.ndarray,
    contact_sequence: np.ndarray | None,
) -> CheckResult:
    details: list[str] = []
    statuses: list[str] = []

    N = joint_vel_traj.shape[0]
    details.append(f"N control steps: {N}")

    expected_state_shape = (N + 1, go2_config.STATES_DIM)
    expected_joint_vel_shape = (N, go2_config.N_JOINTS)
    expected_grf_shape = (N, 12)

    if state_traj.shape != expected_state_shape:
        statuses.append(FAIL)
        details.append(
            f"state shape mismatch: got {state_traj.shape}, expected {expected_state_shape}"
        )
    else:
        details.append(f"state shape OK: {state_traj.shape}")

    if joint_vel_traj.shape != expected_joint_vel_shape:
        statuses.append(FAIL)
        details.append(
            "joint velocity shape mismatch: "
            f"got {joint_vel_traj.shape}, expected {expected_joint_vel_shape}"
        )
    else:
        details.append(f"joint velocity shape OK: {joint_vel_traj.shape}")

    if grf_traj.shape != expected_grf_shape:
        statuses.append(FAIL)
        details.append(
            f"GRF shape mismatch: got {grf_traj.shape}, expected {expected_grf_shape}"
        )
    else:
        details.append(f"GRF shape OK: {grf_traj.shape}")

    if contact_sequence is not None:
        if contact_sequence.shape[0] != 4:
            statuses.append(FAIL)
            details.append(
                f"contact shape mismatch: first dim must be 4, got {contact_sequence.shape}"
            )
        elif contact_sequence.shape[1] < N:
            statuses.append(FAIL)
            details.append(
                "contact sequence too short: "
                f"got {contact_sequence.shape[1]}, expected >= {N}"
            )
        else:
            if contact_sequence.shape[1] > N:
                statuses.append(WARN)
                details.append(
                    "contact sequence has extra columns: "
                    f"got {contact_sequence.shape[1]}, using first {N}"
                )
            else:
                details.append(f"contact shape OK: {contact_sequence.shape}")

    for name, arr in [
        ("state", state_traj),
        ("joint_vel", joint_vel_traj),
        ("grf", grf_traj),
    ]:
        finite_mask = np.isfinite(arr)
        finite_ratio = float(np.mean(finite_mask))
        if finite_ratio < 1.0:
            statuses.append(FAIL)
            details.append(
                f"{name} has NaN/Inf values: finite_ratio={finite_ratio:.6f}"
            )
        else:
            details.append(f"{name} finite check OK")

    # Channel sanity summaries (helpful for debugging scale/frame errors)
    base_z = state_traj[:, 2]
    details.append(
        f"base_z range: [{base_z.min():.4f}, {base_z.max():.4f}]"
    )

    euler = state_traj[:, MPC_X_BASE_EUL]
    max_abs_euler = float(np.max(np.abs(euler)))
    details.append(f"max abs euler: {max_abs_euler:.4f} rad")
    if max_abs_euler > 2.0 * np.pi:
        statuses.append(WARN)
        details.append("orientation magnitude is unusually large (> 2*pi)")

    joint_pos = state_traj[:, MPC_X_Q_JOINTS]
    lower = go2_config.robot_data.joint_limits_lower
    upper = go2_config.robot_data.joint_limits_upper
    below = joint_pos < (lower - 1e-4)
    above = joint_pos > (upper + 1e-4)
    joint_limit_viol = int(np.count_nonzero(below | above))
    if joint_limit_viol > 0:
        statuses.append(FAIL)
        details.append(f"joint position limit violations: {joint_limit_viol}")
    else:
        details.append("joint position limits OK")

    joint_vel_limit = go2_config.robot_data.joint_velocity_limits
    vel_viol = int(np.count_nonzero(np.abs(joint_vel_traj) > (joint_vel_limit + 1e-4)))
    if vel_viol > 0:
        statuses.append(WARN)
        details.append(f"joint velocity limit exceedances: {vel_viol}")
    else:
        details.append("joint velocity limits OK")

    status = _worst_status(statuses) if statuses else PASS
    return CheckResult("shape_numeric", status, details)


def _compute_contact_expected(
    reference: ReferenceTrajectory,
    contact_sequence: np.ndarray | None,
    N: int,
) -> tuple[np.ndarray, str]:
    if contact_sequence is not None:
        return (contact_sequence[:, :N] > 0.5), "from_contact_file"
    return (reference.contact_sequence[:, :N] > 0.5), "derived_from_grf"


def _check_contact_and_grf(
    reference: ReferenceTrajectory,
    contact_sequence: np.ndarray | None,
    grf_traj: np.ndarray,
    threshold: float,
) -> tuple[CheckResult, dict[str, np.ndarray | float]]:
    details: list[str] = []
    statuses: list[str] = []

    N = grf_traj.shape[0]
    grf = grf_traj.reshape(N, 4, 3)
    fx = grf[:, :, 0]
    fy = grf[:, :, 1]
    fz = grf[:, :, 2]
    tangential = np.sqrt(fx**2 + fy**2)

    expected_contact, source = _compute_contact_expected(reference, contact_sequence, N)
    inferred_contact = fz > threshold

    details.append(f"contact source: {source}")

    if source == "from_contact_file":
        mismatch = expected_contact != inferred_contact
        mismatch_count = int(np.count_nonzero(mismatch))
        mismatch_ratio = mismatch_count / float(expected_contact.size)
        details.append(
            "contact mismatch (expected vs inferred): "
            f"{mismatch_count}/{expected_contact.size} ({100.0 * mismatch_ratio:.2f}%)"
        )
        if mismatch_ratio > 0.03:
            statuses.append(FAIL)
        elif mismatch_ratio > 0.0:
            statuses.append(WARN)

    force_norm = np.linalg.norm(grf, axis=2)
    phantom = (~expected_contact) & (force_norm > threshold)
    missing = expected_contact & (fz <= threshold)

    phantom_count = int(np.count_nonzero(phantom))
    missing_count = int(np.count_nonzero(missing))
    details.append(f"phantom forces during swing: {phantom_count}")
    details.append(f"missing support forces during stance: {missing_count}")

    if phantom_count > 0 or missing_count > 0:
        statuses.append(WARN)

    grf_limit = float(go2_config.grf_limits)
    comp_viol = int(np.count_nonzero(np.abs(grf_traj) > (grf_limit + 1e-6)))
    neg_fz = int(np.count_nonzero(fz < -1e-6))
    details.append(f"GRF component limit ({grf_limit:.1f} N) violations: {comp_viol}")
    details.append(f"negative normal-force samples: {neg_fz}")

    if comp_viol > 0 or neg_fz > 0:
        statuses.append(FAIL)

    mu = float(go2_config.experiment.mu_ground)
    # Only evaluate ratio when there is meaningful normal force.
    valid_fz = fz > threshold
    ratio = np.zeros_like(fz)
    ratio[valid_fz] = tangential[valid_fz] / (fz[valid_fz] + 1e-8)
    friction_viol = int(np.count_nonzero(valid_fz & (ratio > (mu + 1e-3))))
    details.append(f"friction ratio violations (mu={mu:.2f}): {friction_viol}")

    if friction_viol > 0:
        statuses.append(WARN)

    weight = float(go2_config.composite_mass * go2_config.experiment.gravity_constant)
    total_fz = np.sum(fz, axis=1)
    details.append(
        "total Fz / BW range: "
        f"[{(total_fz.min() / weight):.3f}, {(total_fz.max() / weight):.3f}]"
    )

    status = _worst_status(statuses) if statuses else PASS
    return (
        CheckResult("contact_grf", status, details),
        {
            "fz": fz,
            "ratio": ratio,
            "expected_contact": expected_contact.astype(np.float64),
            "inferred_contact": inferred_contact.astype(np.float64),
            "total_fz": total_fz,
            "weight": weight,
        },
    )


def _check_kinematics_continuity(
    state_traj: np.ndarray,
    joint_vel_traj: np.ndarray,
    dt: float,
) -> tuple[CheckResult, dict[str, np.ndarray]]:
    details: list[str] = []
    statuses: list[str] = []

    N = joint_vel_traj.shape[0]

    base_vel = np.hstack([state_traj[:N, MPC_X_BASE_VEL], state_traj[:N, MPC_X_BASE_ANG]])
    base_acc = np.zeros((N, 6))
    joint_acc = np.zeros((N, 12))

    if N > 1:
        base_acc[1:] = np.diff(base_vel, axis=0) / dt
        joint_acc[1:] = np.diff(joint_vel_traj, axis=0) / dt

    base_acc_norm = np.linalg.norm(base_acc, axis=1)
    joint_acc_norm = np.linalg.norm(joint_acc, axis=1)

    details.append(
        f"base acceleration norm p95/max: {np.percentile(base_acc_norm, 95):.2f}/{base_acc_norm.max():.2f}"
    )
    details.append(
        f"joint acceleration norm p95/max: {np.percentile(joint_acc_norm, 95):.2f}/{joint_acc_norm.max():.2f}"
    )

    # These are warning thresholds meant for debugging abrupt trajectory artifacts.
    if base_acc_norm.max() > 120.0:
        statuses.append(WARN)
        details.append("base acceleration is very high (>120 SI units)")
    if joint_acc_norm.max() > 400.0:
        statuses.append(WARN)
        details.append("joint acceleration is very high (>400 rad/s^2 norm)")

    base_jerk_norm = np.zeros_like(base_acc_norm)
    joint_jerk_norm = np.zeros_like(joint_acc_norm)
    if N > 2:
        base_jerk = np.diff(base_acc, axis=0) / dt
        joint_jerk = np.diff(joint_acc, axis=0) / dt
        base_jerk_norm[2:] = np.linalg.norm(base_jerk, axis=1)
        joint_jerk_norm[2:] = np.linalg.norm(joint_jerk, axis=1)

    details.append(
        f"base jerk norm p95/max: {np.percentile(base_jerk_norm, 95):.2f}/{base_jerk_norm.max():.2f}"
    )
    details.append(
        f"joint jerk norm p95/max: {np.percentile(joint_jerk_norm, 95):.2f}/{joint_jerk_norm.max():.2f}"
    )

    if joint_jerk_norm.max() > 10000.0:
        statuses.append(WARN)
        details.append("joint jerk spikes detected (>10000 rad/s^3 norm)")

    status = _worst_status(statuses) if statuses else PASS
    return (
        CheckResult("kinematics", status, details),
        {
            "base_acc_norm": base_acc_norm,
            "joint_acc_norm": joint_acc_norm,
            "base_jerk_norm": base_jerk_norm,
            "joint_jerk_norm": joint_jerk_norm,
            "joint_acc": joint_acc,
        },
    )


def _check_torque_feasibility(
    state_traj: np.ndarray,
    joint_vel_traj: np.ndarray,
    grf_traj: np.ndarray,
    dt: float,
) -> tuple[CheckResult, dict[str, np.ndarray]]:
    details: list[str] = []
    statuses: list[str] = []

    N = joint_vel_traj.shape[0]
    kindyn = KinoDynamic_Model()
    ff = FeedforwardComputer(kindyn)

    joint_acc = np.zeros((N, 12))
    if N > 1:
        joint_acc[1:] = np.diff(joint_vel_traj, axis=0) / dt

    abs_tau = np.zeros((N, 12))
    for k in range(N):
        tau_k = ff.compute(
            state_traj[k, MPC_X_BASE_POS],
            state_traj[k, MPC_X_BASE_EUL],
            state_traj[k, MPC_X_BASE_VEL],
            state_traj[k, MPC_X_BASE_ANG],
            state_traj[k, MPC_X_Q_JOINTS],
            joint_vel_traj[k],
            grf_traj[k],
            joint_acc[k],
        )
        abs_tau[k] = np.abs(tau_k)

    hardware_limits = np.asarray(go2_config.robot_data.joint_efforts, dtype=np.float64)
    planning_limits = 0.8 * hardware_limits

    hardware_viol = abs_tau > (hardware_limits + 1e-6)
    planning_viol = abs_tau > (planning_limits + 1e-6)

    n_hardware_viol = int(np.count_nonzero(hardware_viol))
    n_planning_viol = int(np.count_nonzero(planning_viol))

    details.append(f"hardware torque limit violations: {n_hardware_viol}")
    details.append(f"planning-safe torque violations (80%): {n_planning_viol}")

    peak_per_joint = np.max(abs_tau, axis=0)
    for j in range(12):
        details.append(
            f"{JOINT_NAMES[j]} peak={peak_per_joint[j]:.2f} "
            f"plan={planning_limits[j]:.2f} hard={hardware_limits[j]:.2f}"
        )

    if n_hardware_viol > 0:
        statuses.append(FAIL)
    elif n_planning_viol > 0:
        statuses.append(WARN)

    status = _worst_status(statuses) if statuses else PASS
    return (
        CheckResult("torque", status, details),
        {
            "abs_tau": abs_tau,
            "hardware_limits": hardware_limits,
            "planning_limits": planning_limits,
        },
    )


def _print_report(results: list[CheckResult], strict: bool) -> None:
    print("=" * 72)
    print("RL REFERENCE VALIDATION REPORT")
    print("=" * 72)
    for result in results:
        print(f"[{result.status}] {result.name}")
        for line in result.details:
            print(f"  - {line}")
        print()

    final_status = _worst_status([r.status for r in results])
    should_fail_exit = strict and final_status == FAIL
    print("=" * 72)
    print(f"OVERALL: {final_status}")
    print(f"strict_mode: {_fmt_bool(strict)}")
    print(f"nonzero_exit: {_fmt_bool(should_fail_exit)}")
    print("=" * 72)


def _save_plots(
    plot_dir: Path,
    dt: float,
    torque_metrics: dict[str, np.ndarray],
    contact_grf_metrics: dict[str, np.ndarray | float],
    kin_metrics: dict[str, np.ndarray],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency path
        print(f"[WARN] plotting skipped: matplotlib unavailable ({exc})")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    abs_tau = torque_metrics["abs_tau"]
    hardware_limits = torque_metrics["hardware_limits"]
    planning_limits = torque_metrics["planning_limits"]

    fz = contact_grf_metrics["fz"]
    ratio = contact_grf_metrics["ratio"]
    expected_contact = contact_grf_metrics["expected_contact"]
    inferred_contact = contact_grf_metrics["inferred_contact"]
    total_fz = contact_grf_metrics["total_fz"]
    weight = float(contact_grf_metrics["weight"])

    base_acc_norm = kin_metrics["base_acc_norm"]
    joint_acc_norm = kin_metrics["joint_acc_norm"]
    base_jerk_norm = kin_metrics["base_jerk_norm"]
    joint_jerk_norm = kin_metrics["joint_jerk_norm"]

    N = abs_tau.shape[0]
    t = np.arange(N) * dt

    # Torque overview
    fig, ax = plt.subplots(figsize=(12, 4))
    for j in range(12):
        ax.plot(t, abs_tau[:, j], lw=0.7, alpha=0.7)
    ax.plot(t, np.max(abs_tau, axis=1), color="black", lw=2.0, label="max |tau|")
    ax.axhline(float(np.max(hardware_limits)), color="red", ls="--", lw=1.5, label="max hard limit")
    ax.axhline(float(np.max(planning_limits)), color="orange", ls="--", lw=1.5, label="max plan limit")
    ax.set_title("Joint torque magnitude")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("|tau| [Nm]")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(plot_dir / "torque_overview.png", dpi=150)
    plt.close(fig)

    # GRF normal forces and total load
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for i, name in enumerate(FOOT_NAMES):
        ax[0].plot(t, fz[:, i], label=f"{name} Fz")
    ax[0].set_title("Per-foot normal GRF")
    ax[0].set_ylabel("Fz [N]")
    ax[0].grid(True, alpha=0.2)
    ax[0].legend(loc="upper right", ncol=4)

    ax[1].plot(t, total_fz, color="black", label="sum Fz")
    ax[1].axhline(weight, color="green", ls="--", lw=1.2, label="body weight")
    ax[1].set_title("Total normal force")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("sum Fz [N]")
    ax[1].grid(True, alpha=0.2)
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(plot_dir / "grf_overview.png", dpi=150)
    plt.close(fig)

    # Friction ratio
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, name in enumerate(FOOT_NAMES):
        ax.plot(t, ratio[:, i], label=f"{name} ||Ft||/Fz")
    ax.axhline(float(go2_config.experiment.mu_ground), color="red", ls="--", label="mu")
    ax.set_title("Friction ratio")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("||Ft|| / Fz")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", ncol=4)
    fig.tight_layout()
    fig.savefig(plot_dir / "friction_ratio.png", dpi=150)
    plt.close(fig)

    # Contact timeline
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    for i, name in enumerate(FOOT_NAMES):
        ax[0].step(t, expected_contact[:, i], where="post", label=name)
        ax[1].step(t, inferred_contact[:, i], where="post", label=name)
    ax[0].set_title("Expected contact")
    ax[1].set_title("Inferred contact from Fz")
    ax[0].set_ylabel("stance")
    ax[1].set_ylabel("stance")
    ax[1].set_xlabel("time [s]")
    ax[0].set_ylim(-0.1, 1.1)
    ax[1].set_ylim(-0.1, 1.1)
    ax[0].grid(True, alpha=0.2)
    ax[1].grid(True, alpha=0.2)
    ax[0].legend(loc="upper right", ncol=4)
    fig.tight_layout()
    fig.savefig(plot_dir / "contact_timeline.png", dpi=150)
    plt.close(fig)

    # Acceleration and jerk overview
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(t, base_acc_norm, label="base acc norm")
    ax[0].plot(t, joint_acc_norm, label="joint acc norm")
    ax[0].set_title("Acceleration norms")
    ax[0].set_ylabel("norm")
    ax[0].grid(True, alpha=0.2)
    ax[0].legend(loc="upper right")

    ax[1].plot(t, base_jerk_norm, label="base jerk norm")
    ax[1].plot(t, joint_jerk_norm, label="joint jerk norm")
    ax[1].set_title("Jerk norms")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("norm")
    ax[1].grid(True, alpha=0.2)
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(plot_dir / "kinematics_overview.png", dpi=150)
    plt.close(fig)

    print(f"[INFO] plots saved to: {plot_dir}")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    paths = _resolve_paths(args)
    file_result = _check_files(paths)
    if file_result.status == FAIL:
        _print_report([file_result], strict=args.strict)
        return 1 if args.strict else 0

    contact_path = paths["contact"]
    contact_sequence = np.load(contact_path) if contact_path is not None else None

    reference = ReferenceTrajectory.from_files(
        state_traj_path=str(paths["state"]),
        joint_vel_traj_path=str(paths["joint_vel"]),
        grf_traj_path=str(paths["grf"]),
        contact_sequence_path=str(contact_path) if contact_path is not None else None,
        control_dt=args.control_dt,
    )

    state_traj = reference.state_traj
    joint_vel_traj = reference.joint_vel_traj
    grf_traj = reference.grf_traj

    results: list[CheckResult] = [file_result]

    shape_result = _check_shapes_and_numeric(
        state_traj,
        joint_vel_traj,
        grf_traj,
        contact_sequence,
    )
    results.append(shape_result)

    # Heavy checks run only if basic shape checks passed.
    if shape_result.status != FAIL:
        contact_result, contact_metrics = _check_contact_and_grf(
            reference,
            contact_sequence,
            grf_traj,
            threshold=args.grf_contact_threshold,
        )
        kin_result, kin_metrics = _check_kinematics_continuity(
            state_traj,
            joint_vel_traj,
            args.control_dt,
        )

        results.extend([contact_result, kin_result])

        try:
            torque_result, torque_metrics = _check_torque_feasibility(
                state_traj,
                joint_vel_traj,
                grf_traj,
                args.control_dt,
            )
            results.append(torque_result)

            if args.plot_dir:
                _save_plots(
                    plot_dir=Path(args.plot_dir),
                    dt=args.control_dt,
                    torque_metrics=torque_metrics,
                    contact_grf_metrics=contact_metrics,
                    kin_metrics=kin_metrics,
                )
        except Exception as exc:
            results.append(
                CheckResult(
                    "torque",
                    FAIL,
                    [f"torque reconstruction failed: {type(exc).__name__}: {exc}"],
                )
            )

    _print_report(results, strict=args.strict)

    overall = _worst_status([r.status for r in results])
    if args.strict and overall == FAIL:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
