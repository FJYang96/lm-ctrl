"""Phase-0.5 reference-quality inspector.

Loads an MPC iteration's {state_traj, grf_traj, joint_vel_traj,
contact_sequence}_iter_<N>.npy quartet and emits a reference_report.json
next to it. Reports the per-step properties that the data-driven Tier-A
promotion rule reads:

  * joint velocity / acceleration / jerk per joint vs hardware-limit margin
  * CoM linear & angular velocity through flight
  * attitude rate (especially at flip apex)
  * friction-cone usage per foot
  * FF-torque magnitude vs effort_limit (full inverse dynamics)
  * quaternion sign-continuity through the 180° apex
  * contact-schedule density per foot

Usage:
    python -m rl_isaac.inspect_reference --traj-dir <run_dir> --iter <N>
or
    python -m rl_isaac.inspect_reference \\
        --state-traj <path> --grf-traj <path> \\
        --joint-vel-traj <path> --contact-sequence <path> \\
        --output <reference_report.json>

FF-torque computation requires the mpc/gym_quadruped stack; if those imports
fail (e.g. running outside the lm-ctrl conda env), the script still emits
every other diagnostic and tags ff_torque section as "unavailable".
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# Set MUJOCO_GL so any downstream gym_quadruped import doesn't blow up.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")

# Repo-root import path for sibling packages (utils, mpc, go2_config).
import sys  # noqa: E402
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def _load_quartet(args) -> dict:
    if args.traj_dir:
        d = Path(args.traj_dir)
        n = args.iter
        paths = {
            "state": d / f"state_traj_iter_{n}.npy",
            "grf": d / f"grf_traj_iter_{n}.npy",
            "joint_vel": d / f"joint_vel_traj_iter_{n}.npy",
            "contact": d / f"contact_sequence_iter_{n}.npy",
        }
    else:
        paths = {
            "state": Path(args.state_traj),
            "grf": Path(args.grf_traj),
            "joint_vel": Path(args.joint_vel_traj),
            "contact": Path(args.contact_sequence) if args.contact_sequence else None,
        }
    out = {"paths": {k: (str(v) if v is not None else None) for k, v in paths.items()}}
    out["state_traj"] = np.load(paths["state"])
    out["grf_traj"] = np.load(paths["grf"])
    out["joint_vel_traj"] = np.load(paths["joint_vel"])
    out["contact_sequence"] = np.load(paths["contact"]) if paths["contact"] else None
    return out


def _euler_to_quat(eul: np.ndarray) -> np.ndarray:
    """Convert (N, 3) [roll, pitch, yaw] to (N, 4) [w, x, y, z]."""
    r, p, y = eul[:, 0], eul[:, 1], eul[:, 2]
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], axis=-1)


def _safe_max_per_joint(arr: np.ndarray) -> np.ndarray:
    return np.max(np.abs(arr), axis=0) if arr.size else np.zeros(arr.shape[1])


def _rms_per_joint(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(arr ** 2, axis=0)) if arr.size else np.zeros(arr.shape[1])


def _analyse_joints(joint_vel: np.ndarray, dt: float) -> dict:
    from go2_config import urdf_joint_velocities, joint_acceleration_limits
    vel = joint_vel  # (N, 12)
    accel = np.diff(vel, axis=0) / dt if vel.shape[0] > 1 else np.zeros((0, 12))
    jerk = np.diff(accel, axis=0) / dt if accel.shape[0] > 1 else np.zeros((0, 12))
    vel_limits = np.asarray(urdf_joint_velocities, dtype=float)
    accel_limits = np.asarray(joint_acceleration_limits, dtype=float)
    return {
        "n_frames": int(vel.shape[0]),
        "vel_max_per_joint": _safe_max_per_joint(vel).tolist(),
        "vel_limits": vel_limits.tolist(),
        "vel_margin_min": float(np.min(vel_limits - _safe_max_per_joint(vel))) if vel.size else None,
        "accel_max_per_joint": _safe_max_per_joint(accel).tolist(),
        "accel_limits": accel_limits.tolist(),
        "accel_margin_min": float(np.min(accel_limits - _safe_max_per_joint(accel))) if accel.size else None,
        "jerk_rms_per_joint": _rms_per_joint(jerk).tolist(),
        "jerk_peak_per_joint": _safe_max_per_joint(jerk).tolist(),
        "jerk_rms_overall": float(np.sqrt(np.mean(jerk ** 2))) if jerk.size else None,
        "jerk_peak_overall": float(np.max(np.abs(jerk))) if jerk.size else None,
    }


def _analyse_base(state_traj: np.ndarray, dt: float) -> dict:
    base_pos = state_traj[:, 0:3]
    base_lin_vel = state_traj[:, 3:6]
    base_eul = state_traj[:, 6:9]
    base_ang_vel = state_traj[:, 9:12]
    pitch = base_eul[:, 1]
    apex_idx = int(np.argmax(np.abs(pitch))) if pitch.size else None
    return {
        "com_z_max": float(np.max(base_pos[:, 2])) if base_pos.size else None,
        "com_z_min": float(np.min(base_pos[:, 2])) if base_pos.size else None,
        "com_z_gain": float(np.max(base_pos[:, 2]) - base_pos[0, 2]) if base_pos.size else None,
        "lin_vel_peak": float(np.max(np.linalg.norm(base_lin_vel, axis=-1))) if base_lin_vel.size else None,
        "lin_vel_z_peak": float(np.max(np.abs(base_lin_vel[:, 2]))) if base_lin_vel.size else None,
        "ang_vel_peak": float(np.max(np.linalg.norm(base_ang_vel, axis=-1))) if base_ang_vel.size else None,
        "ang_vel_y_peak": float(np.max(np.abs(base_ang_vel[:, 1]))) if base_ang_vel.size else None,
        "apex_frame": apex_idx,
        "apex_pitch_rad": float(pitch[apex_idx]) if apex_idx is not None else None,
        "apex_ang_vel": float(np.linalg.norm(base_ang_vel[apex_idx])) if apex_idx is not None else None,
        "terminal_lin_vel": base_lin_vel[-1].tolist() if base_lin_vel.size else None,
        "terminal_ang_vel": base_ang_vel[-1].tolist() if base_ang_vel.size else None,
        "total_pitch_rotation": float(pitch[-1] - pitch[0]) if base_eul.size else None,
    }


def _analyse_friction(grf_traj: np.ndarray, mu: float) -> dict:
    """grf_traj is (N, 12) = 4 feet × (Fx, Fy, Fz). Friction usage = |F_horiz| / (mu*Fz)."""
    n = grf_traj.shape[0]
    per_foot = []
    for foot in range(4):
        fx = grf_traj[:, foot * 3 + 0]
        fy = grf_traj[:, foot * 3 + 1]
        fz = grf_traj[:, foot * 3 + 2]
        contact = fz > 1.0
        f_horiz = np.sqrt(fx ** 2 + fy ** 2)
        denom = np.maximum(mu * np.maximum(fz, 1e-6), 1e-6)
        usage = np.where(contact, f_horiz / denom, 0.0)
        per_foot.append({
            "foot_idx": foot,
            "max_usage": float(np.max(usage)) if usage.size else None,
            "n_violations": int(np.sum(usage > 1.0)),
            "max_fz": float(np.max(fz)) if fz.size else None,
        })
    return {"mu_ground": mu, "per_foot": per_foot}


def _analyse_quat_continuity(state_traj: np.ndarray, apex_idx: int | None) -> dict:
    eul = state_traj[:, 6:9]
    quat = _euler_to_quat(eul)
    # Dot products between consecutive quaternions; sign flip iff dot < 0.
    dots = np.sum(quat[1:] * quat[:-1], axis=-1)
    flips = np.where(dots < 0.0)[0]
    near_apex = []
    if apex_idx is not None:
        for f in flips:
            if abs(int(f) - apex_idx) <= 5:
                near_apex.append(int(f))
    return {
        "n_sign_flips_total": int(len(flips)),
        "sign_flip_frames": [int(f) for f in flips.tolist()],
        "n_sign_flips_near_apex": len(near_apex),
        "sign_flip_frames_near_apex": near_apex,
    }


def _analyse_contacts(contact_sequence: np.ndarray | None, dt: float) -> dict:
    if contact_sequence is None:
        return {"available": False}
    cs = contact_sequence
    n_frames = cs.shape[1]
    per_foot = []
    foot_min_dwell = []
    for foot in range(4):
        in_contact = cs[foot, :] > 0.5
        density = float(np.mean(in_contact))
        # Run-length encode to find shortest contact / flight segment.
        runs = []
        if n_frames:
            cur_val = bool(in_contact[0])
            cur_len = 1
            for i in range(1, n_frames):
                v = bool(in_contact[i])
                if v == cur_val:
                    cur_len += 1
                else:
                    runs.append((cur_val, cur_len))
                    cur_val, cur_len = v, 1
            runs.append((cur_val, cur_len))
        n_transitions = max(0, len(runs) - 1)
        shortest_run = min((r[1] for r in runs), default=0)
        per_foot.append({
            "foot_idx": foot,
            "density": density,
            "n_transitions": n_transitions,
            "shortest_run_frames": shortest_run,
            "shortest_run_seconds": shortest_run * dt,
        })
        foot_min_dwell.append(shortest_run)
    return {
        "available": True,
        "n_frames": int(n_frames),
        "dt_seconds": dt,
        "per_foot": per_foot,
        "shortest_segment_any_foot_frames": int(min(foot_min_dwell)) if foot_min_dwell else None,
    }


def _analyse_ff_torques(state_traj, joint_vel_traj, grf_traj, dt) -> dict:
    """Full inverse dynamics FF torques. Optional — falls back if MPC stack
    isn't importable (e.g. running outside the conda env)."""
    try:
        from rl_isaac.feedforward import FeedforwardComputer
        from rl_isaac.reference import ReferenceTrajectory
        from mpc.dynamics.model import KinoDynamic_Model
        from go2_config import urdf_joint_efforts
    except Exception as e:
        return {"available": False, "error": f"{type(e).__name__}: {e}"}
    try:
        ref = ReferenceTrajectory(
            state_traj=state_traj,
            joint_vel_traj=joint_vel_traj,
            grf_traj=grf_traj,
            contact_sequence=None,
            control_dt=dt,
        )
        ff = FeedforwardComputer(KinoDynamic_Model()).precompute_trajectory(ref)
        eff = np.asarray(urdf_joint_efforts, dtype=float)
        abs_ff = np.abs(ff)
        sat = abs_ff / np.maximum(eff[None, :], 1e-6)  # (N, 12)
        return {
            "available": True,
            "max_per_joint": _safe_max_per_joint(ff).tolist(),
            "effort_limits": eff.tolist(),
            "saturation_max_per_joint": np.max(sat, axis=0).tolist(),
            "saturation_max_overall": float(np.max(sat)),
            "n_frames_over_20pct": int(np.sum(np.any(sat > 0.20, axis=-1))),
            "n_frames_over_50pct": int(np.sum(np.any(sat > 0.50, axis=-1))),
            "n_frames_over_90pct": int(np.sum(np.any(sat > 0.90, axis=-1))),
        }
    except Exception as e:
        return {"available": False, "error": f"{type(e).__name__}: {e}"}


def _promotion_check(report: dict) -> dict:
    """Apply the binding promotion rule from the plan against the report."""
    j = report["joints"]
    ff = report.get("ff_torques", {})
    contacts = report["contacts"]
    quat = report["quaternion"]
    triggers = []
    if j.get("jerk_rms_overall") and j["jerk_rms_overall"] > 50_000:
        triggers.append("jerk_rms_overall > 50000")
    # Per-joint peak jerk threshold (calibration note: tighten to 80k after
    # the first run if iter_12 reproduces ~99k peak jerk on joint 8).
    if j.get("jerk_peak_overall") and j["jerk_peak_overall"] > 80_000:
        triggers.append(f"jerk_peak_overall > 80000 (= {j['jerk_peak_overall']:.0f})")
    if ff.get("available") and ff.get("saturation_max_overall", 0.0) > 0.20:
        triggers.append(f"ff_torque_saturation_max > 0.20 (= {ff['saturation_max_overall']:.3f})")
    if contacts.get("available"):
        # 12 frames at dt = 240ms — matches CONTACT_GRACE_WINDOW.
        if contacts.get("shortest_segment_any_foot_frames", 999) < 12:
            triggers.append(
                f"contact segment shorter than grace window (= "
                f"{contacts['shortest_segment_any_foot_frames']} frames)"
            )
    if quat.get("n_sign_flips_near_apex", 0) > 0:
        triggers.append(
            f"quaternion sign flip near apex (frames {quat['sign_flip_frames_near_apex']})"
        )
    return {"triggers": triggers, "promote_constraint_fix": bool(triggers)}


def main():
    parser = argparse.ArgumentParser(
        description="Phase-0.5 MPC reference-quality inspector",
    )
    parser.add_argument("--traj-dir", type=str, default="",
                        help="Directory containing iter_<N>.npy quartet")
    parser.add_argument("--iter", type=int, default=12,
                        help="Iteration index when --traj-dir is used")
    parser.add_argument("--state-traj", type=str, default="")
    parser.add_argument("--grf-traj", type=str, default="")
    parser.add_argument("--joint-vel-traj", type=str, default="")
    parser.add_argument("--contact-sequence", type=str, default="")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="Control timestep (seconds)")
    parser.add_argument("--mu", type=float, default=None,
                        help="Friction coefficient (defaults to go2_config.mu_ground)")
    parser.add_argument("--output", type=str, default="",
                        help="Output JSON path (default: <traj-dir>/reference_report_iter_<N>.json)")
    args = parser.parse_args()

    if not args.traj_dir and not (args.state_traj and args.grf_traj and args.joint_vel_traj):
        parser.error("Provide --traj-dir or all of --state-traj/--grf-traj/--joint-vel-traj")

    data = _load_quartet(args)
    state, grf, jvel, contact = (
        data["state_traj"], data["grf_traj"], data["joint_vel_traj"], data["contact_sequence"],
    )

    # Friction coefficient: explicit override → go2_config → fallback 0.8.
    mu = args.mu
    if mu is None:
        try:
            from go2_config import experiment as _exp
            mu = float(_exp.mu_ground)
        except Exception:
            mu = 0.8

    base = _analyse_base(state, args.dt)
    report = {
        "source_paths": data["paths"],
        "shape_state": list(state.shape),
        "shape_grf": list(grf.shape),
        "shape_joint_vel": list(jvel.shape),
        "shape_contact": list(contact.shape) if contact is not None else None,
        "control_dt": args.dt,
        "joints": _analyse_joints(jvel, args.dt),
        "base": base,
        "friction": _analyse_friction(grf, mu),
        "quaternion": _analyse_quat_continuity(state, base.get("apex_frame")),
        "contacts": _analyse_contacts(contact, args.dt),
        "ff_torques": _analyse_ff_torques(state, jvel, grf, args.dt),
    }
    report["promotion_check"] = _promotion_check(report)

    if args.output:
        out_path = Path(args.output)
    elif args.traj_dir:
        out_path = Path(args.traj_dir) / f"reference_report_iter_{args.iter}.json"
    else:
        out_path = Path(args.state_traj).with_name("reference_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"Reference report written to {out_path}")
    if report["promotion_check"]["triggers"]:
        print("Promotion-rule triggers detected:")
        for t in report["promotion_check"]["triggers"]:
            print(f"  - {t}")
    else:
        print("No promotion-rule triggers detected.")


if __name__ == "__main__":
    main()
