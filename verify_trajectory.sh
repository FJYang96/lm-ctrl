#!/bin/bash
# Verify MPC trajectory is physically feasible for Go2.
# Checks Yicong's concerns: torque limits, joint accelerations, GRF feasibility.
#
# Usage: ./verify_trajectory.sh [traj_dir] [iter_num]
# Or edit the defaults below:

set -e

# ── Edit these defaults for your trajectory ──
DEFAULT_TRAJ_DIR="results/llm_iterations/good_backflip"
DEFAULT_ITER=7

TRAJ_DIR="${1:-$DEFAULT_TRAJ_DIR}"
ITER="${2:-$DEFAULT_ITER}"
PYTHON="${PYTHON:-/home/aryanroy/miniconda3/bin/python3}"

STATE="$TRAJ_DIR/state_traj_iter_${ITER}.npy"
GRF="$TRAJ_DIR/grf_traj_iter_${ITER}.npy"
JVEL="$TRAJ_DIR/joint_vel_traj_iter_${ITER}.npy"

for f in "$STATE" "$GRF" "$JVEL"; do
    [ -f "$f" ] || { echo "Missing: $f"; exit 1; }
done

cd "$(dirname "$0")"

$PYTHON - "$STATE" "$GRF" "$JVEL" "$TRAJ_DIR" "$ITER" <<'PYEOF'
"""Trajectory diagnostic — prints key metrics, no frills."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np

BODY_WEIGHT = 15.019 * 9.81
MU          = 0.8
DT          = 0.02
KD          = 1.5
TORQUE_LIMS = np.array([23.7, 23.7, 45.43] * 4)
JVEL_LIMIT  = 30.0
VEL_CAP     = 15.70
JVEL_CAP_PER = np.array([30.1, 30.1, 15.7] * 4)
JPOS_LO = np.array([-1.047,-1.571,-2.723,-1.047,-1.571,-2.723,-1.047,-0.524,-2.723,-1.047,-0.524,-2.723])
JPOS_HI = np.array([ 1.047, 3.491,-0.838, 1.047, 3.491,-0.838, 1.047, 4.538,-0.838, 1.047, 4.538,-0.838])
JNAMES = ["FL_hip","FL_thigh","FL_calf","FR_hip","FR_thigh","FR_calf",
          "RL_hip","RL_thigh","RL_calf","RR_hip","RR_thigh","RR_calf"]


def l_world_analysis(state, grf, jvel, contact, lo, touchdown):
    """World-frame centroidal angular momentum in pure flight."""
    if lo is None or touchdown is None or touchdown <= lo:
        return None
    try:
        sys.path.insert(0, str(Path.cwd()))
        import casadi as cs
        from mpc.dynamics.model import KinoDynamic_Model
    except ImportError:
        return None

    model = KinoDynamic_Model()
    x_sym = cs.SX.sym("x", 30); u_sym = cs.SX.sym("u", 24)
    roll, pitch, yaw = x_sym[6], x_sym[7], x_sym[8]
    cR,sR = cs.cos(roll), cs.sin(roll); cP,sP = cs.cos(pitch), cs.sin(pitch); cY,sY = cs.cos(yaw), cs.sin(yaw)
    Rx = cs.vertcat(cs.horzcat(1,0,0), cs.horzcat(0,cR,-sR), cs.horzcat(0,sR,cR))
    Ry = cs.vertcat(cs.horzcat(cP,0,sP), cs.horzcat(0,1,0), cs.horzcat(-sP,0,cP))
    Rz = cs.vertcat(cs.horzcat(cY,-sY,0), cs.horzcat(sY,cY,0), cs.horzcat(0,0,1))
    H = cs.SX.eye(4); H[0:3,0:3] = Rz@Ry@Rx; H[0:3,3] = x_sym[0:3]
    A_G = model.centroidal_momentum_matrix_fun(H, x_sym[12:24])
    v_gen = cs.vertcat(x_sym[3:6], x_sym[9:12], u_sym[0:12])
    f_h = cs.Function("hfun", [x_sym, u_sym], [A_G @ v_gen])

    N = grf.shape[0]
    L = np.zeros((N, 3))
    for k in range(N):
        L[k] = np.array(f_h(state[k], np.concatenate([jvel[k], grf[k]]))).ravel()[3:6]
    L_flight = L[lo:touchdown]
    if len(L_flight) < 2:
        return None
    dL_per_step = np.diff(L_flight, axis=0)
    return {
        "L_first": L_flight[0],
        "L_last": L_flight[-1],
        "max_dL_per_axis": np.max(np.abs(dL_per_step), axis=0),
        "total_drift": L_flight[-1] - L_flight[0],
        "tol": 0.5,
    }


def ff_torque_analysis(sf, gf, jf, cf):
    """Full inverse-dynamics FF torques for entire trajectory. Requires pympc docker."""
    try:
        sys.path.insert(0, str(Path.cwd()))
        import rl_isaac._utils_fix  # noqa: F401
        from mpc.dynamics.model import KinoDynamic_Model
        from rl_isaac.feedforward import FeedforwardComputer
        from rl_isaac.reference import ReferenceTrajectory
    except ImportError:
        return None

    ref = ReferenceTrajectory.from_files(
        str(sf), str(jf), str(gf),
        str(cf) if cf else None,
    )
    return FeedforwardComputer(KinoDynamic_Model()).precompute_trajectory(ref)


def fk_analysis(state, grf, jvel, contact, liftoff, landing):
    """FK-based analysis: foot velocities, MDP violations, angular torque from GRF."""
    try:
        from scipy.spatial.transform import Rotation
        sys.path.insert(0, str(Path.cwd()))
        from mpc.dynamics.model import KinoDynamic_Model
    except ImportError:
        return None

    model = KinoDynamic_Model()
    foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    jac_funs = [model.kindyn.jacobian_fun(f) for f in foot_names]
    fk_funs  = [model.kindyn.forward_kinematics_fun(f) for f in foot_names]

    N = jvel.shape[0]
    foot_vt  = np.zeros((N, 4, 2))
    foot_pos = np.zeros((N, 4, 3))

    for k in range(N):
        euler = state[k, 6:9]
        R = Rotation.from_euler('xyz', euler).as_matrix()
        H = np.eye(4); H[:3,:3] = R; H[:3,3] = state[k,0:3]
        v_gen = np.concatenate([state[k,3:6], state[k,9:12], jvel[k]])
        jpos  = state[k, 12:24]
        for fi in range(4):
            J = np.array(jac_funs[fi](H, jpos))[:3, :]
            foot_vt[k, fi] = (J @ v_gen)[:2]
            foot_pos[k, fi] = np.array(fk_funs[fi](H, jpos))[:3, 3]

    mdp_viols = []
    stance_end = liftoff if liftoff else N
    for k in range(stance_end):
        for fi in range(4):
            if contact is not None and contact[fi, k] < 0.5: continue
            fz = grf[k, fi*3+2]
            if fz < 5: continue
            ft = grf[k, fi*3:fi*3+2]
            vt = foot_vt[k, fi]
            if np.linalg.norm(vt) < 0.01: continue
            dot = float(np.dot(ft, vt))
            if dot > 0:
                mdp_viols.append((k, fi, dot, float(np.linalg.norm(ft)), float(np.linalg.norm(vt))))

    euler = state[:, 6:9]
    ax = int(np.abs(euler[-1] - euler[0]).argmax())

    tau_rot = np.zeros(N)
    for k in range(N):
        for fi in range(4):
            px, py, pz = foot_pos[k, fi]
            fx, fy, fz_ = grf[k, fi*3], grf[k, fi*3+1], grf[k, fi*3+2]
            if ax == 2:
                tau_rot[k] += px * fy - py * fx
            elif ax == 1:
                tau_rot[k] += pz * fx - px * fz_
            else:
                tau_rot[k] += py * fz_ - pz * fy

    return dict(foot_vt=foot_vt, foot_pos=foot_pos, mdp_viols=mdp_viols,
                tau_rot=tau_rot, ax=ax)


def detect_phases(grf, contact):
    """Return (liftoff, landing, touchdown).

    liftoff   = first step of pure flight (all 4 feet off)
    touchdown = first step after liftoff where ANY foot is back in contact
    landing   = first step after liftoff where ALL 4 feet are in contact
    """
    N = grf.shape[0]
    fz = grf.reshape(N,4,3)[:,:,2]
    lo = touchdown = land = None
    if contact is not None:
        all_off = np.all(contact < 0.5, axis=0)
        any_on  = np.any(contact > 0.5, axis=0)
        all_on  = np.all(contact > 0.5, axis=0)
        for k in range(N):
            if all_off[k] and lo is None: lo = k
            if lo is not None and k > lo and any_on[k] and touchdown is None: touchdown = k
            if lo is not None and k > lo and all_on[k] and land is None: land = k
    else:
        fzt = fz.sum(1)
        for k in range(N):
            if fzt[k] < 5 and lo is None: lo = k
            if lo is not None and k > lo and fzt[k] > 5 and touchdown is None: touchdown = k
            if lo is not None and k > lo and fzt[k] > 50 and land is None: land = k
    return lo, land, touchdown


def run(traj_dir, sf, gf, jf, cf=None, iter_num=None):
    state   = np.load(sf)
    grf     = np.load(gf)
    jvel    = np.load(jf)
    contact = np.load(cf) if cf else None
    N = grf.shape[0]
    lo, land, touchdown = detect_phases(grf, contact)

    fz  = grf.reshape(N,4,3)[:,:,2]
    ft  = grf.reshape(N,4,3)[:,:,:2]
    euler = state[:, 6:9]
    omega = state[:, 9:12]

    label = traj_dir.name + (f"  iter {iter_num}" if iter_num else "")
    print(f"\n{'─'*60}")
    print(f"  {label}")
    flight_steps = (touchdown-lo) if (lo is not None and touchdown is not None) else '?'
    print(f"  N={N}  liftoff={lo}  touchdown={touchdown}  landing={land}  pure_flight={flight_steps} steps")
    print(f"{'─'*60}")

    # ── Initial conditions ────────────────────────────────────────
    fz0 = fz[0].sum()
    jv0_max = np.abs(jvel[0]).max()
    alpha0 = np.abs(omega[1] - omega[0]).max() / DT
    com_vel0_norm = np.linalg.norm(state[0, 3:6])
    omega0_norm   = np.linalg.norm(state[0, 9:12])
    print(f"\n  [initial]")
    print(f"    step-0 fz     = {fz0:.0f} N  ({fz0/BODY_WEIGHT*100:.0f}% body weight)")
    print(f"    step-0 jvel   = {jv0_max:.2f} rad/s  ({JNAMES[np.abs(jvel[0]).argmax()]})")
    print(f"    step 0→1 α    = {alpha0:.0f} rad/s²")
    if com_vel0_norm < 0.05 and omega0_norm < 0.05 and jv0_max > 1.0:
        print(f"    *** initial inconsistency: com_vel≈0 but jvel={jv0_max:.2f} r/s")
        print(f"        joints already moving at t=0 → PhysX contact unstable")
        print(f"        per-foot fz: FL={fz[0,0]:.0f} FR={fz[0,1]:.0f} RL={fz[0,2]:.0f} RR={fz[0,3]:.0f} N  (static eq = {BODY_WEIGHT/4:.0f} N each)")

    # ── Kinematics ───────────────────────────────────────────────
    jpos_viol = [(k,j,state[k,12+j]) for k in range(N+1) for j in range(12)
                 if state[k,12+j] < JPOS_LO[j]-0.05 or state[k,12+j] > JPOS_HI[j]+0.05]
    jvel_peaks = [(JNAMES[j], np.abs(jvel[:,j]).max()) for j in range(12)]
    cap_hits   = {JNAMES[j]: int((np.abs(jvel[:,j]) >= JVEL_CAP_PER[j]-0.01).sum())
                  for j in range(12) if (np.abs(jvel[:,j]) >= JVEL_CAP_PER[j]-0.01).any()}
    qddot = np.diff(jvel, axis=0) / DT
    qddot_peak_idx = np.unravel_index(np.argmax(np.abs(qddot)), qddot.shape)
    rot_deg = np.degrees(np.abs(euler[-1] - euler[0]))
    print(f"\n  [kinematics]")
    print(f"    jpos violations = {len(jpos_viol)}")
    print(f"    jvel peak       = {max(v for _,v in jvel_peaks):.2f} rad/s  "
          f"({max(jvel_peaks,key=lambda x:x[1])[0]})  URDF limits hip/thigh=30.1 calf=15.7")
    print(f"    vel cap hits    = {dict(cap_hits) if cap_hits else 'none'}  (per-joint URDF cap)")
    print(f"    q̈ peak         = {np.abs(qddot).max():.0f} rad/s²  ({JNAMES[qddot_peak_idx[1]]} at step {qddot_peak_idx[0]+1})")
    print(f"    total rotation  = roll {rot_deg[0]:.0f}°  pitch {rot_deg[1]:.0f}°  yaw {rot_deg[2]:.0f}°")

    # ── Contact transitions ──────────────────────────────────────
    print(f"\n  [transitions]")
    for t, lbl in [(lo,'liftoff'), (land,'landing')]:
        if t is None or t == 0 or t >= N: continue
        dv = np.abs(jvel[t] - jvel[t-1])
        sat = [(JNAMES[j], KD*dv[j], TORQUE_LIMS[j]) for j in range(12) if KD*dv[j] > TORQUE_LIMS[j]]
        print(f"    {lbl:8s}: max Δdq={dv.max():.1f} r/s ({JNAMES[dv.argmax()]})  "
              f"KD_sat={len(sat)} joints")
        for nm,kd,lim in sat:
            print(f"               {nm}: {kd:.0f} Nm > {lim:.0f} Nm")
    if lo is not None and lo < state.shape[0]-1:
        dw = np.abs(omega[lo] - omega[lo-1]).max()
        print(f"    omega jump at liftoff: {dw:.3f} rad/s")

    # ── GRF quality ──────────────────────────────────────────────
    stance_end = lo if lo else N
    util_all = []
    for k in range(stance_end):
        for foot in range(4):
            if contact is not None and contact[foot,k] < 0.5: continue
            z = fz[k,foot]
            if z < 5: continue
            util_all.append((k, foot, np.linalg.norm(ft[k,foot]) / (MU*z)))
    util_all.sort(key=lambda x:-x[2])

    print(f"\n  [grf]")
    if lo: print(f"    stance avg fz   = {fz[:lo].sum(1).mean():.0f} N")
    print(f"    friction util   = max {util_all[0][2]*100:.0f}% at step {util_all[0][0]} foot {util_all[0][1]}"
          if util_all else "    friction util   = n/a")
    print(f"    top 5 util steps: {[(k,f,f'{u*100:.0f}%') for k,f,u in util_all[:5]]}")
    if land is not None:
        win = fz[max(0,land-1):min(N,land+3)].sum(1)
        print(f"    landing impact  = {win.max()/BODY_WEIGHT:.2f}× body weight  (peak {win.max():.0f} N)")

    # ── Angular momentum ─────────────────────────────────────────
    ax = int(np.abs(euler[-1]-euler[0]).argmax())
    ax_name = ['roll','pitch','yaw'][ax]
    print(f"\n  [angular momentum]  axis={ax_name}")
    if lo:
        om_lo = omega[lo, ax]
        windup = [k for k in range(lo) if np.sign(omega[k,ax])==-np.sign(om_lo) and abs(omega[k,ax])>0.3]
        print(f"    liftoff omega   = {om_lo:.3f} rad/s")
        print(f"    wind-up steps   = {windup}  (peak {max((abs(omega[k,ax]) for k in windup),default=0):.2f} rad/s)")
        flight_end = touchdown if touchdown is not None else land
        if flight_end is not None:
            rot_flight = euler[flight_end, ax] - euler[lo, ax]
            span = flight_end - lo
            avg_needed = rot_flight / (span * DT) if span > 0 else float('nan')
            print(f"    flight rotation = {np.degrees(rot_flight):.1f}°  avg_omega_needed = {avg_needed:.2f} r/s  (over {span} pure-flight steps)")
            drift = omega[flight_end-1, ax] - omega[lo, ax]
            print(f"    flight drift    = {drift:+.3f} rad/s  (body-frame ω{['x','y','z'][ax]}, pure flight only)")
            print(f"    note: body-frame ω is NOT a conserved quantity — Euler's equation allows ω drift even in torque-free flight;")
            print(f"          use world-frame centroidal L to check conservation (MPC enforces |ΔL|≤0.5 per axis)")
        alpha = np.diff(omega[:lo+1, ax]) / DT
        print(f"    thrust α        = {np.abs(alpha).max():.0f} rad/s² (max during stance)")

        lw = l_world_analysis(state, grf, jvel, contact, lo, touchdown)
        if lw is not None:
            print(f"\n  [world-frame L_centroidal, pure flight]")
            print(f"    L[first flight] = {np.round(lw['L_first'], 3)}")
            print(f"    L[last  flight] = {np.round(lw['L_last'], 3)}")
            print(f"    max |ΔL_k|      = {np.round(lw['max_dL_per_axis'], 3)}  (MPC tol ±{lw['tol']} per axis)")
            print(f"    total drift     = {np.round(lw['total_drift'], 3)}")
            viol = (lw['max_dL_per_axis'] > lw['tol']).any()
            print(f"    conservation    = {'VIOLATED' if viol else 'OK'}")

    # ── FF torque feasibility ───────────────────────────────────
    ff = ff_torque_analysis(sf, gf, jf, cf)
    if ff is not None:
        sat_mask = np.abs(ff) > TORQUE_LIMS[None, :]
        sat_steps = np.where(sat_mask.any(axis=1))[0]
        print(f"\n  [ff torques]  full inverse dynamics")
        print(f"    saturated steps: {len(sat_steps)} / {N}")
        for k in sat_steps:
            bad = [(JNAMES[j], ff[k,j], TORQUE_LIMS[j]) for j in range(12) if sat_mask[k,j]]
            parts = "  ".join(f"{nm}={val:+.0f}Nm(lim={lim:.0f})" for nm,val,lim in bad)
            print(f"      step {k:2d}: {parts}")
        if len(sat_steps) == 0:
            print(f"    peak FF  = {np.abs(ff).max():.1f} Nm  ({JNAMES[np.abs(ff).argmax() % 12]})")

    # ── PD+FF analysis ──────────────────────────────────────────
    if ff is not None:
        KP = 25.0
        jpos = state[:N, 12:24]

        ff_max_per_step    = np.abs(ff).max(axis=1)
        headroom_per_step  = (TORQUE_LIMS[None, :] - np.abs(ff)).min(axis=1)

        print(f"\n  [pd+ff]")
        print(f"    {'step':>4}  {'FF_max':>7}  {'headroom':>9}  contact")
        for k in range(N):
            if contact is not None:
                phase = "flight" if np.all(contact[:, k] < 0.5) else "stance"
            else:
                phase = "flight" if (lo is not None and land is not None and lo <= k < land) else "stance"
            note = ""
            if k == lo:   note = " TAKEOFF"
            elif k == land: note = " LANDING"
            print(f"    {k:4d}  {ff_max_per_step[k]:7.1f}  {headroom_per_step[k]:9.1f}  {phase}{note}")

        print(f"\n    1-step lag violations:")
        any_viol = False
        for k in range(1, N):
            dq   = jpos[k] - jpos[k - 1]
            ddq  = jvel[k] - jvel[k - 1]
            tau_lag = KP * dq + KD * ddq + ff[k]
            over = [(JNAMES[j], tau_lag[j], TORQUE_LIMS[j])
                    for j in range(12) if abs(tau_lag[j]) > TORQUE_LIMS[j]]
            if over:
                any_viol = True
                parts = "  ".join(f"{nm}={val:+.0f}Nm(lim={lim:.0f})" for nm, val, lim in over)
                print(f"      step {k:2d}: max={np.abs(tau_lag).max():.1f}Nm  {parts}")
        if not any_viol:
            print(f"      none")

    # ── FK analysis ─────────────────────────────────────────────
    fk = fk_analysis(state, grf, jvel, contact, lo, land)
    if fk:
        viols = fk["mdp_viols"]
        tau   = fk["tau_rot"]
        fvt   = fk["foot_vt"]
        ax_   = fk["ax"]
        print(f"\n  [foot kinematics]")
        stance_lo = lo or 0
        if stance_lo > 0:
            print(f"    foot |v_tang| m/s  (stance steps 0–{stance_lo - 1}):")
        else:
            print("    foot |v_tang| m/s  (stance steps: liftoff unknown, skipping)")
        for k in range(0, stance_lo):
            print(f"      step {k:2d}: FL={np.linalg.norm(fvt[k,0]):.2f} FR={np.linalg.norm(fvt[k,1]):.2f}"
                  f" RL={np.linalg.norm(fvt[k,2]):.2f} RR={np.linalg.norm(fvt[k,3]):.2f} m/s")
        ax_name2 = ['roll','pitch','yaw'][ax_]
        print(f"\n    planned τ_{ax_name2} from GRF (stance):")
        for k in range(lo or 0):
            print(f"      step {k:2d}: {tau[k]:+.1f} Nm")
        print(f"    cumulative ΔL_{ax_name2} (stance) = {np.sum(tau[:lo or 0])*DT:+.3f} kg·m²/s")
        print(f"\n    MDP violations: {len(viols)} / {sum(1 for k in range(lo or 0) for fi in range(4) if (contact is None or contact[fi,k]>0.5) and grf[k,fi*3+2]>5)} foot-steps")
        viols_sorted = sorted(viols, key=lambda x: -x[2])
        print(f"    top 10 by f_t·v_t (worst first):")
        for k, fi, dot, ft_mag, vt_mag in viols_sorted[:10]:
            fname = ['FL', 'FR', 'RL', 'RR'][fi]
            print(f"      step {k:2d} {fname}: dot={dot:+.1f} |f_t|={ft_mag:.1f}N |v_t|={vt_mag:.3f}m/s")

    # ── Terminal state ───────────────────────────────────────────
    last    = state[-1]
    com_vel_f = last[3:6]
    omega_f   = last[9:12]
    jvel_f    = jvel[-1]
    fz_f      = grf[-1].reshape(4, 3)[:, 2]
    print(f"\n  [terminal state]")
    print(f"    com_vel  = {np.linalg.norm(com_vel_f):.3f} m/s  {np.round(com_vel_f, 3)}")
    print(f"    omega    = {np.linalg.norm(omega_f):.3f} rad/s  {np.round(omega_f, 3)}")
    print(f"    jvel max = {np.abs(jvel_f).max():.3f} rad/s  ({JNAMES[np.abs(jvel_f).argmax()]})")
    print(f"    fz_final = FL={fz_f[0]:.0f}  FR={fz_f[1]:.0f}  RL={fz_f[2]:.0f}  RR={fz_f[3]:.0f} N")

    # ═══ Error summary ══════════════════════════════════════════════════════
    # Plain-English list of problems found above. Re-examines already-computed
    # quantities — no new data is loaded, saved, or modified.
    issues = []

    if com_vel0_norm < 0.05 and omega0_norm < 0.05 and jv0_max > 1.0:
        issues.append((
            "Robot starts with moving joints",
            f"At step 0, the body is still but a joint is already spinning at "
            f"{jv0_max:.1f} rad/s ({JNAMES[np.abs(jvel[0]).argmax()]}). "
            f"Physics sim becomes unstable on the very first step."
        ))

    if jpos_viol:
        issues.append((
            "Joint bent past its limit",
            f"{len(jpos_viol)} moments in the trajectory have a joint bent further "
            f"than the robot can physically reach."
        ))

    if cap_hits:
        details = ", ".join(f"{n} ({c} steps)" for n, c in cap_hits.items())
        issues.append((
            "Joint spinning at max speed",
            f"Some joints hit their top motor speed. Motors can't go any faster. "
            f"Joints affected: {details}."
        ))

    QDDOT_BOUND = np.array([500., 500., 1000.] * 4)
    qddot_max_per_j = np.abs(qddot).max(axis=0)
    qddot_over = [(JNAMES[j], qddot_max_per_j[j], QDDOT_BOUND[j])
                  for j in range(12) if qddot_max_per_j[j] > QDDOT_BOUND[j]]
    if qddot_over:
        worst = max(qddot_over, key=lambda x: x[1] - x[2])
        issues.append((
            "Joint accelerating too fast",
            f"{len(qddot_over)} of 12 joints are told to change speed faster than "
            f"is safe for the real motors. Worst offender: {worst[0]} at "
            f"{worst[1]:.0f} rad/s² (recommended limit: {worst[2]:.0f})."
        ))

    for t, lbl in [(lo, "liftoff"), (land, "landing")]:
        if t is None or t == 0 or t >= N: continue
        dv = np.abs(jvel[t] - jvel[t-1])
        sat = [(JNAMES[j], KD*dv[j]) for j in range(12) if KD*dv[j] > TORQUE_LIMS[j]]
        if sat:
            worst_nm, worst_nm_val = max(sat, key=lambda x: x[1])
            issues.append((
                f"Not enough motor power at {lbl}",
                f"{len(sat)} joints would need more torque than the motor can produce "
                f"at {lbl}. Worst: {worst_nm} needs {worst_nm_val:.0f} Nm but the "
                f"motor only gives {TORQUE_LIMS[JNAMES.index(worst_nm)]:.0f} Nm."
            ))

    if util_all and util_all[0][2] >= 0.999:
        foot_names = ["FL", "FR", "RL", "RR"]
        n_fat = sum(1 for _, _, u in util_all if u >= 0.999)
        issues.append((
            "No grip margin",
            f"{n_fat} moments push the foot sideways at the very edge of what the "
            f"floor grip allows. If the real floor is any less grippy than simulated, "
            f"the robot will slip."
        ))

    if land is not None:
        impact_N = fz[max(0, land-1):min(N, land+3)].sum(1).max()
        if impact_N / BODY_WEIGHT > 2.0:
            issues.append((
                "Hard landing",
                f"Robot lands with {impact_N/BODY_WEIGHT:.1f}× its body weight of "
                f"force ({impact_N:.0f} N peak). The ground force jumps from 0 to "
                f"this value instantly when the feet touch down."
            ))

    lw2 = l_world_analysis(state, grf, jvel, contact, lo, touchdown) if lo else None
    if lw2 is not None and (lw2["max_dL_per_axis"] > lw2["tol"]).any():
        issues.append((
            "Spin changes in mid-air",
            f"While airborne, the robot's spin drifts more than the MPC should allow. "
            f"Spin can't change mid-flight without touching anything — something is "
            f"off in the plan."
        ))

    if ff is not None:
        ff_sat_mask = np.abs(ff) > TORQUE_LIMS[None, :]
        n_ff_sat = int(ff_sat_mask.any(axis=1).sum())
        if n_ff_sat:
            issues.append((
                "Motor overload",
                f"{n_ff_sat} out of {N} steps ask for more torque than the motor is "
                f"rated to give."
            ))

    if ff is not None:
        KP_SIM = 25.0
        jp_all = state[:N, 12:24]
        n_pd = 0
        for k in range(1, N):
            tau_lag = KP_SIM * (jp_all[k] - jp_all[k-1]) + KD * (jvel[k] - jvel[k-1]) + ff[k]
            if np.any(np.abs(tau_lag) > TORQUE_LIMS):
                n_pd += 1
        if n_pd:
            issues.append((
                "No room for the controller to correct errors",
                f"{n_pd} steps use up all the motor's torque budget. If the real "
                f"robot drifts off the plan even slightly, the onboard controller "
                f"can't push it back — it has nothing left to give."
            ))

    if fk is not None and fk.get("mdp_viols"):
        issues.append((
            "Ground friction pointing the wrong way",
            f"{len(fk['mdp_viols'])} moments have ground friction pushing the foot "
            f"in the same direction it's sliding. Friction must oppose sliding — "
            f"this is physically impossible."
        ))

    # ── Print formatted summary ─────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  SUMMARY — {len(issues)} issue{'s' if len(issues) != 1 else ''} found")
    print(f"{'═'*60}")
    if not issues:
        print(f"  No problems detected. Trajectory looks clean.")
    else:
        for i, (title, desc) in enumerate(issues, 1):
            print(f"\n  {i}. {title}")
            print(f"     {desc}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    sf       = Path(sys.argv[1])
    gf       = Path(sys.argv[2])
    jf       = Path(sys.argv[3])
    traj_dir = Path(sys.argv[4])
    iter_num = sys.argv[5]

    # Auto-detect contact sequence file if present (committed shell does not pass one).
    cf_candidates = sorted(traj_dir.glob(f"contact_seq*_iter_{iter_num}.npy"))
    cf = cf_candidates[0] if cf_candidates else None

    run(traj_dir, sf, gf, jf, cf, iter_num)
    print()
PYEOF
