#!/bin/bash
# Run trajectory diagnostics for a specific iteration.
#
# Usage:
#   ./run_diagnose.sh results/llm_iterations/backflip2 10
#   ./run_diagnose.sh results/llm_iterations/backflip 20
#   ./run_diagnose.sh results/llm_iterations/backflip 20 --sim path/to/debug_steps.txt

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-/home/aryanroy/miniconda3/bin/python3}"

# ── Edit these defaults ──
DEFAULT_TRAJ_DIR="results/llm_iterations/do_a_backflip_1775891288"
DEFAULT_ITER=4

TRAJ_DIR="${1:-$DEFAULT_TRAJ_DIR}"
ITER="${2:-$DEFAULT_ITER}"
shift 2 2>/dev/null || true

# Collect --sim if provided
SIM_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --sim) SIM_ARG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Build temp dir with symlinks to the iter-specific .npy files
ITER_DIR=$(mktemp -d)
trap "rm -rf $ITER_DIR" EXIT

found=0
for f in "$TRAJ_DIR"/*_iter_${ITER}.npy "$TRAJ_DIR"/*_iter${ITER}.npy; do
    [ -f "$f" ] || continue
    ln -s "$(realpath "$f")" "$ITER_DIR/$(basename "$f")"
    found=1
done

if [ "$found" -eq 0 ]; then
    echo "Error: no .npy files found for iter ${ITER} in ${TRAJ_DIR}"
    echo "Available files:"
    ls "$TRAJ_DIR"/*.npy 2>/dev/null | head -20
    exit 1
fi

echo "Found .npy files for iter ${ITER}:"
ls -1 "$ITER_DIR"

"$PYTHON" -u - "$ITER_DIR" "$SIM_ARG" <<'PYEOF'
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
VEL_CAP_RAW = 15.70
VEL_LIMIT_FRAC = 0.8   # must match path_constraint_params["VEL_LIMIT_FRAC"]
VEL_CAP     = VEL_CAP_RAW * VEL_LIMIT_FRAC
JPOS_LO = np.array([-1.047,-1.571,-2.723,-1.047,-1.571,-2.723,-1.047,-0.524,-2.723,-1.047,-0.524,-2.723])
JPOS_HI = np.array([ 1.047, 3.491,-0.838, 1.047, 3.491,-0.838, 1.047, 4.538,-0.838, 1.047, 4.538,-0.838])
JNAMES = ["FL_hip","FL_thigh","FL_calf","FR_hip","FR_thigh","FR_calf",
          "RL_hip","RL_thigh","RL_calf","RR_hip","RR_thigh","RR_calf"]


def ff_torque_analysis(sf, gf, jf, cf):
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
    N = grf.shape[0]
    fz = grf.reshape(N,4,3)[:,:,2]
    lo = land = None
    if contact is not None:
        af = np.all(contact < 0.5, axis=0)
        as_ = np.all(contact > 0.5, axis=0)
        for k in range(N):
            if af[k] and lo is None: lo = k
            if lo and as_[k] and land is None: land = k
    else:
        fzt = fz.sum(1)
        for k in range(N):
            if fzt[k] < 5 and lo is None: lo = k
            if lo and fzt[k] > 50 and land is None: land = k
    return lo, land


def parse_sim(path):
    steps = []
    for line in Path(path).read_text().splitlines():
        m = re.match(r"\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
                     r"\s*([\d.]+)\s*\|\s*([+-]?[\d.]+)\s*\|\s*([+-]?[\d.]+)\s*\|"
                     r"\s*([+-]?[\d.]+)\s*\|\s*(\S+)\s*\|\s*(.*)", line)
        if m:
            steps.append(dict(step=int(m[1]), pos_err=float(m[2]), ori_err=float(m[3]),
                               jnt_err=float(m[4]), torque=float(m[5]),
                               wz_act=float(m[6]), wz_ref=float(m[7]),
                               contact=m[9], reason=m[10].strip()))
    return steps


def run(traj_dir, sim_path=None):
    npy = list(traj_dir.glob("*.npy"))
    def pick(p): m=[f for f in npy if re.search(p,f.name)]; return m[0] if m else None
    sf,gf,jf,cf = pick(r"state_traj"),pick(r"grf_traj"),pick(r"joint_vel_traj"),pick(r"contact_seq")
    if not (sf and gf and jf):
        print(f"Missing required .npy files in {traj_dir}")
        return

    state   = np.load(sf)
    grf     = np.load(gf)
    jvel    = np.load(jf)
    contact = np.load(cf) if cf else None
    N = grf.shape[0]
    lo, land = detect_phases(grf, contact)

    fz  = grf.reshape(N,4,3)[:,:,2]
    ft  = grf.reshape(N,4,3)[:,:,:2]
    euler = state[:, 6:9]
    omega = state[:, 9:12]

    # ── Color helpers ──
    USE_COLOR = sys.stdout.isatty()
    def _c(code, text):
        return f"\033[{code}m{text}\033[0m" if USE_COLOR else text
    def red(t):    return _c("1;31", t)
    def yellow(t): return _c("1;33", t)
    def green(t):  return _c("1;32", t)
    def bold(t):   return _c("1", t)
    def dim(t):    return _c("2", t)

    def tag_pass():  return green("PASS")
    def tag_fail():  return red("FAIL")
    def tag_warn():  return yellow("WARN")

    # Collect all flagged issues for the final summary
    flags = []

    print(f"\n{'='*64}")
    print(f"  {bold(traj_dir.name)}")
    print(f"  N={N}  liftoff={lo}  landing={land}  "
          f"flight={(land-lo) if lo and land else '?'} steps  dt={DT}s")
    print(f"{'='*64}")

    # ── 1. Initial conditions ──
    fz0 = fz[0].sum()
    jv0_max = np.abs(jvel[0]).max()
    alpha0 = np.abs(omega[1] - omega[0]).max() / DT
    com_vel0_norm = np.linalg.norm(state[0, 3:6])
    omega0_norm   = np.linalg.norm(state[0, 9:12])
    init_bad = com_vel0_norm < 0.05 and omega0_norm < 0.05 and jv0_max > 1.0
    init_fz_low = fz0 < BODY_WEIGHT * 0.6

    print(f"\n  {bold('[1] initial conditions')}  "
          f"{tag_fail() if init_bad else (tag_warn() if init_fz_low else tag_pass())}")
    print(f"      step-0 fz   = {fz0:.0f} N  ({fz0/BODY_WEIGHT*100:.0f}% BW)"
          f"{'  ' + red('<< low') if init_fz_low else ''}")
    print(f"      step-0 jvel = {jv0_max:.2f} r/s  ({JNAMES[np.abs(jvel[0]).argmax()]})"
          f"{'  ' + red('<< nonzero at rest') if init_bad else ''}")
    print(f"      step 0->1 a = {alpha0:.0f} rad/s^2")
    if init_bad:
        flags.append(f"init: joints at {jv0_max:.1f} r/s while body at rest")
        print(f"      {red('per-foot fz')}: FL={fz[0,0]:.0f}  FR={fz[0,1]:.0f}  "
              f"RL={fz[0,2]:.0f}  RR={fz[0,3]:.0f} N  "
              f"{dim('(need ' + f'{BODY_WEIGHT/4:.0f}' + ' each)')}")
    if init_fz_low and not init_bad:
        flags.append(f"init: total fz only {fz0:.0f} N ({fz0/BODY_WEIGHT*100:.0f}% BW)")

    # ── 2. Kinematics ──
    jpos_viol = [(k,j,state[k,12+j]) for k in range(N+1) for j in range(12)
                 if state[k,12+j] < JPOS_LO[j]-0.05 or state[k,12+j] > JPOS_HI[j]+0.05]
    jvel_peaks = [(JNAMES[j], np.abs(jvel[:,j]).max()) for j in range(12)]
    cap_hits   = {JNAMES[j]: int((np.abs(jvel[:,j]) >= VEL_CAP-0.01).sum()) for j in range(12)
                  if (np.abs(jvel[:,j]) >= VEL_CAP-0.01).any()}
    rot_deg = np.degrees(np.abs(euler[-1] - euler[0]))
    peak_name, peak_val = max(jvel_peaks, key=lambda x: x[1])
    has_jpos = len(jpos_viol) > 0
    has_cap  = len(cap_hits) > 0

    print(f"\n  {bold('[2] kinematics')}  "
          f"{tag_fail() if (has_jpos or has_cap) else tag_pass()}")
    print(f"      jpos violations = {red(str(len(jpos_viol))) if has_jpos else green('0')}")
    print(f"      jvel peak       = {peak_val:.2f} r/s ({peak_name})  "
          f"{dim('limit=' + str(JVEL_LIMIT))}")
    if has_cap:
        flags.append(f"vel cap: {', '.join(f'{n} x{c}' for n, c in cap_hits.items())}")
        print(f"      vel cap hits    = {red(str(cap_hits))}")
    else:
        print(f"      vel cap hits    = {green('none')}")
    if has_jpos:
        flags.append(f"jpos: {len(jpos_viol)} violations outside URDF limits")
    print(f"      total rotation  = roll {rot_deg[0]:.0f}  pitch {rot_deg[1]:.0f}  yaw {rot_deg[2]:.0f} deg")

    # ── 3. Contact transitions ──
    any_kd_sat = False
    trans_details = []
    for t, label in [(lo,'liftoff'), (land,'landing')]:
        if t is None or t == 0 or t >= N: continue
        dv = np.abs(jvel[t] - jvel[t-1])
        sat = [(JNAMES[j], KD*dv[j], TORQUE_LIMS[j]) for j in range(12) if KD*dv[j] > TORQUE_LIMS[j]]
        trans_details.append((label, t, dv, sat))
        if sat:
            any_kd_sat = True

    print(f"\n  {bold('[3] contact transitions')}  "
          f"{tag_fail() if any_kd_sat else tag_pass()}")
    for label, t, dv, sat in trans_details:
        marker = red("!!") if sat else "  "
        print(f"    {marker} {label:8s} (step {t}): max dq={dv.max():.1f} r/s ({JNAMES[dv.argmax()]})  "
              f"KD_sat={red(str(len(sat))) if sat else green('0')} joints")
        for nm, kd, lim in sat:
            flags.append(f"KD sat @ {label}: {nm} {kd:.0f} Nm > {lim:.0f} Nm limit")
            print(f"         {red(f'{nm}: {kd:.0f} Nm')} > {lim:.0f} Nm limit")
    if lo is not None and lo < state.shape[0]-1:
        dw = np.abs(omega[lo] - omega[lo-1]).max()
        print(f"      omega jump at liftoff: {dw:.3f} rad/s")

    # ── 4. GRF quality ──
    stance_end = lo if lo else N
    util_all = []
    for k in range(stance_end):
        for foot in range(4):
            if contact is not None and contact[foot,k] < 0.5: continue
            z = fz[k,foot]
            if z < 5: continue
            util_all.append((k, foot, np.linalg.norm(ft[k,foot]) / (MU*z)))
    util_all.sort(key=lambda x:-x[2])

    friction_high = util_all[0][2] > 0.9 if util_all else False
    impact_bw = None
    if land is not None:
        win = fz[max(0,land-1):min(N,land+3)].sum(1)
        impact_bw = win.max() / BODY_WEIGHT
    impact_bad = impact_bw is not None and impact_bw > 3.0

    print(f"\n  {bold('[4] GRF quality')}  "
          f"{tag_fail() if impact_bad else (tag_warn() if friction_high else tag_pass())}")
    if lo:
        print(f"      stance avg fz = {fz[:lo].sum(1).mean():.0f} N")
    if util_all:
        foot_names = ['FL','FR','RL','RR']
        u_max = util_all[0]
        print(f"      friction util = max {u_max[2]*100:.0f}% at step {u_max[0]} "
              f"foot {foot_names[u_max[1]]}"
              f"{'  ' + yellow('<< near cone') if friction_high else ''}")
        print(f"      top 5:  {', '.join(f's{k} {foot_names[f]} {u*100:.0f}%' for k,f,u in util_all[:5])}")
        if friction_high:
            flags.append(f"friction: {u_max[2]*100:.0f}% utilization (near cone limit)")
    else:
        print(f"      friction util = n/a")
    if impact_bw is not None:
        marker = red("!!") if impact_bad else "  "
        val_str = f"{impact_bw:.2f}x BW  (peak {impact_bw * BODY_WEIGHT:.0f} N)"
        print(f"    {marker} landing impact = "
              f"{red(val_str) if impact_bad else val_str}")
        if impact_bad:
            flags.append(f"landing: {impact_bw:.1f}x BW impact (PhysX cannot reproduce)")

    # ── 5. Angular momentum ──
    ax = int(np.abs(euler[-1]-euler[0]).argmax())
    ax_name = ['roll','pitch','yaw'][ax]
    print(f"\n  {bold('[5] angular momentum')}  axis={ax_name}")
    if lo:
        om_lo = omega[lo, ax]
        windup = [k for k in range(lo) if np.sign(omega[k,ax])==-np.sign(om_lo) and abs(omega[k,ax])>0.3]
        print(f"      liftoff omega = {om_lo:.3f} rad/s")
        print(f"      wind-up steps = {windup}  "
              f"(peak {max((abs(omega[k,ax]) for k in windup),default=0):.2f} rad/s)")
        if land is not None:
            rot_flight = euler[land,ax] - euler[lo,ax]
            avg_needed = rot_flight / ((land-lo)*DT)
            print(f"      flight rot    = {np.degrees(rot_flight):.1f} deg  "
                  f"avg_omega_needed = {avg_needed:.2f} r/s")
            drift = omega[land-1,ax] - omega[lo,ax]
            big_drift = abs(drift) > 1.0
            print(f"      flight drift  = {drift:+.3f} rad/s over {land-lo} steps"
                  f"{'  ' + yellow('<< large') if big_drift else ''}")
            if big_drift:
                flags.append(f"ang mom: {drift:+.3f} r/s drift in flight (not conserved)")
        alpha = np.diff(omega[:lo+1, ax]) / DT
        print(f"      thrust alpha  = {np.abs(alpha).max():.0f} rad/s^2 (max during stance)")

    # ── 6. FF torque feasibility (pympc docker only) ──
    ff = ff_torque_analysis(sf, gf, jf, cf)
    if ff is not None:
        sat_mask = np.abs(ff) > TORQUE_LIMS[None, :]
        sat_steps = np.where(sat_mask.any(axis=1))[0]
        has_ff_sat = len(sat_steps) > 0

        print(f"\n  {bold('[6] FF torques')}  {dim('(full inverse dynamics)')}  "
              f"{tag_fail() if has_ff_sat else tag_pass()}")
        print(f"      saturated steps: "
              f"{red(f'{len(sat_steps)} / {N}') if has_ff_sat else green(f'0 / {N}')}")
        if has_ff_sat:
            flags.append(f"FF torque: {len(sat_steps)}/{N} steps exceed motor limits")
        for k in sat_steps:
            bad = [(JNAMES[j], ff[k,j], TORQUE_LIMS[j]) for j in range(12) if sat_mask[k,j]]
            parts = "  ".join(f"{nm}={val:+.0f}/{lim:.0f}Nm" for nm,val,lim in bad)
            print(f"        step {k:2d}: {red(parts)}")
        if not has_ff_sat:
            print(f"      peak FF  = {np.abs(ff).max():.1f} Nm  ({JNAMES[np.abs(ff).argmax() % 12]})")

    # ── 7. PD+FF analysis (requires ff, pympc docker only) ──
    if ff is not None:
        KP = 25.0
        jpos = state[:N, 12:24]
        ff_max_per_step    = np.abs(ff).max(axis=1)
        headroom_per_step  = (TORQUE_LIMS[None, :] - np.abs(ff)).min(axis=1)

        lag_viols = []
        for k in range(1, N):
            dq   = jpos[k] - jpos[k - 1]
            ddq  = jvel[k] - jvel[k - 1]
            tau_lag = KP * dq + KD * ddq + ff[k]
            over = [(JNAMES[j], tau_lag[j], TORQUE_LIMS[j])
                    for j in range(12) if abs(tau_lag[j]) > TORQUE_LIMS[j]]
            if over:
                lag_viols.append((k, np.abs(tau_lag).max(), over))

        print(f"\n  {bold('[7] PD+FF tracking')}  "
              f"{tag_fail() if lag_viols else tag_pass()}")
        print(f"      {'step':>4}  {'FF_max':>7}  {'headroom':>9}  phase")
        for k in range(N):
            if contact is not None:
                phase = "flight" if np.all(contact[:, k] < 0.5) else "stance"
            else:
                phase = "flight" if (lo is not None and land is not None and lo <= k < land) else "stance"
            note = ""
            if k == lo:   note = " << TAKEOFF"
            elif k == land: note = " << LANDING"
            hr = headroom_per_step[k]
            hr_str = red(f"{hr:9.1f}") if hr < 0 else (yellow(f"{hr:9.1f}") if hr < 5 else f"{hr:9.1f}")
            print(f"      {k:4d}  {ff_max_per_step[k]:7.1f}  {hr_str}  {phase}{note}")

        print(f"\n      1-step lag violations: "
              f"{red(str(len(lag_viols))) if lag_viols else green('0')}")
        if lag_viols:
            flags.append(f"PD+FF: {len(lag_viols)} steps exceed torque limits with 1-step lag")
        for k, peak, over in lag_viols:
            parts = "  ".join(f"{nm}={val:+.0f}/{lim:.0f}Nm" for nm,val,lim in over)
            print(f"        step {k:2d}: peak={peak:.1f}Nm  {red(parts)}")
        if not lag_viols:
            print(f"        {green('none')}")

    # ── 8. FK analysis (pympc docker only) ──
    fk = fk_analysis(state, grf, jvel, contact, lo, land)
    if fk:
        viols = fk["mdp_viols"]
        tau   = fk["tau_rot"]
        fvt   = fk["foot_vt"]
        ax_   = fk["ax"]
        total_foot_steps = sum(1 for k in range(lo or 0) for fi in range(4)
                               if (contact is None or contact[fi,k]>0.5) and grf[k,fi*3+2]>5)
        has_mdp = len(viols) > 0

        print(f"\n  {bold('[8] foot kinematics / MDP')}  "
              f"{tag_fail() if has_mdp else tag_pass()}")
        stance_lo = lo or 0
        if stance_lo > 0:
            print(f"      foot |v_tang| m/s  (stance steps 0-{stance_lo - 1}):")
        else:
            print("      foot |v_tang| m/s  (liftoff unknown, skipping)")
        for k in range(0, stance_lo):
            vals = [np.linalg.norm(fvt[k,fi]) for fi in range(4)]
            parts = "  ".join(
                f"{'FL FR RL RR'.split()[fi]}="
                f"{red(f'{vals[fi]:.2f}') if vals[fi] > 0.05 else f'{vals[fi]:.2f}'}"
                for fi in range(4))
            print(f"        step {k:2d}: {parts}")
        print(f"\n      planned tau_{ax_name} from GRF (stance):")
        for k in range(lo or 0):
            print(f"        step {k:2d}: {tau[k]:+.1f} Nm")
        print(f"      cumulative dL_{ax_name} (stance) = {np.sum(tau[:lo or 0])*DT:+.3f} kg*m^2/s")
        print(f"\n      MDP violations: "
              f"{red(f'{len(viols)} / {total_foot_steps}') if has_mdp else green(f'0 / {total_foot_steps}')} foot-steps")
        if has_mdp:
            flags.append(f"MDP: {len(viols)}/{total_foot_steps} foot-steps have f_t . v_t > 0 (energy injection)")
        viols_sorted = sorted(viols, key=lambda x: -x[2])
        if viols_sorted:
            print(f"      worst 10 by f_t . v_t:")
        for k, fi, dot, ft_mag, vt_mag in viols_sorted[:10]:
            fname = ['FL', 'FR', 'RL', 'RR'][fi]
            print(f"        step {k:2d} {fname}: dot={red(f'{dot:+.1f}')} "
                  f"|f_t|={ft_mag:.1f}N  |v_t|={vt_mag:.3f}m/s")

    # ── 9. Terminal state ──
    last    = state[-1]
    com_vel_f = last[3:6]
    omega_f   = last[9:12]
    jvel_f    = jvel[-1]
    fz_f      = grf[-1].reshape(4, 3)[:, 2]
    term_vel_bad  = np.linalg.norm(com_vel_f) > 1.0
    term_omeg_bad = np.linalg.norm(omega_f) > 1.0

    print(f"\n  {bold('[9] terminal state')}  "
          f"{tag_warn() if (term_vel_bad or term_omeg_bad) else tag_pass()}")
    vel_str = f"{np.linalg.norm(com_vel_f):.3f} m/s  {np.round(com_vel_f, 3)}"
    omg_str = f"{np.linalg.norm(omega_f):.3f} rad/s  {np.round(omega_f, 3)}"
    print(f"      com_vel  = {red(vel_str) if term_vel_bad else vel_str}")
    print(f"      omega    = {red(omg_str) if term_omeg_bad else omg_str}")
    print(f"      jvel max = {np.abs(jvel_f).max():.3f} rad/s  ({JNAMES[np.abs(jvel_f).argmax()]})")
    print(f"      fz_final = FL={fz_f[0]:.0f}  FR={fz_f[1]:.0f}  RL={fz_f[2]:.0f}  RR={fz_f[3]:.0f} N")
    if term_vel_bad:
        flags.append(f"terminal: COM velocity {np.linalg.norm(com_vel_f):.2f} m/s (unstable)")
    if term_omeg_bad:
        flags.append(f"terminal: angular velocity {np.linalg.norm(omega_f):.2f} r/s (unstable)")

    # ── 10. Simulation ──
    if sim_path:
        steps = parse_sim(sim_path)
        if steps:
            n_total_m = re.search(r"Terminated at step \d+/(\d+)", Path(sim_path).read_text())
            n_total = int(n_total_m[1]) if n_total_m else len(steps)
            def ref_contact(s):
                return s["contact"].split("/")[1] if "/" in s["contact"] else s["contact"]
            stance_ori = [s["ori_err"] for s in steps if all(c=="S" for c in ref_contact(s))]
            flight_ori = [s["ori_err"] for s in steps if all(c=="F" for c in ref_contact(s))]
            land_ori   = [s["ori_err"] for s in steps[-6:] if "S" in ref_contact(s)]
            liftoff_s  = next((s for s in steps if all(c=="F" for c in ref_contact(s))), None)
            omega_eff  = (liftoff_s["wz_act"]/liftoff_s["wz_ref"]*100
                          if liftoff_s and abs(liftoff_s["wz_ref"])>0.1 else float("nan"))
            early_term = len(steps) < n_total * 0.9
            print(f"\n  {bold('[10] simulation')}  {len(steps)}/{n_total} steps  "
                  f"{tag_fail() if early_term else tag_pass()}")
            if early_term:
                flags.append(f"sim: terminated early at step {len(steps)}/{n_total} ({steps[-1]['reason']})")
            print(f"      stance ori_err  = {np.mean(stance_ori):.3f} rad" if stance_ori else f"      stance ori_err  = n/a")
            print(f"      flight ori_err  = {np.mean(flight_ori):.3f} rad" if flight_ori else f"      flight ori_err  = n/a")
            print(f"      landing ori_err = {np.mean(land_ori):.3f} rad"  if land_ori   else f"      landing ori_err = n/a")
            print(f"      liftoff w_eff   = {omega_eff:.0f}%" if not np.isnan(omega_eff) else f"      liftoff w_eff   = n/a")
            print(f"      term reason     = {steps[-1]['reason']}")

    # ══════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*64}")
    if flags:
        print(f"  {red(f'FLAGGED: {len(flags)} issue(s)')}")
        for i, f in enumerate(flags, 1):
            print(f"    {i}. {red(f)}")
    else:
        print(f"  {green('ALL CHECKS PASSED')}")
    print(f"{'='*64}")


def main():
    traj_arg = sys.argv[1]
    sim_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None

    p = Path(traj_arg)
    if not p.is_dir():
        print(f"Error: {p} is not a directory")
        sys.exit(1)

    matched = sim_path
    if not matched:
        er = Path.cwd() / "rl_isaac" / "eval_output"
        if er.exists():
            for sd in sorted(er.iterdir(), reverse=True):
                dbg = sd / "debug_steps.txt"
                meta = sd / "meta.json"
                if dbg.exists():
                    if meta.exists() and json.loads(meta.read_text()).get("traj_name") == p.name:
                        matched = dbg; break
                    elif p.name in sd.name:
                        matched = dbg; break

    run(p, matched)
    print()

if __name__ == "__main__":
    main()
PYEOF
