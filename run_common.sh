# Shared environment for local scripts (run_local.sh, run_mpc_replay.sh, etc.)
# Source from the repo root after setting SCRIPT_DIR:
#   SCRIPT_DIR="$(cd "$(dirname "$0")")" && pwd)"
#   # shellcheck source=run_common.sh
#   source "$SCRIPT_DIR/run_common.sh"

# Non-interactive matplotlib
export MPLBACKEND=Agg
# Headless OpenGL for MuJoCo rendering (avoids gladLoadGL errors)
export MUJOCO_GL=egl

# IPOPT/MUMPS + OpenBLAS must run single-threaded locally; multi-threaded BLAS
# inside CasADi/Ipopt commonly causes segfaults or hangs on this problem.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Optional: pass --solver-verbose to mpc_replay when set to 1/true/yes
_mpc_solver_verbose_flag() {
    case "${MPC_SOLVER_VERBOSE:-}" in
        1|true|TRUE|yes|YES) echo --solver-verbose ;;
        *) echo ;;
    esac
}
