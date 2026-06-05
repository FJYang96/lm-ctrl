#!/bin/bash
# Replay MPC from saved constraint code (no LLM). Same env as run_local.sh.
#
# Usage:
#   ./run_mpc_replay.sh results/example/sideflip.py
#   ./run_mpc_replay.sh results/example/sideflip.py --output-dir results/mpc_replay/sideflip
#   ./run_mpc_replay.sh --constraint-from-summary results/llm_iterations/.../summary_prompt_iter_2.txt
#
# IPOPT solver verbosity (default: quiet, ipopt.print_level=0 in go2_config):
#   ./run_mpc_replay.sh results/example/sideflip.py --solver-verbose
#   MPC_SOLVER_VERBOSE=1 ./run_mpc_replay.sh results/example/sideflip.py

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=run_common.sh
source "$SCRIPT_DIR/run_common.sh"

cd "$SCRIPT_DIR"

SOLVER_VERBOSE_FLAG="$(_mpc_solver_verbose_flag)"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <constraint.py> [mpc_replay options...]"
    echo "   or: $0 --constraint-from-summary <summary_prompt_iter_N.txt> [options...]"
    echo ""
    echo "Options (passed to mpc_replay.py):"
    echo "  --output-dir PATH      Output directory (default: results/mpc_replay)"
    echo "  --iteration N          Trajectory filename index (default: 1)"
    echo "  --no-slack             Use hard constraints"
    echo "  --skip-render          Skip trajectory video"
    echo "  --config MODE          standard | complementarity"
    echo "  --solver-verbose       Print IPOPT iteration log (ipopt.print_level=5)"
    echo "  --verbose              Python tracebacks on errors"
    echo ""
    echo "Environment:"
    echo "  PYTHON               Python executable (default: same as run_local.sh)"
    echo "  MPC_SOLVER_VERBOSE=1 Same as --solver-verbose"
    echo ""
    echo "Examples:"
    echo "  $0 results/example/sideflip.py"
    echo "  $0 results/example/sideflip.py --output-dir results/mpc_replay/sideflip --solver-verbose"
    echo "  $0 --constraint-from-summary results/llm_iterations/.../summary_prompt_iter_2.txt"
    exit 1
fi

if [ "$1" = "--constraint-from-summary" ]; then
    exec "$PYTHON" mpc_replay.py $SOLVER_VERBOSE_FLAG "$@"
fi

exec python mpc_replay.py $SOLVER_VERBOSE_FLAG --constraint-file "$@"
