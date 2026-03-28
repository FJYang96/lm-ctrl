"""Trajectory solve logic for the MPC with slack formulation."""

from __future__ import annotations

import logging
from typing import Any

import casadi as cs
import numpy as np

import go2_config

logger = logging.getLogger("llm_integration")


def solve_trajectory(
    mpc: Any,
    initial_state: np.ndarray,
    ref: np.ndarray,
    contact_sequence: np.ndarray,
    ref_trajectory: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Solve the trajectory optimization problem.

    Args:
        mpc: QuadrupedMPCOptiSlack instance.
        initial_state: Initial state vector.
        ref: Reference trajectory (shape: states_dim + inputs_dim).
        contact_sequence: Contact sequence array (shape: 4 x horizon).
        ref_trajectory: Optional dict with 'X_ref' and 'U_ref' for cost + initial guess.
    """
    mpc.opti.set_value(mpc.P_initial_state, initial_state)
    mpc.opti.set_value(mpc.P_contact, contact_sequence)

    if ref_trajectory is not None:
        X_ref_param = ref_trajectory["X_ref"]
        U_ref_param = ref_trajectory["U_ref"]
    else:
        ref_state = ref[: mpc.states_dim]
        ref_input = ref[mpc.states_dim :]
        terminal_state = initial_state.copy()
        terminal_state[0] = ref[0]
        terminal_state[1] = ref[1]
        terminal_state[2] = initial_state[2]
        terminal_state[3:6] = 0.0
        terminal_state[9:12] = 0.0
        terminal_state[12:24] = initial_state[12:24]

        X_ref_param = np.zeros((mpc.states_dim, mpc.horizon + 1))
        for k in range(mpc.horizon + 1):
            if k < mpc.pre_flight_steps:
                X_ref_param[:, k] = initial_state
            elif k < mpc.landing_start:
                X_ref_param[:, k] = ref_state
            else:
                X_ref_param[:, k] = terminal_state
        U_ref_param = np.tile(ref_input.reshape(-1, 1), (1, mpc.horizon))

    mpc.opti.set_value(mpc.P_X_ref, X_ref_param)
    mpc.opti.set_value(mpc.P_U_ref, U_ref_param)

    # Initial guess
    if ref_trajectory is not None:
        logger.info("Using reference trajectory as initial guess")
        X_init = ref_trajectory["X_ref"].copy()
        U_init = ref_trajectory["U_ref"].copy()
    else:
        logger.info("Cold-starting with heuristic initial guess")
        X_init = np.tile(initial_state.reshape(-1, 1), (1, mpc.horizon + 1))
        U_init = np.zeros((mpc.inputs_dim, mpc.horizon))
        for i in range(mpc.horizon):
            U_init[0:12, i] = 0.001 * np.sin(np.arange(12) * 0.1)
            contact_i = contact_sequence[:, min(i, contact_sequence.shape[1] - 1)]
            for foot in range(4):
                if contact_i[foot] > 0.5:
                    U_init[12 + foot * 3 + 2, i] = (
                        go2_config.robot_data.mass
                        * go2_config.experiment.gravity_constant / 4 * 0.8
                    )

    mpc.opti.set_initial(mpc.X, X_init)
    mpc.opti.set_initial(mpc.U, U_init)

    if mpc.use_slack:
        for (s_l, s_u, _) in mpc.slack_variables.values():
            mpc.opti.set_initial(s_l, 0)
            mpc.opti.set_initial(s_u, 0)

    try:
        sol = mpc.opti.solve()
        mpc._last_solution = sol
        X_opt = sol.value(mpc.X)
        U_opt = sol.value(mpc.U)
        status = 0
    except Exception as e:
        print(f"Optimization failed: {e}")
        mpc._last_solution = None
        X_opt = mpc.opti.debug.value(mpc.X)
        U_opt = mpc.opti.debug.value(mpc.U)
        mpc._debug_accessor = mpc.opti.debug
        status = 1

    return X_opt.T, U_opt[12:24, :].T, U_opt[0:12, :].T, status
