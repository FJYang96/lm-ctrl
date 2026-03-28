"""Constraint analysis functions for MPC with slack formulation."""

from __future__ import annotations

from typing import Any

import numpy as np


def get_constraint_violations(mpc: Any) -> dict[str, Any]:
    """Analyze the current (possibly infeasible) iterate for debugging."""
    info: dict[str, Any] = {
        "terminal_state": {},
        "trajectory_info": [],
        "summary": [],
    }

    X_debug = mpc.opti.debug.value(mpc.X)

    x_terminal = X_debug[:, -1]
    info["terminal_state"] = {
        "position": {"x": x_terminal[0], "y": x_terminal[1], "z": x_terminal[2]},
        "velocity": {"vx": x_terminal[3], "vy": x_terminal[4], "vz": x_terminal[5]},
        "orientation": {
            "roll": x_terminal[6], "pitch": x_terminal[7], "yaw": x_terminal[8],
        },
        "angular_velocity": {
            "wx": x_terminal[9], "wy": x_terminal[10], "wz": x_terminal[11],
        },
    }

    for k in range(X_debug.shape[1]):
        height = X_debug[2, k]
        if height < 0.0:
            info["trajectory_info"].append(
                f"Step {k}: height={height:.3f}m (robot underground!)"
            )

    if not info["trajectory_info"]:
        info["summary"].append("No physics violations detected")
    else:
        info["summary"].append(f"Physics issues: {len(info['trajectory_info'])}")

    return info


def analyze_constraint_hardness(mpc: Any) -> dict[str, dict[str, Any]]:
    """Analyze hardness of each constraint based on slack values.

    Returns dict mapping constraint names to hardness metrics.
    """
    if not mpc.use_slack:
        return {}

    value_fn = None
    if hasattr(mpc, "_last_solution") and mpc._last_solution is not None:
        value_fn = mpc._last_solution.value
    elif hasattr(mpc, "_debug_accessor") and mpc._debug_accessor is not None:
        value_fn = mpc._debug_accessor.value
    else:
        return {}

    results: dict[str, dict[str, Any]] = {}

    for (constraint_name, k), (s_l, s_u, n_constraints) in mpc.slack_variables.items():
        s_l_val = np.array(value_fn(s_l)).flatten()
        s_u_val = np.array(value_fn(s_u)).flatten()

        slack_L1 = float(np.sum(np.abs(s_l_val)) + np.sum(np.abs(s_u_val)))
        slack_Linf = float(max(np.max(np.abs(s_l_val)), np.max(np.abs(s_u_val))))

        if constraint_name not in results:
            results[constraint_name] = {
                "total_slack_L1": 0.0,
                "max_slack_Linf": 0.0,
                "total_dims": 0,
                "active_timesteps": [],
                "slack_by_timestep": {},
            }

        results[constraint_name]["total_slack_L1"] += slack_L1
        results[constraint_name]["max_slack_Linf"] = max(
            results[constraint_name]["max_slack_Linf"], slack_Linf
        )
        results[constraint_name]["total_dims"] += n_constraints
        results[constraint_name]["slack_by_timestep"][k] = slack_L1

        if slack_L1 > 1e-6:
            results[constraint_name]["active_timesteps"].append(k)

    for name in results:
        total_dims = results[name]["total_dims"]
        results[name]["mean_slack_per_dim"] = (
            results[name]["total_slack_L1"] / total_dims if total_dims > 0 else 0.0
        )

    return dict(sorted(results.items(), key=lambda x: -x[1]["max_slack_Linf"]))
