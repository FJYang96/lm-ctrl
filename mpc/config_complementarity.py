"""
Complementarity-based contact optimization configuration.

Key concepts:
1. Contact forces and velocities must satisfy complementarity: f_i * v_i = 0
2. We relax this using a barrier function: f_i * v_i <= ε
3. Contact forces are only non-zero when in contact (determined by gait schedule)
4. Foot velocities should be zero during stance phase
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import casadi as cs
import numpy as np

import go2_config

from .constraints import (
    body_clearance_constraints,
    complementarity_constraints,
    foot_height_constraints,
    friction_cone_constraints,
    input_limits_constraints,
    joint_limits_constraints,
)
from .dynamics.model import KinoDynamic_Model
from .mpc_config import MPCConfig


@dataclass
class ComplementarityMPCConfig(MPCConfig):
    """
    MPC configuration with complementarity constraints for contact handling.

    This extends the base MPCConfig to include complementarity constraints
    that better model contact dynamics by ensuring forces and velocities
    satisfy physical contact mechanics.
    """

    pre_flight_stance_duration: float = go2_config.default_pre_flight_stance_duration
    flight_duration: float = go2_config.default_flight_duration
    path_constraints: list[
        Callable[
            [cs.MX, cs.MX, KinoDynamic_Model, Any, cs.MX], tuple[cs.MX, cs.MX, cs.MX]
        ]
    ] = field(
        default_factory=lambda: [
            friction_cone_constraints,
            foot_height_constraints,
            complementarity_constraints,  # Add complementarity
            joint_limits_constraints,
            input_limits_constraints,
            body_clearance_constraints,  # Ensure body stays above ground
        ]
    )
    path_constraint_params: dict[str, float] = field(
        default_factory=lambda: {
            "COMPLEMENTARITY_EPS": 1e-3,
            "SWING_GRF_EPS": 0.0,
            "STANCE_HEIGHT_EPS": 0.02,
            "NO_SLIP_EPS": 0.005,
        }
    )

    @property
    def contact_sequence(self) -> np.ndarray:
        """Generate contact sequence based on gait parameters."""
        if self._contact_sequence is None:
            mpc_horizon = int(self.duration / self.mpc_dt)
            pre_flight_steps = int(self.pre_flight_stance_duration / self.mpc_dt)
            flight_steps = int(self.flight_duration / self.mpc_dt)
            contact_sequence = np.ones((4, mpc_horizon))
            contact_sequence[:, pre_flight_steps : pre_flight_steps + flight_steps] = (
                0.0
            )
            self._contact_sequence = contact_sequence
        return self._contact_sequence
