"""MPC config for contact-scheduled stance (no-slip + bounded GRF via input limits)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import casadi as cs
import numpy as np

import go2_config

from .constraints import (
    angular_momentum_flight_constraint,
    body_clearance_constraints,
    foot_height_constraints,
    friction_cone_constraints,
    input_limits_constraints,
    joint_limits_constraints,
    link_clearance_constraints,
    no_slip_constraints,
    torque_feasibility_constraints,
)
from .dynamics.model import KinoDynamic_Model
from .mpc_config import MPCConfig


@dataclass
class ComplementarityMPCConfig(MPCConfig):
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
            no_slip_constraints,
            joint_limits_constraints,
            input_limits_constraints,
            body_clearance_constraints,
            link_clearance_constraints,
            torque_feasibility_constraints,
            angular_momentum_flight_constraint,
        ]
    )
    path_constraint_params: dict[str, float] = field(
        default_factory=lambda: {
            "SWING_GRF_EPS": 0.0,
            "STANCE_HEIGHT_EPS": 0.02,
            "NO_SLIP_EPS": 0.005,
        }
    )

    @property
    def contact_sequence(self) -> np.ndarray:
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
