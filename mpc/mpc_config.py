from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import casadi as cs
import numpy as np

from .dynamics.model import KinoDynamic_Model


@dataclass
class MPCConfig:
    # Simulation parameters
    duration: float
    mpc_dt: float

    # Cost weights
    q_base: np.ndarray
    q_joint: np.ndarray
    r_joint_vel: np.ndarray
    r_forces: np.ndarray
    q_terminal_base: np.ndarray
    q_terminal_joint: np.ndarray

    # Reference trajectory
    ref_state: np.ndarray | None
    ref_input: np.ndarray | None

    # Warmstart trajectory
    warmstart_state: np.ndarray | None
    warmstart_input: np.ndarray | None

    # Path constraints
    path_constraints: list[
        Callable[
            [cs.MX, cs.MX, KinoDynamic_Model, Any, cs.MX], tuple[cs.MX, cs.MX, cs.MX]
        ]
    ]

    # Contact sequence
    _contact_sequence: np.ndarray | None  # Stored in this field for caching

    @property
    def contact_sequence(self) -> np.ndarray:
        # generates the contact sequence based on other configuration parameters
        raise NotImplementedError


@dataclass
class HoppingMPCConfig(MPCConfig):
    pre_flight_stance_duration: float = 0.3
    flight_duration: float = 0.4

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
