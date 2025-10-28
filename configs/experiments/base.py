from dataclasses import dataclass

import numpy as np


@dataclass
class BaseExperiment:
    # Task parameters
    initial_qpos: np.ndarray
    initial_qvel: np.ndarray
    terrain: str = "flat"

    # Physical parameters
    mu_ground: float = 0.5
    gravity_constant: float = 9.81

    # Simulation parameters
    duration: float = 1.0
    sim_dt: float = 0.01
    render: bool = True

    @property
    def contact_sequence(self) -> np.ndarray:
        raise NotImplementedError
