import dataclasses

import numpy as np
from gym_quadruped.robot_cfgs import RobotConfig


@dataclasses.dataclass
class RobotData:
    name: str
    mass: float
    inertia: np.ndarray  # (3, 3)
    urdf_filename: str
    xml_filename: str
    joint_limits_lower: np.ndarray  # (12,) FL, FR, RL, RR
    joint_limits_upper: np.ndarray  # (12,) FL, FR, RL, RR
    robot_cfg: RobotConfig
    joint_velocity_limits: np.ndarray  # (12,) FL, FR, RL, RR
    grf_limits: float
    initial_qpos: np.ndarray  # (19,)
    initial_qvel: np.ndarray  # (18,)
