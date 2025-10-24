import os

import gym_quadruped
import numpy as np
from gym_quadruped.robot_cfgs import get_robot_config

from .robot_data import RobotData

go2_stand_qpos = np.zeros(19)
go2_stand_qpos[0:3] = [0.0, 0.0, 0.2117]
go2_stand_qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
go2_stand_qpos[7:19] = [
    0.0,
    1.0,
    -2.1,
    0.0,
    1.0,
    -2.1,
    0.0,
    1.0,
    -2.1,
    0.0,
    1.0,
    -2.1,
]
go2_stand_qvel = np.zeros(18)

go2 = RobotData(
    name="go2",
    mass=15.019,
    inertia=np.array(
        [
            [1.58460467e-01, 1.21660000e-04, -1.55444692e-02],
            [1.21660000e-04, 4.68645637e-01, -3.12000000e-05],
            [-1.55444692e-02, -3.12000000e-05, 5.24474661e-01],
        ]
    ),
    urdf_filename=os.path.dirname(gym_quadruped.__file__) + "/robot_model/go2/go2.urdf",
    xml_filename=os.path.dirname(gym_quadruped.__file__) + "/robot_model/go2/go2.xml",
    joint_limits_lower=np.array(
        [-0.8, -1.6, -2.6, -0.8, -1.6, -2.6, -0.8, -1.6, -2.6, -0.8, -1.6, -2.6]
    ),
    joint_limits_upper=np.array(
        [0.8, 1.6, -0.5, 0.8, 1.6, -0.5, 0.8, 1.6, -0.5, 0.8, 1.6, -0.5]
    ),
    robot_cfg=get_robot_config(robot_name="go2"),
    joint_velocity_limits=np.ones(12) * 10.0,
    grf_limits=500.0,
    initial_qpos=go2_stand_qpos,
    initial_qvel=go2_stand_qvel,
)
