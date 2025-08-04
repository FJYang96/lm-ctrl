import numpy as np
from gym_quadruped.robot_cfgs import RobotConfig, get_robot_config

# ----------------------------------------------------------------------------------------------------------------
# Select the robot
robot = 'go2'  # 'aliengo', 'go1', 'go2', 'b2', 'hyqreal1', 'hyqreal2', 'mini_cheetah'  # TODO: Load from robot_descriptions.py
# ----------------------------------------------------------------------------------------------------------------

robot_cfg: RobotConfig = get_robot_config(robot_name=robot)
robot_leg_joints = robot_cfg.leg_joints
robot_feet_geom_names = robot_cfg.feet_geom_names
qpos0_js = robot_cfg.qpos0_js
hip_height = robot_cfg.hip_height

# ----------------------------------------------------------------------------------------------------------------
if (robot == 'go1'):
    mass = 12.019
    inertia = np.array([[1.58460467e-01, 1.21660000e-04, -1.55444692e-02],
                        [1.21660000e-04, 4.68645637e-01, -3.12000000e-05],
                        [-1.55444692e-02, -3.12000000e-05, 5.24474661e-01]])

elif (robot == 'go2'):
    mass = 15.019
    inertia = np.array([[1.58460467e-01, 1.21660000e-04, -1.55444692e-02],
                        [1.21660000e-04, 4.68645637e-01, -3.12000000e-05],
                        [-1.55444692e-02, -3.12000000e-05, 5.24474661e-01]])

elif (robot == 'aliengo'):
    mass = 24.637
    inertia = np.array([[0.2310941359705289, -0.0014987128245817424, -0.021400468992761768],
                        [-0.0014987128245817424, 1.4485084687476608, 0.0004641447134275615],
                        [-0.021400468992761768, 0.0004641447134275615, 1.503217877350808]])

elif (robot == 'b2'):
    mass = 83.49
    inertia = np.array([[0.2310941359705289, -0.0014987128245817424, -0.021400468992761768],
                        [-0.0014987128245817424, 1.4485084687476608, 0.0004641447134275615],
                        [-0.021400468992761768, 0.0004641447134275615, 1.503217877350808]])


elif (robot == 'hyqreal1'):
    mass = 108.40
    inertia = np.array([[4.55031444e+00, 2.75249434e-03, -5.11957307e-01],
                        [2.75249434e-03, 2.02411774e+01, -7.38560592e-04],
                        [-5.11957307e-01, -7.38560592e-04, 2.14269772e+01]])
    
elif (robot == 'hyqreal2'):
    mass = 126.69
    inertia = np.array([[4.55031444e+00, 2.75249434e-03, -5.11957307e-01],
                        [2.75249434e-03, 2.02411774e+01, -7.38560592e-04],
                        [-5.11957307e-01, -7.38560592e-04, 2.14269772e+01]])
    
elif (robot == 'mini_cheetah'):
    mass = 12.5
    inertia = np.array([[1.58460467e-01, 1.21660000e-04, -1.55444692e-02],
                        [1.21660000e-04, 4.68645637e-01, -3.12000000e-05],
                        [-1.55444692e-02, -3.12000000e-05, 5.24474661e-01]])


gravity_constant = 9.81 # Exposed in case of different gravity conditions

mpc_params = {
    'horizon': 20,
    'dt': 0.05,
    'use_RTI': False,
    'use_integrators': False,
    'use_warm_start': False,
    'grf_max': 500,
    'grf_min': 0,
    'mu': 0.5,
    'external_wrenches_compensation': False,
    'use_nonuniform_discretization': False,
    'external_wrenches_compensation_num_step': 0,
    'solver_mode': 'balance',
    'num_qp_iterations': 100,
    'as_rti_type': 'AS-RTI-C',
    'trot_stability_margin': 0.05,
    'pace_stability_margin': 0.05,
    'crawl_stability_margin': 0.05,
    'use_static_stability': False,
    'use_zmp_stability': True,
    'use_foothold_constraints': False,
    'use_DDP': False
}