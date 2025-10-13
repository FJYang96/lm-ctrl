import pathlib

import numpy as np
from gym_quadruped.robot_cfgs import RobotConfig, get_robot_config

import mpc.constraints.constraints as constr

# ----------------------------------------------------------------------------------------------------------------
# Path constraints
# ----------------------------------------------------------------------------------------------------------------
path_constraints = [
    constr.friction_cone_constraints,
    constr.foot_height_constraints,
    constr.foot_velocity_constraints,
    constr.joint_limits_constraints,
    constr.input_limits_constraints,
]

# ----------------------------------------------------------------------------------------------------------------
# Select the robot
robot = "go2"  # 'aliengo', 'go1', 'go2', 'b2', 'hyqreal1', 'hyqreal2', 'mini_cheetah'
duration = 1.0  # total duration in seconds
mpc_dt = 0.1  # time step in seconds
sim_dt = 0.01  # time step in seconds
# ----------------------------------------------------------------------------------------------------------------

mpc_params = {
    "horizon": int(duration / mpc_dt),
    "dt": mpc_dt,
    "qp_solver": "PARTIAL_CONDENSING_HPIPM",
    "hessian_approx": "GAUSS_NEWTON",
    "integrator_type": "ERK",  # "IRK", "ERK"
    "use_RTI": False,  # use RTI
    "nlp_solver_type": "SQP",  # "SQP", "SQP_RTI"
    "nlp_solver_max_iter": 5000,  # number of iterations
    "compile_dir": str(pathlib.Path(__file__).parent / "c_generated_code"),
    "q_base": np.diag([10, 10, 50, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    * 1.0,  # Penalize CoM and orientation errors
    "q_joint": np.eye(12) * 0.1,  # Penalize joint angle errors
    "r_joint_vel": np.eye(12) * 1e-4,  # Penalize joint velocity
    "r_forces": np.eye(12) * 1e-5,  # Penalize ground reaction forces
    "q_terminal_base": np.eye(12) * 10.0,  # Strongly penalize final base error
    "q_terminal_joint": np.eye(12) * 0.5,  # Penalize final joint error
    "grf_max": 500,  # maximum ground reaction force
    "grf_min": 0,  # minimum ground reaction force
    "mu": 0.5,  # friction coefficient
    # --- Below not used for now ---
    "use_integrators": False,  # use integrators
    "use_warm_start": False,  # use warm start
    "external_wrenches_compensation": False,
    "use_nonuniform_discretization": False,
    "external_wrenches_compensation_num_step": 0,
    "solver_mode": "balance",
    "num_qp_iterations": 100,
    "as_rti_type": "AS-RTI-C",
    "trot_stability_margin": 0.05,
    "pace_stability_margin": 0.05,
    "crawl_stability_margin": 0.05,
    "use_static_stability": False,
    "use_zmp_stability": True,
    "use_foothold_constraints": False,
    "use_DDP": False,
}

sim_params = {
    "robot": robot,
    "terrain_type": "flat",
    "sim_dt": sim_dt,
    "sim_duration": duration,
    "ground_friction_coeff": (0.2, 1.5),
    "base_vel_command_type": "forward",
}

from typing import Any

import numpy as np

from configs.experiments import BaseExperiment
from configs.robots import RobotData, get_robot_data
from mpc.constraints import constraints as constr
from mpc.mpc_config import HoppingMPCConfig, MPCConfig

robot: str = "go2"
duration: float = 1.0
robot_data: RobotData = get_robot_data(robot)

experiment: BaseExperiment = BaseExperiment(
    mu_ground=0.5,
    gravity_constant=9.81,
    duration=1.0,
    sim_dt=0.01,
    terrain="flat",
    initial_qpos=robot_data.initial_qpos,
    initial_qvel=robot_data.initial_qvel,
)

mpc_config: MPCConfig = HoppingMPCConfig(
    duration=duration,
    mpc_dt=0.1,
    pre_flight_stance_duration=0.3,  # Used for finding contact sequence
    flight_duration=0.4,  # Used for finding contact sequence
    q_base=np.diag([10, 10, 50, 1, 1, 1, 1, 1, 1, 1, 1, 1]) * 1.0,
    q_joint=np.eye(12) * 0.1,
    r_joint_vel=np.eye(12) * 1e-4,
    r_forces=np.eye(12) * 1e-5,
    q_terminal_base=np.eye(12) * 10.0,
    q_terminal_joint=np.eye(12) * 0.5,
    ref_state=None,
    ref_input=None,
    path_constraints=[
        constr.friction_cone_constraints,
        constr.foot_height_constraints,
        constr.foot_velocity_constraints,
        constr.joint_limits_constraints,
        constr.input_limits_constraints,
    ],
    warmstart_state=None,
    warmstart_input=None,
    _contact_sequence=None,
)

solver_config: dict[str, Any] = {
    "ipopt.print_level": 5,
    "print_time": True,
    "ipopt.max_iter": 1000,
    "ipopt.tol": 1e-4,
    "ipopt.acceptable_tol": 1e-3,
    "ipopt.mu_init": 1e-2,
    "ipopt.mu_strategy": "adaptive",
    "ipopt.alpha_for_y": "primal",
    "ipopt.recalc_y": "yes",
}
