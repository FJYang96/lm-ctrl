import pathlib

import numpy as np
from gym_quadruped.robot_cfgs import RobotConfig, get_robot_config
import mpc.constraints.constraints as constr

#----------------------------------------------------------------------------------------------------------------
# Path constraints
#----------------------------------------------------------------------------------------------------------------
path_constraints= [
    constr.friction_cone_constraints,
    constr.foot_height_constraints,
    constr.foot_velocity_constraints,
    constr.joint_limits_constraints,
    constr.input_limits_constraints,
]

# ----------------------------------------------------------------------------------------------------------------
# Select the robot
robot = "go2"  # 'aliengo', 'go1', 'go2', 'b2', 'hyqreal1', 'hyqreal2', 'mini_cheetah'
duration = 3.0  # total duration in seconds
mpc_dt = 0.1  # time step in seconds
sim_dt = 0.01  # time step in seconds
# ----------------------------------------------------------------------------------------------------------------

robot_cfg: RobotConfig = get_robot_config(robot_name=robot)
robot_leg_joints = robot_cfg.leg_joints
robot_feet_geom_names = robot_cfg.feet_geom_names
qpos0_js = robot_cfg.qpos0_js
hip_height = robot_cfg.hip_height

# ----------------------------------------------------------------------------------------------------------------
if robot == "go1":
    mass = 12.019
    inertia = np.array(
        [
            [1.58460467e-01, 1.21660000e-04, -1.55444692e-02],
            [1.21660000e-04, 4.68645637e-01, -3.12000000e-05],
            [-1.55444692e-02, -3.12000000e-05, 5.24474661e-01],
        ]
    )

elif robot == "go2":
    mass = 15.019
    inertia = np.array(
        [
            [1.58460467e-01, 1.21660000e-04, -1.55444692e-02],
            [1.21660000e-04, 4.68645637e-01, -3.12000000e-05],
            [-1.55444692e-02, -3.12000000e-05, 5.24474661e-01],
        ]
    )

elif robot == "aliengo":
    mass = 24.637
    inertia = np.array(
        [
            [0.2310941359705289, -0.0014987128245817424, -0.021400468992761768],
            [-0.0014987128245817424, 1.4485084687476608, 0.0004641447134275615],
            [-0.021400468992761768, 0.0004641447134275615, 1.503217877350808],
        ]
    )

elif robot == "b2":
    mass = 83.49
    inertia = np.array(
        [
            [0.2310941359705289, -0.0014987128245817424, -0.021400468992761768],
            [-0.0014987128245817424, 1.4485084687476608, 0.0004641447134275615],
            [-0.021400468992761768, 0.0004641447134275615, 1.503217877350808],
        ]
    )


elif robot == "hyqreal1":
    mass = 108.40
    inertia = np.array(
        [
            [4.55031444e00, 2.75249434e-03, -5.11957307e-01],
            [2.75249434e-03, 2.02411774e01, -7.38560592e-04],
            [-5.11957307e-01, -7.38560592e-04, 2.14269772e01],
        ]
    )

elif robot == "hyqreal2":
    mass = 126.69
    inertia = np.array(
        [
            [4.55031444e00, 2.75249434e-03, -5.11957307e-01],
            [2.75249434e-03, 2.02411774e01, -7.38560592e-04],
            [-5.11957307e-01, -7.38560592e-04, 2.14269772e01],
        ]
    )

elif robot == "mini_cheetah":
    mass = 12.5
    inertia = np.array(
        [
            [1.58460467e-01, 1.21660000e-04, -1.55444692e-02],
            [1.21660000e-04, 4.68645637e-01, -3.12000000e-05],
            [-1.55444692e-02, -3.12000000e-05, 5.24474661e-01],
        ]
    )


gravity_constant = 9.81  # Exposed in case of different gravity conditions
mu_friction = 0.5

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
    "q_base": np.diag([10, 10, 50, 1, 1, 1, 1, 1, 1, 1, 1, 1]) * 1.0, # Penalize CoM and orientation errors
    "q_joint": np.eye(12) * 0.1,                                     # Penalize joint angle errors
    "r_joint_vel": np.eye(12) * 1e-4,                                # Penalize joint velocity
    "r_forces": np.eye(12) * 1e-5,                                   # Penalize ground reaction forces
    "q_terminal_base": np.eye(12) * 10.0,                            # Strongly penalize final base error
    "q_terminal_joint": np.eye(12) * 0.5,                            # Penalize final joint error
    "grf_max": 500,  # maximum ground reaction force
    "grf_min": 0,  # minimum ground reaction force
    "mu": mu_friction,  # friction coefficient
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

# ----------------------------------------------------------------------------------------------------------------
# Define the gait / contact sequence
# ----------------------------------------------------------------------------------------------------------------
pre_flight_stance_duration = 0.3  # seconds to crouch and push off
flight_duration = 0.4             # seconds in the air
post_flight_stance_duration = 0.3 # seconds to absorb landing

# --- UPDATE THE TOTAL DURATION TO MATCH THE SEQUENCE ---
duration = pre_flight_stance_duration + flight_duration + post_flight_stance_duration

# Recalculate horizon based on new duration
mpc_params["horizon"] = int(duration / mpc_dt)
sim_params["sim_duration"] = duration

# Define the number of MPC steps for each phase
pre_flight_steps = int(pre_flight_stance_duration / mpc_dt)
flight_steps = int(flight_duration / mpc_dt)
# The remaining steps will be for the post-flight stance

# Create the contact sequence: STANCE -> FLIGHT -> STANCE
contact_sequence = np.ones((4, mpc_params["horizon"])) # Start with all feet in contact

# Set the middle section to 0 for the flight phase
contact_sequence[:, pre_flight_steps : pre_flight_steps + flight_steps] = 0.0

# ----------------------------------------------------------------------------------------------------------------
# Define the initial state
# ----------------------------------------------------------------------------------------------------------------
initial_qpos = np.zeros(19)
initial_qpos[0:3] = [0.0, 0.0, 0.23]  # base position
initial_qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # base orientation (quaternion)
initial_qpos[7:19] = [  # joint angles
    0.0,
    1.0,
    -2.1,  # FL: no abd, folded hip/knee
    0.0,
    1.0,
    -2.1,  # FR
    0.0,
    1.0,
    -2.1,  # RL
    0.0,
    1.0,
    -2.1,  # RR
]
initial_qvel = np.zeros(18)

# Basic joint limits for Go2 robot (approximate)
# Hip: ±45°, Thigh: -90° to +90°, Calf: -150° to -30°
joint_limits_lower = np.array(
    [
        -0.8,
        -1.6,
        -2.6,  # FL: hip, thigh, calf
        -0.8,
        -1.6,
        -2.6,  # FR
        -0.8,
        -1.6,
        -2.6,  # RL
        -0.8,
        -1.6,
        -2.6,  # RR
    ]
)
joint_limits_upper = np.array(
    [
        0.8,
        1.6,
        -0.5,  # FL: hip, thigh, calf
        0.8,
        1.6,
        -0.5,  # FR
        0.8,
        1.6,
        -0.5,  # RL
        0.8,
        1.6,
        -0.5,  # RR
    ]
)

joint_velocity_limits = 10
grf_limits = 300