"""Default MPC configuration — constructed from go2_config data values."""

import numpy as np

import go2_config
import mpc.constraints as constr
from mpc.config_complementarity import ComplementarityMPCConfig
from mpc.mpc_config import HoppingMPCConfig

if go2_config.CONSTRAINT_MODE == "complementarity":
    mpc_config = ComplementarityMPCConfig(
        duration=go2_config.duration,
        mpc_dt=go2_config.default_mpc_dt_complementarity,
        pre_flight_stance_duration=go2_config.default_pre_flight_stance_duration,
        flight_duration=go2_config.default_flight_duration,
        q_base=np.diag([10, 10, 50, 1, 1, 1, 1, 1, 1, 1, 1, 1]) * 1.0,
        q_joint=np.eye(12) * 0.1,
        r_joint_vel=np.eye(12) * 1e-4,
        r_forces=np.eye(12) * 1e-5,
        q_terminal_base=np.eye(12) * 10.0,
        q_terminal_joint=np.eye(12) * 0.5,
        ref_state=None,
        ref_input=None,
        path_constraint_params={
            "COMPLEMENTARITY_EPS": 1e-3,
            "SWING_GRF_EPS": 0.0,
            "STANCE_HEIGHT_EPS": 0.04,
            "NO_SLIP_EPS": 0.01,
            "BODY_CLEARANCE_MIN": 0.02,
        },
        _contact_sequence=None,
    )
else:
    mpc_config = HoppingMPCConfig(
        duration=go2_config.duration,
        mpc_dt=go2_config.default_mpc_dt_standard,
        pre_flight_stance_duration=go2_config.default_pre_flight_stance_duration,
        flight_duration=go2_config.default_flight_duration,
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
            constr.body_clearance_constraints,
        ],
        path_constraint_params={
            "SWING_GRF_EPS": 0.0,
            "STANCE_HEIGHT_EPS": 0.02,
            "NO_SLIP_EPS": 0.005,
            "BODY_CLEARANCE_MIN": 0.02,
        },
        _contact_sequence=None,
    )
