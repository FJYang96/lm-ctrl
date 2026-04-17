"""Go2 OPT-Mimic tracking environment configuration for Isaac Lab.

Matches the MJX implementation exactly: 1kHz physics, 50Hz control (decimation=20),
explicit torque actuation with stiffness=0/damping=0, and ContactSensor for termination.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


@configclass
class Go2TrackingEnvCfg(DirectRLEnvCfg):
    """Configuration for Go2 OPT-Mimic trajectory tracking."""

    # --- Env ---
    decimation: int = 20  # 50Hz control / 1kHz physics = 20 substeps (matches N_SUBSTEPS)
    episode_length_s: float = 10.0  # overridden by trajectory length at runtime
    action_space: int = 12
    observation_space: int = 33  # OPT-Mimic: quat(4)+joints(12)+ang_vel(3)+joint_vel(12)+phase(2)
    state_space: int = 0

    # --- Simulation ---
    sim: SimulationCfg = SimulationCfg(
        dt=0.001,  # 1kHz physics (matches MJX sim_dt)
        render_interval=20,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            solver_type=1,  # TGS
            bounce_threshold_velocity=0.2,
            gpu_found_lost_pairs_capacity=2**21,
            gpu_total_aggregate_pairs_capacity=2**21,
        ),
    )

    # --- Terrain ---
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # --- Scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # --- Robot ---
    # Use Isaac Lab's Go2 USD but override actuators for explicit torque control.
    # stiffness=0, damping=0 means set_joint_effort_target sends raw torques (no implicit PD).
    # This matches MJX's data.replace(ctrl=torque) where ctrl are raw torques.
    robot: ArticulationCfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=6,
                solver_velocity_iteration_count=6,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            joint_pos={
                ".*L_hip_joint": 0.0,
                ".*R_hip_joint": 0.0,
                ".*_thigh_joint": 1.0,
                ".*_calf_joint": -2.1,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=45.43,  # max across all joints (calf); per-joint limits applied manually
                velocity_limit=30.0,
                stiffness=0.0,  # NO implicit PD — we compute PD manually
                damping=0.0,   # NO implicit PD — we compute PD manually
            ),
        },
    )
    robot.prim_path = "/World/envs/env_.*/Robot"

    # --- Contact Sensor ---
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.0,  # every physics step
        track_air_time=False,
    )

    # --- Trajectory data paths (set at runtime) ---
    state_traj_path: str = ""
    grf_traj_path: str = ""
    joint_vel_traj_path: str = ""
    contact_sequence_path: str = ""
