"""Policy rollout — drop-in replacement for inv_dyn + simulate_trajectory.

Returns the same (qpos_traj, qvel_traj, grf_traj, images) format.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import config

from .feedforward import FeedforwardComputer
from .reference import ReferenceTrajectory
from .tracking_env import Go2TrackingEnv

if TYPE_CHECKING:
    from gym_quadruped.quadruped_env import QuadrupedEnv

    from mpc.dynamics.model import KinoDynamic_Model

DEFAULT_MODEL_PATH = (
    Path(__file__).parent / "trained_models" / "best_model" / "best_model.zip"
)
DEFAULT_NORMALIZE_PATH = Path(__file__).parent / "trained_models" / "vec_normalize.pkl"


def execute_policy_rollout(
    env: QuadrupedEnv,
    state_traj: np.ndarray,
    grf_traj: np.ndarray,
    joint_vel_traj: np.ndarray,
    kindyn_model: KinoDynamic_Model,
    policy: PPO,
    vec_normalize_path: str | Path | None = None,
    planned_traj_images: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Roll out trained tracking policy on an MPC trajectory.

    This replaces compute_joint_torques() + simulate_trajectory().
    Same return format: (qpos_traj, qvel_traj, grf_traj, images).
    """
    sim_dt = config.experiment.sim_dt
    control_dt = config.mpc_config.mpc_dt
    substeps = int(control_dt / sim_dt)

    # Build reference + feedforward
    ref = ReferenceTrajectory(
        state_traj=state_traj,
        joint_vel_traj=joint_vel_traj,
        grf_traj=grf_traj,
        control_dt=control_dt,
    )
    ff = FeedforwardComputer(kindyn_model)
    ref.set_feedforward(ff.precompute_trajectory(ref))

    # Load VecNormalize stats if available (for observation normalization)
    normalize_path = vec_normalize_path or DEFAULT_NORMALIZE_PATH
    obs_normalizer = None
    if Path(normalize_path).exists():
        # Create a dummy VecEnv to load normalization stats
        dummy_env = DummyVecEnv(
            [
                lambda: Go2TrackingEnv(
                    ref=ref, sim_dt=sim_dt, control_dt=control_dt, randomize=False
                )
            ]
        )
        obs_normalizer = VecNormalize.load(str(normalize_path), dummy_env)
        obs_normalizer.training = False
        obs_normalizer.norm_reward = False

    # Reset env to reference initial state
    init_qpos = np.zeros(19)
    init_qpos[0:3] = ref.get_body_pos(0)
    init_qpos[3:7] = ref.get_body_quat(0)
    init_qpos[7:19] = ref.get_joint_pos(0)

    init_qvel = np.zeros(18)
    init_qvel[0:3] = ref.get_body_vel(0)
    init_qvel[3:6] = ref.get_body_ang_vel(0)
    init_qvel[6:18] = ref.get_joint_vel(0)

    env.reset(qpos=init_qpos, qvel=init_qvel)

    # Offscreen renderer for video capture (no GUI needed)
    renderer = None
    if config.experiment.render:
        renderer = mujoco.Renderer(env.mjModel, height=480, width=640)

    # PD gains (must match training)
    KP = Go2TrackingEnv.KP
    KD = Go2TrackingEnv.KD
    TORQUE_LIMIT = Go2TrackingEnv.TORQUE_LIMIT

    # Rollout loop
    num_policy_steps = ref.max_phase
    qpos_traj_out = []
    qvel_traj_out = []
    grf_traj_out = []
    images = []
    # Build sensor/action history buffers (matching training env)
    sim_obs = env._get_obs()
    init_sensor = np.concatenate(
        [sim_obs["qpos"][3:7], sim_obs["qpos"][7:19], sim_obs["qvel"][6:18]]
    )
    sensor_history = [init_sensor.copy() for _ in range(3)]
    action_history = [np.zeros(12) for _ in range(3)]

    for phase in range(num_policy_steps):
        # Build observation (same as tracking_env._build_obs)
        sim_obs = env._get_obs()
        current_sensor = np.concatenate(
            [sim_obs["qpos"][3:7], sim_obs["qpos"][7:19], sim_obs["qvel"][6:18]]
        )
        phase_enc = ref.get_phase_encoding(phase)
        obs = np.concatenate(
            [
                current_sensor,
                phase_enc,
                np.concatenate(sensor_history),
                np.concatenate(action_history),
            ]
        ).astype(np.float32)

        # Normalize observation if normalizer available
        if obs_normalizer is not None:
            obs = obs_normalizer.normalize_obs(obs)

        # Query policy
        action, _ = policy.predict(obs, deterministic=True)
        action = action * Go2TrackingEnv.ACTION_LIMIT

        # PD + feedforward
        ref_joint_pos = ref.get_joint_pos(phase)
        ref_joint_vel = ref.get_joint_vel(phase)
        ff = ref.get_feedforward_torque(phase)

        target_pos = ref_joint_pos + action
        actual_joint_pos = sim_obs["qpos"][7:19]
        actual_joint_vel = sim_obs["qvel"][6:18]

        torque = (
            KP * (target_pos - actual_joint_pos)
            + KD * (ref_joint_vel - actual_joint_vel)
            + ff
        )
        torque = np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT)

        # Step MuJoCo (substeps per policy step)
        for _ in range(substeps):
            sim_obs, _, _, _, _ = env.step(action=torque)
            qpos_traj_out.append(sim_obs["qpos"].copy())
            qvel_traj_out.append(sim_obs["qvel"].copy())
            grf_traj_out.append(sim_obs["contact_forces:base"].copy())

            if renderer is not None:
                renderer.update_scene(env.mjData)
                image = renderer.render()
                images.append(image)

        # Update histories
        sensor_history.pop(0)
        sensor_history.append(current_sensor.copy())
        action_history.pop(0)
        action_history.append(action.copy())

    if renderer is not None:
        renderer.close()
    if obs_normalizer is not None:
        obs_normalizer.venv.close()

    return (
        np.array(qpos_traj_out),
        np.array(qvel_traj_out),
        np.array(grf_traj_out),
        images,
    )
