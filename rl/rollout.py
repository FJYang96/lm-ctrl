"""Policy rollout — Flax/JAX policy on CPU MuJoCo for rendering.

Returns the same (qpos_traj, qvel_traj, grf_traj, images) format.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import mujoco
import numpy as np

import go2_config as config

from .feedforward import FeedforwardComputer
from .ppo import NormalizerState, normalize_obs
from .reference import ReferenceTrajectory
from .tracking_env import Go2TrackingEnv


def execute_policy_rollout(
    state_traj: np.ndarray,
    grf_traj: np.ndarray,
    joint_vel_traj: np.ndarray,
    kindyn_model: Any,
    params: Any,
    apply_fn: Any,
    normalizer: NormalizerState | None = None,
    render: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Roll out trained tracking policy on an MPC trajectory.

    Uses CPU MuJoCo for physics + rendering (MJX can't render).
    Same return format: (qpos_traj, qvel_traj, grf_traj, images).
    """
    sim_dt = 0.001
    control_dt = config.mpc_config.mpc_dt
    _substeps = int(control_dt / sim_dt)

    ref = ReferenceTrajectory(
        state_traj=state_traj,
        joint_vel_traj=joint_vel_traj,
        grf_traj=grf_traj,
        control_dt=control_dt,
    )
    ff = FeedforwardComputer(kindyn_model)
    ref.set_feedforward(ff.precompute_trajectory(ref))

    env = Go2TrackingEnv(ref=ref, sim_dt=sim_dt, control_dt=control_dt, randomize=False)

    obs = env.reset()

    renderer = None
    if render:
        try:
            renderer = mujoco.Renderer(env.mjModel, height=480, width=640)
        except Exception:
            print("Warning: Could not create renderer, skipping video capture")

    _KP = Go2TrackingEnv.KP
    _KD = Go2TrackingEnv.KD
    _TORQUE_LIMITS = Go2TrackingEnv.TORQUE_LIMITS

    num_policy_steps = ref.max_phase
    qpos_traj_out = []
    qvel_traj_out = []
    grf_traj_out = []
    images = []

    for _phase in range(num_policy_steps):
        obs_jnp = jnp.array(obs)
        if normalizer is not None:
            obs_jnp = normalize_obs(normalizer, obs_jnp)

        mean, _, _ = apply_fn(params, obs_jnp)
        action = np.array(mean)

        obs, reward, terminated, truncated, info = env.step(action)

        # Record state from the underlying MuJoCo env
        sim_obs = env._quad_env._get_obs()
        qpos_traj_out.append(sim_obs["qpos"].copy())
        qvel_traj_out.append(sim_obs["qvel"].copy())
        if "contact_forces:base" in sim_obs:
            grf_traj_out.append(sim_obs["contact_forces:base"].copy())
        else:
            grf_traj_out.append(np.zeros(12))

        if renderer is not None:
            try:
                renderer.update_scene(env.mjData)
                image = renderer.render()
                images.append(image)
            except Exception:
                pass

        if terminated or truncated:
            break

    if renderer is not None:
        try:
            renderer.close()
        except Exception:
            pass

    env.close()

    return (
        np.array(qpos_traj_out) if qpos_traj_out else np.zeros((0, 19)),
        np.array(qvel_traj_out) if qvel_traj_out else np.zeros((0, 18)),
        np.array(grf_traj_out) if grf_traj_out else np.zeros((0, 12)),
        images,
    )
