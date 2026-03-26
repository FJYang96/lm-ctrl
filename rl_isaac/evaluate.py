"""Evaluate trained Isaac Lab OPT-Mimic policy.

Uses a CPU MuJoCo environment for rollout and video rendering.
Does NOT import JAX — runs entirely in PyTorch + MuJoCo.

Usage:
    python -m rl_isaac.evaluate \
        --model-path rl_isaac/trained_models/run_50M/best_model \
        --state-traj results/llm_iterations/.../state_traj_iter_7.npy \
        --grf-traj results/llm_iterations/.../grf_traj_iter_7.npy \
        --joint-vel-traj results/llm_iterations/.../joint_vel_traj_iter_7.npy
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import os
import sys
import types

# Block broken GLFW and mujoco.viewer so gym_quadruped doesn't crash on headless Docker.
os.environ.setdefault("MUJOCO_GL", "egl")
if "mujoco.viewer" not in sys.modules:
    _fake_viewer = types.ModuleType("mujoco.viewer")
    _fake_viewer.Handle = type("Handle", (), {})
    sys.modules["mujoco.viewer"] = _fake_viewer
if "glfw" not in sys.modules:
    _fake_glfw = types.ModuleType("glfw")
    _fake_glfw._glfw = True
    sys.modules["glfw"] = _fake_glfw
    sys.modules["glfw.library"] = types.ModuleType("glfw.library")

import mujoco
import numpy as np
import torch

import go2_config
from mpc.dynamics.model import KinoDynamic_Model
from rl_isaac.feedforward import FeedforwardComputer
from rl_isaac.reference import ReferenceTrajectory
from utils.conversion import sim_to_mpc

from rl_isaac.network import OPTMimicActorCritic

# OPT-Mimic constants (same as tracking_env.py)
KP = 25.0
KD = 1.5
TORQUE_LIMITS = np.array([23.7, 23.7, 45.43] * 4, dtype=np.float32)
ACTION_LIMIT = 0.2


def load_isaac_checkpoint(model_path: str, device: str = "cpu"):
    """Load Isaac Lab checkpoint."""
    ckpt_file = Path(model_path) / "checkpoint.pt"
    if not ckpt_file.exists():
        ckpt_file = Path(model_path)
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    actor_critic = OPTMimicActorCritic(num_obs=39, num_privileged_obs=0, num_actions=12)
    actor_critic.load_state_dict(ckpt["model_state_dict"])
    actor_critic.eval()
    normalizer_state = ckpt.get("normalizer_state_dict", None)
    step = ckpt.get("step", 0)
    return actor_critic, normalizer_state, step


def execute_rollout(
    state_traj: np.ndarray,
    grf_traj: np.ndarray,
    joint_vel_traj: np.ndarray,
    actor_critic,
    normalizer_state: dict | None = None,
    contact_sequence: np.ndarray | None = None,
    render: bool = True,
):
    """Roll out the trained policy on CPU MuJoCo for evaluation + video.

    Uses gym_quadruped directly (no JAX dependency).
    """
    sim_dt = 0.001
    control_dt = go2_config.mpc_config.mpc_dt
    substeps = int(control_dt / sim_dt)

    kindyn = KinoDynamic_Model()
    ref = ReferenceTrajectory(
        state_traj=state_traj,
        joint_vel_traj=joint_vel_traj,
        grf_traj=grf_traj,
        contact_sequence=contact_sequence,
        control_dt=control_dt,
    )
    ff = FeedforwardComputer(kindyn)
    ref.set_feedforward(ff.precompute_trajectory(ref))

    # Create CPU MuJoCo env via gym_quadruped
    from gym_quadruped.quadruped_env import QuadrupedEnv
    quad_env = QuadrupedEnv(
        robot="go2", scene="flat",
        ground_friction_coeff=0.8,
        state_obs_names=QuadrupedEnv._DEFAULT_OBS + ("contact_forces:base",),
        sim_dt=sim_dt,
    )

    # Reset to reference state at phase 0
    init_qpos = np.zeros(19)
    init_qpos[0:3] = ref.get_body_pos(0)
    init_qpos[3:7] = ref.get_body_quat(0)
    init_qpos[7:19] = ref.get_joint_pos(0)
    init_qvel = np.zeros(18)
    init_qvel[0:3] = ref.get_body_vel(0)
    init_qvel[3:6] = ref.get_body_ang_vel(0)
    init_qvel[6:18] = ref.get_joint_vel(0)
    quad_env.reset(qpos=init_qpos, qvel=init_qvel)

    # Normalizer
    norm_mean = None
    norm_var = None
    if normalizer_state is not None:
        norm_mean = normalizer_state.get("_mean", None)
        norm_var = normalizer_state.get("_var", None)
        if norm_mean is not None:
            norm_mean = norm_mean.float()
        if norm_var is not None:
            norm_var = norm_var.float()

    renderer = None
    if render:
        try:
            renderer = mujoco.Renderer(quad_env.mjModel, height=480, width=640)
        except Exception:
            print("Warning: Could not create renderer")

    num_policy_steps = ref.max_phase
    qpos_traj = []
    qvel_traj = []
    grf_traj_out = []
    images = []
    prev_action = np.zeros(12)

    for phase in range(num_policy_steps):
        sim_obs = quad_env._get_obs()

        # Build 39D observation (same as tracking_env.py)
        body_pos = sim_obs["qpos"][0:3]
        body_quat = sim_obs["qpos"][3:7]
        joint_pos = sim_obs["qpos"][7:19]
        body_vel = sim_obs["qvel"][0:3]
        body_ang_vel = sim_obs["qvel"][3:6]
        joint_vel = sim_obs["qvel"][6:18]
        angle = 2.0 * np.pi * phase / ref.max_phase
        phase_enc = np.array([np.cos(angle), np.sin(angle)])
        obs = np.concatenate([body_pos, body_quat, joint_pos, body_vel, body_ang_vel, joint_vel, phase_enc]).astype(np.float32)

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        # Normalize
        if norm_mean is not None and norm_var is not None:
            obs_t = (obs_t - norm_mean) / torch.sqrt(norm_var + 1e-8)
            obs_t = obs_t.clamp(-10.0, 10.0)

        with torch.no_grad():
            action = actor_critic.act_inference(obs_t)
        action = action.squeeze(0).numpy()

        # Apply PD + FF actuation
        action_scaled = action * ACTION_LIMIT
        ref_joint_pos = ref.get_joint_pos(phase)
        ref_joint_vel = ref.get_joint_vel(phase)
        ff_torque = ref.get_feedforward_torque(phase)

        actual_joint_pos = sim_obs["qpos"][7:19]
        actual_joint_vel = sim_obs["qvel"][6:18]

        target_pos = ref_joint_pos + action_scaled
        torque = (
            KP * (target_pos - actual_joint_pos)
            + KD * (ref_joint_vel - actual_joint_vel)
            + ff_torque
        )
        torque = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)

        # Step physics
        for _ in range(substeps):
            sim_obs, _, _, _, _ = quad_env.step(action=torque)

        # Record
        qpos_traj.append(sim_obs["qpos"].copy())
        qvel_traj.append(sim_obs["qvel"].copy())
        if "contact_forces:base" in sim_obs:
            grf_traj_out.append(sim_obs["contact_forces:base"].copy())
        else:
            grf_traj_out.append(np.zeros(12))

        if renderer is not None:
            try:
                renderer.update_scene(quad_env.mjData)
                images.append(renderer.render())
            except Exception:
                pass

        prev_action = action_scaled.copy()

    if renderer:
        try:
            renderer.close()
        except Exception:
            pass
    quad_env.close()

    return (
        np.array(qpos_traj) if qpos_traj else np.zeros((0, 19)),
        np.array(qvel_traj) if qvel_traj else np.zeros((0, 18)),
        np.array(grf_traj_out) if grf_traj_out else np.zeros((0, 12)),
        images,
    )


def compute_tracking_error(planned_state, qpos_traj, qvel_traj):
    """RMS tracking error (same as rl/evaluate.py)."""
    sim_states = []
    for i in range(min(len(qpos_traj), planned_state.shape[0])):
        sim_state, _ = sim_to_mpc(qpos_traj[i], qvel_traj[i])
        sim_states.append(sim_state)
    if not sim_states:
        return {"pos_rms": float("nan"), "ori_rms": float("nan"), "joint_rms": float("nan")}
    sim_traj = np.array(sim_states)
    n = min(planned_state.shape[0], sim_traj.shape[0])
    return {
        "pos_rms": np.sqrt(np.mean((planned_state[:n, 0:3] - sim_traj[:n, 0:3]) ** 2)),
        "ori_rms": np.sqrt(np.mean((planned_state[:n, 6:9] - sim_traj[:n, 6:9]) ** 2)),
        "joint_rms": np.sqrt(np.mean((planned_state[:n, 12:24] - sim_traj[:n, 12:24]) ** 2)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Isaac Lab OPT-Mimic policy")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--state-traj", type=str, required=True)
    parser.add_argument("--grf-traj", type=str, required=True)
    parser.add_argument("--joint-vel-traj", type=str, required=True)
    parser.add_argument("--contact-sequence", type=str, default=None)
    parser.add_argument("--output-video", type=str, default="results/rl_isaac_tracking.mp4")
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")

    state_traj = np.load(args.state_traj)
    grf_traj = np.load(args.grf_traj)
    joint_vel_traj = np.load(args.joint_vel_traj)
    contact_seq = np.load(args.contact_sequence) if args.contact_sequence else None

    print(f"Loading model from {args.model_path}...")
    actor_critic, normalizer_state, step = load_isaac_checkpoint(args.model_path)
    print(f"  Loaded checkpoint at step {step}")

    print("Running rollout...")
    qpos_rl, qvel_rl, grf_rl, images = execute_rollout(
        state_traj, grf_traj, joint_vel_traj,
        actor_critic, normalizer_state, contact_seq,
        render=True,
    )

    n_tracked = len(qpos_rl)
    err = compute_tracking_error(state_traj, qpos_rl, qvel_rl) if n_tracked > 0 else {
        "pos_rms": float("nan"), "ori_rms": float("nan"), "joint_rms": float("nan")
    }

    print("")
    print("=" * 40)
    print("ISAAC LAB RL TRACKING ERRORS")
    print("=" * 40)
    print(f"  Steps tracked: {n_tracked}/{state_traj.shape[0] - 1}")
    print(f"  Position RMS:  {err['pos_rms']:.4f} m")
    print(f"  Orientation RMS: {err['ori_rms']:.4f} rad")
    print(f"  Joint RMS:     {err['joint_rms']:.4f} rad")
    print("=" * 40)

    if images:
        import imageio
        output_path = Path(args.output_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = int(1.0 / config.mpc_config.mpc_dt)
        imageio.mimsave(str(output_path), images, fps=fps)
        print(f"Video saved to {output_path}")
    else:
        print("No frames captured — video not saved.")


if __name__ == "__main__":
    main()
