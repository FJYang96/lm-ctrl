"""Train a per-trajectory Go2 tracking policy using JAX PPO (OPT-Mimic).

Multi-GPU accelerated training with MuJoCo MJX via jax.pmap.
Single compiled train_step (rollout + PPO) for maximum throughput.

Usage:
    python -m rl.train \
        --state-traj results/state_traj.npy \
        --grf-traj results/grf_traj.npy \
        --joint-vel-traj results/joint_vel_traj.npy \
        --output-dir rl/trained_models \
        --total-timesteps 2000000 \
        --num-envs 1024
"""

from __future__ import annotations

import argparse
import functools
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
import optax

import config
from mpc.dynamics.model import KinoDynamic_Model

from .callbacks import get_log, save_reward_curve, set_log_dir, write_training_header_jax
from .feedforward import FeedforwardComputer
from .ppo import (
    ActorCritic,
    NormalizerState,
    compute_gae,
    init_normalizer,
    normalize_obs,
    ppo_loss,
    sample_action,
    save_checkpoint,
    update_normalizer,
)
from .reference import ReferenceTrajectory
from .tracking_env import (
    get_obs,
    load_mjx_model,
    make_ref_data,
    reset_fast,
)
from .tracking_env import (
    step as env_step,
)
from .rollout import execute_policy_rollout


def build_reference(
    state_traj_path: str,
    grf_traj_path: str,
    joint_vel_traj_path: str,
    contact_sequence_path: str | None = None,
    control_dt: float = 0.02,
) -> tuple[ReferenceTrajectory, KinoDynamic_Model]:
    """Load MPC trajectory and precompute feedforward torques."""
    ref = ReferenceTrajectory.from_files(
        state_traj_path,
        joint_vel_traj_path,
        grf_traj_path,
        contact_sequence_path=contact_sequence_path,
        control_dt=control_dt,
    )
    kindyn = KinoDynamic_Model(config)
    ff = FeedforwardComputer(kindyn)
    ref.set_feedforward(ff.precompute_trajectory(ref))
    return ref, kindyn


def train(args: argparse.Namespace) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_log_dir(output_dir)
    log = get_log()

    n_devices = jax.local_device_count()
    log.info(f"JAX devices: {jax.devices()} ({n_devices} GPUs)")

    # Load MJX model
    log.info("Loading MJX model...")
    mj_model, mjx_model = load_mjx_model(sim_dt=args.sim_dt)

    # Load reference trajectory (CPU)
    log.info("Loading reference trajectory...")
    ref, kindyn = build_reference(
        args.state_traj, args.grf_traj, args.joint_vel_traj,
        contact_sequence_path=args.contact_sequence,
        control_dt=args.control_dt,
    )
    ref_data = make_ref_data(ref, mj_model)
    log.info(f"Reference: {ref.max_phase} steps, {ref.duration:.2f}s")

    # Round num_envs to multiple of n_devices
    num_envs = max(n_devices, (args.num_envs // n_devices) * n_devices)
    envs_per_device = num_envs // n_devices

    # PPO hyperparameters
    # n_steps must cover at least one full episode for good GAE credit assignment
    n_steps = ref.max_phase  # match episode length (e.g. 80)
    samples_per_update = n_steps * num_envs
    n_updates = max(1, args.total_timesteps // samples_per_update)
    n_epochs = args.n_epochs
    gamma = 0.995
    gae_lambda = 0.95
    clip_range = 0.2
    ent_coef_start = 0.002
    ent_coef_end = 0.0005
    vf_coef = 0.5
    max_grad_norm = 0.5

    # Batch size per device (must divide evenly into per-device samples)
    per_device_samples = n_steps * envs_per_device
    n_minibatches = max(1, per_device_samples // 5000)
    batch_size = per_device_samples // n_minibatches

    log.info(f"Training: {args.total_timesteps} steps, {num_envs} envs "
             f"({envs_per_device}/GPU x {n_devices} GPUs)")
    log.info(f"  {n_steps} steps/update, {n_updates} updates, {n_epochs} epochs, "
             f"{n_minibatches} minibatches of {batch_size}")

    # Initialize network
    network = ActorCritic(action_dim=12)
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    params = network.init(init_rng, jnp.zeros(39))
    apply_fn = network.apply

    # Optimizer with LR schedule
    lr_denom = n_epochs * n_minibatches
    def lr_schedule(step):
        return 1e-3 * (0.999 ** (step / lr_denom))

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.zero_nans(),
        optax.adam(learning_rate=lr_schedule),
    )
    opt_state = optimizer.init(params)
    normalizer = init_normalizer(39)

    # Create MJX data template
    mjx_data_template = mjx.put_data(mj_model, mujoco.MjData(mj_model))

    # --- Initialize envs (num_envs total, then reshape for pmap) ---
    log.info("Initializing environments...")
    def reset_one(rng):
        return reset_fast(rng, mjx_data_template, ref_data, randomize=True)

    rng, init_rng = jax.random.split(rng)
    all_rngs = jax.random.split(init_rng, num_envs)
    all_states = jax.jit(jax.vmap(reset_one))(all_rngs)
    all_states.phase.block_until_ready()

    # Reshape to (n_devices, envs_per_device, ...)
    env_states = jax.tree.map(
        lambda x: x.reshape(n_devices, envs_per_device, *x.shape[1:]),
        all_states,
    )

    # Replicate params/opt_state/normalizer across devices
    def _replicate(x):
        return jnp.stack([x] * n_devices)

    params = jax.tree.map(_replicate, params)
    opt_state = jax.tree.map(_replicate, opt_state)
    normalizer = jax.tree.map(_replicate, normalizer)

    # Device RNGs: (n_devices, 2)
    rng, *d_rngs = jax.random.split(rng, n_devices + 1)
    device_rngs = jnp.stack(d_rngs)

    # --- Build single compiled train_step (rollout + PPO) ---
    def _get_obs_fn(state):
        return get_obs(state, ref_data)

    def _maybe_reset(state, done, rng):
        new_state = reset_fast(rng, state.mjx_data, ref_data, randomize=True)
        return jax.tree.map(lambda n, o: jnp.where(done, n, o), new_state, state)

    @functools.partial(jax.pmap, axis_name="d")
    def train_step(params, opt_state, normalizer, env_states, rng, update_frac):
        # Anneal entropy coefficient
        ent_coef = ent_coef_start + (ent_coef_end - ent_coef_start) * update_frac

        # Per-update friction DR: generate on each device, sync via all_gather
        rng, dr_rng = jax.random.split(rng)
        friction_coeff = jnp.clip(
            0.8 + jax.random.normal(dr_rng, ()) * 0.25, 0.1, 1.0
        )
        friction_coeff = jax.lax.all_gather(friction_coeff, axis_name="d")[0]
        dr_model = mjx_model.replace(
            geom_friction=mjx_model.geom_friction.at[:, 0].set(friction_coeff),
        )

        def _step_fn(state, action):
            return env_step(state, action, dr_model, ref_data)
        # ---- Rollout collection ----
        def scan_body(carry, _):
            states, norm, rng = carry
            rng, act_rng, reset_rng = jax.random.split(rng, 3)

            obs = jax.vmap(_get_obs_fn)(states)
            obs_norm = jax.vmap(lambda o: normalize_obs(norm, o))(obs)

            act_rngs = jax.random.split(act_rng, envs_per_device)
            actions, log_probs, values = jax.vmap(
                lambda o, r: sample_action(params, apply_fn, o, r)
            )(obs_norm, act_rngs)

            new_states, _, rewards, dones, rw_info = jax.vmap(_step_fn)(states, actions)

            reset_rngs = jax.random.split(reset_rng, envs_per_device)
            new_states = jax.vmap(_maybe_reset)(new_states, dones, reset_rngs)

            norm = update_normalizer(norm, obs)

            return (new_states, norm, rng), (obs_norm, actions, log_probs, values, rewards, dones, rw_info)

        (env_states, normalizer, rng), rollout = jax.lax.scan(
            scan_body, (env_states, normalizer, rng), None, length=n_steps
        )
        obs_r, act_r, lp_r, val_r, rew_r, done_r, rw_info_r = rollout

        # Last value for GAE
        last_obs = jax.vmap(_get_obs_fn)(env_states)
        last_obs_norm = jax.vmap(lambda o: normalize_obs(normalizer, o))(last_obs)
        _, _, last_vals = jax.vmap(lambda o: apply_fn(params, o))(last_obs_norm)

        # GAE
        advs, rets = compute_gae(rew_r, val_r, done_r, last_vals, gamma, gae_lambda)

        # Flatten for PPO
        total = n_steps * envs_per_device
        obs_f = obs_r.reshape(total, -1)
        act_f = act_r.reshape(total, -1)
        lp_f = lp_r.reshape(total)
        adv_f = advs.reshape(total)
        ret_f = rets.reshape(total)

        # ---- PPO update ----
        rng, ppo_rng = jax.random.split(rng)

        def epoch_fn(carry, epoch_rng):
            params, opt_state = carry
            perm = jax.random.permutation(epoch_rng, total)

            def mb_fn(carry, mb_idx):
                params, opt_state = carry
                idx = jax.lax.dynamic_slice(perm, (mb_idx * batch_size,), (batch_size,))
                grad_fn = jax.grad(ppo_loss, has_aux=True)
                grads, info = grad_fn(
                    params, apply_fn, obs_f[idx], act_f[idx], lp_f[idx],
                    adv_f[idx], ret_f[idx], clip_range, vf_coef, ent_coef,
                )
                # Grad norm before clipping
                grad_norm = jnp.sqrt(sum(
                    jnp.sum(x**2) for x in jax.tree.leaves(grads)
                ))
                info["grad_norm"] = grad_norm
                # Sync gradients across GPUs
                grads = jax.lax.pmean(grads, axis_name="d")
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), info

            (params, opt_state), infos = jax.lax.scan(
                mb_fn, (params, opt_state), jnp.arange(n_minibatches)
            )
            return (params, opt_state), infos

        epoch_rngs = jax.random.split(ppo_rng, n_epochs)
        (params, opt_state), all_infos = jax.lax.scan(
            epoch_fn, (params, opt_state), epoch_rngs
        )

        # Sync normalizer across GPUs
        normalizer = NormalizerState(
            mean=jax.lax.pmean(normalizer.mean, axis_name="d"),
            var=jax.lax.pmean(normalizer.var, axis_name="d"),
            count=normalizer.count,
        )

        # ---- Metrics ----
        metrics = jax.tree.map(lambda x: jnp.mean(x[-1]), all_infos)
        metrics["mean_reward"] = jnp.mean(rew_r)
        metrics["ent_coef"] = ent_coef

        # Per-component reward means (nanmean: NaN qpos from physics divergence)
        metrics["r_pos"] = jnp.nanmean(rw_info_r["rw_pos"])
        metrics["r_ori"] = jnp.nanmean(rw_info_r["rw_ori"])
        metrics["r_joint"] = jnp.nanmean(rw_info_r["rw_joint"])
        metrics["r_smooth"] = jnp.nanmean(rw_info_r["rw_smooth"])
        metrics["r_torque"] = jnp.nanmean(rw_info_r["rw_torque"])

        # Mean std from log_std parameter
        log_std_val = params["params"]["log_std"]
        metrics["mean_std"] = jnp.mean(jnp.exp(log_std_val))

        # Episode returns and lengths
        def _ep_scan(running, step_data):
            r, d = step_data
            ret, length = running
            ret = ret + r
            length = length + 1.0
            ep_ret = jnp.where(d, ret, 0.0)
            ep_len = jnp.where(d, length, 0.0)
            ep_valid = d.astype(jnp.float32)
            ret = jnp.where(d, 0.0, ret)
            length = jnp.where(d, 0.0, length)
            return (ret, length), (ep_ret, ep_len, ep_valid)

        _, (ep_rets, ep_lens, ep_valids) = jax.lax.scan(
            _ep_scan,
            (jnp.zeros(envs_per_device), jnp.zeros(envs_per_device)),
            (rew_r, done_r),
        )
        n_eps = jnp.sum(ep_valids)
        metrics["total_ep_return"] = jnp.where(
            n_eps > 0, jnp.sum(ep_rets) / n_eps, jnp.sum(rew_r) / envs_per_device
        )
        metrics["ep_length"] = jnp.where(
            n_eps > 0, jnp.sum(ep_lens) / n_eps, jnp.float32(n_steps)
        )

        # Termination reason breakdown
        term_r = rw_info_r["term_reason"]  # (n_steps, envs_per_device)
        metrics["n_episodes"] = n_eps
        metrics["term_thresh"] = jnp.sum((term_r == 1.0) & done_r)
        metrics["term_body"] = jnp.sum((term_r == 2.0) & done_r)
        metrics["term_contact"] = jnp.sum((term_r == 3.0) & done_r)
        metrics["term_nan"] = jnp.sum((term_r == 4.0) & done_r)
        metrics["term_trunc"] = jnp.sum((term_r == 5.0) & done_r)

        return (params, opt_state, normalizer, env_states, rng), metrics

    # --- Training loop ---
    write_training_header_jax(args.total_timesteps, num_envs, ref)

    reward_history = []
    timestep_history = []
    total_steps = 0
    best_ep_return = -float("inf")
    best_params_cpu = None
    best_norm_cpu = None
    next_video_step = 100_000

    log.info(f"\nStarting training ({n_updates} updates)...")
    log.info(f"  ent_coef: {ent_coef_start} -> {ent_coef_end} (annealed)")
    t_start = time.time()

    for update_idx in range(n_updates):
        t_update = time.time()

        # Annealing fraction (replicated across devices)
        update_frac = jnp.array([update_idx / max(1, n_updates - 1)] * n_devices)

        carry, metrics = train_step(
            params, opt_state, normalizer, env_states, device_rngs, update_frac
        )
        params, opt_state, normalizer, env_states, device_rngs = carry

        # Block for timing
        jax.tree.map(lambda x: x.block_until_ready(), metrics)
        dt = time.time() - t_update

        total_steps += n_steps * num_envs

        # Metrics from device 0
        ep_return = float(metrics["total_ep_return"][0])
        reward_history.append(ep_return)
        timestep_history.append(total_steps)

        pg_loss = float(metrics["pg_loss"][0])
        vf_loss = float(metrics["vf_loss"][0])
        entropy = float(metrics["entropy"][0])
        kl = float(metrics["approx_kl"][0])
        clip_frac = float(metrics["clip_fraction"][0])
        grad_norm = float(metrics["grad_norm"][0])
        mean_std = float(metrics["mean_std"][0])
        ep_len = float(metrics["ep_length"][0])
        r_pos = float(metrics["r_pos"][0])
        r_ori = float(metrics["r_ori"][0])
        r_joint = float(metrics["r_joint"][0])
        r_smooth = float(metrics["r_smooth"][0])
        r_torque = float(metrics["r_torque"][0])
        cur_ent_coef = float(metrics["ent_coef"][0])

        # Termination breakdown
        n_eps = float(metrics["n_episodes"][0])
        if n_eps > 0:
            t_thresh = float(metrics["term_thresh"][0]) / n_eps * 100
            t_body = float(metrics["term_body"][0]) / n_eps * 100
            t_contact = float(metrics["term_contact"][0]) / n_eps * 100
            t_nan = float(metrics["term_nan"][0]) / n_eps * 100
            t_trunc = float(metrics["term_trunc"][0]) / n_eps * 100
            term_str = (
                f"term: thresh={t_thresh:.0f}% body={t_body:.0f}% "
                f"cntct={t_contact:.0f}% nan={t_nan:.0f}% trunc={t_trunc:.0f}%"
            )
        else:
            term_str = "term: no_eps"

        msg = (
            f"[step {total_steps:>9,}  update {update_idx + 1:>5}]  "
            f"ep_return={ep_return:.2f}  ep_len={ep_len:.1f}  "
            f"r_pos={r_pos:.3f}  r_ori={r_ori:.3f}  r_joint={r_joint:.3f}  "
            f"r_smooth={r_smooth:.3f}  r_torque={r_torque:.3f}  "
            f"std={mean_std:.3f}  grad_norm={grad_norm:.3f}  "
            f"pg={pg_loss:.4g}  vf={vf_loss:.4g}  kl={kl:.4g}  "
            f"clip={clip_frac:.3f}  ent={entropy:.2f}  "
            f"ent_c={cur_ent_coef:.4f}  dt={dt:.1f}s  "
            f"{term_str}"
        )
        log.info(msg)

        # Save best checkpoint
        if ep_return > best_ep_return:
            best_ep_return = ep_return
            params_cpu = jax.tree.map(lambda x: x[0], params)
            norm_cpu = NormalizerState(
                mean=normalizer.mean[0], var=normalizer.var[0], count=normalizer.count[0]
            )
            best_params_cpu = params_cpu
            best_norm_cpu = norm_cpu
            best_dir = str(output_dir / "best_model")
            save_checkpoint(best_dir, params_cpu, norm_cpu, total_steps)

        # Periodic video rendering (every 100K steps)
        if total_steps >= next_video_step and best_params_cpu is not None:
            video_path = output_dir / "runs" / f"step_{total_steps:07d}.mp4"
            log.info(f"  Rendering tracking video at step {total_steps:,}...")
            try:
                import imageio
                _, _, _, images = execute_policy_rollout(
                    ref.state_traj, ref.grf_traj, ref.joint_vel_traj,
                    kindyn, best_params_cpu, apply_fn,
                    normalizer=best_norm_cpu, render=True,
                )
                if images:
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    fps = int(1.0 / args.control_dt)
                    imageio.mimsave(str(video_path), images, fps=fps)
                    log.info(f"  Video saved: {video_path}")
                else:
                    log.info(f"  No frames captured for video")
            except Exception as e:
                log.info(f"  Video render failed: {e}")
            next_video_step = (total_steps // 100_000 + 1) * 100_000

        # Save checkpoint periodically
        if (update_idx + 1) % max(1, n_updates // 10) == 0 or update_idx == n_updates - 1:
            params_cpu = jax.tree.map(lambda x: x[0], params)
            norm_cpu = NormalizerState(
                mean=normalizer.mean[0], var=normalizer.var[0], count=normalizer.count[0]
            )
            ckpt_path = str(output_dir / "checkpoints" / f"step_{total_steps}")
            save_checkpoint(ckpt_path, params_cpu, norm_cpu, total_steps)

        # Save plots periodically
        if (update_idx + 1) % max(1, n_updates // 5) == 0 or update_idx == n_updates - 1:
            save_reward_curve(
                np.array(reward_history), np.array(timestep_history),
                str(output_dir), total_steps
            )

    elapsed = time.time() - t_start
    log.info(f"\nTraining complete in {elapsed:.1f}s ({total_steps:,} steps)")
    log.info(f"Best ep_return: {best_ep_return:.2f}")

    # Save final checkpoint (separate from best)
    params_cpu = jax.tree.map(lambda x: x[0], params)
    norm_cpu = NormalizerState(
        mean=normalizer.mean[0], var=normalizer.var[0], count=normalizer.count[0]
    )
    final_dir = str(output_dir / "final_model")
    save_checkpoint(final_dir, params_cpu, norm_cpu, total_steps)
    log.info(f"Final model saved to {final_dir}")
    log.info(f"Best model saved to {str(output_dir / 'best_model')}")

    np.savez(
        str(output_dir / "normalizer.npz"),
        mean=np.array(norm_cpu.mean),
        var=np.array(norm_cpu.var),
        count=np.array(norm_cpu.count),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OPT-Mimic tracking policy (JAX/MJX)")
    parser.add_argument("--state-traj", type=str, required=True)
    parser.add_argument("--grf-traj", type=str, required=True)
    parser.add_argument("--joint-vel-traj", type=str, required=True)
    parser.add_argument("--contact-sequence", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="rl/trained_models")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--sim-dt", type=float, default=0.001)
    parser.add_argument("--control-dt", type=float, default=0.02)
    args = parser.parse_args()
    train(args)
