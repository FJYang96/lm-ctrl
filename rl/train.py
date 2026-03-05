"""Train a per-trajectory Go2 tracking policy using JAX PPO (OPT-Mimic).

GPU-accelerated training with MuJoCo MJX. Falls back to CPU JAX if no GPU.

Usage:
    python -m rl.train \
        --state-traj results/state_traj.npy \
        --grf-traj results/grf_traj.npy \
        --joint-vel-traj results/joint_vel_traj.npy \
        --output-dir rl/trained_models \
        --total-timesteps 2000000 \
        --num-envs 256
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

import config
from mpc.dynamics.model import KinoDynamic_Model

from .callbacks import get_log, write_training_header_jax, save_reward_curve, save_component_plot
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
    EnvState,
    RefData,
    load_mjx_model,
    make_ref_data,
    get_obs,
    reset_fast,
    step as env_step,
    _get_foot_body_ids,
    _get_ground_geom_id,
)


def build_reference(
    state_traj_path: str,
    grf_traj_path: str,
    joint_vel_traj_path: str,
    contact_sequence_path: str | None = None,
    control_dt: float = 0.02,
) -> ReferenceTrajectory:
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
    return ref


def train(args: argparse.Namespace) -> None:
    # Suppress warp import warnings
    os.environ.setdefault("MUJOCO_GL", "egl")

    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference trajectory (CPU)
    print("Loading reference trajectory...")
    ref = build_reference(
        args.state_traj, args.grf_traj, args.joint_vel_traj,
        contact_sequence_path=args.contact_sequence,
        control_dt=args.control_dt,
    )
    ref_data = make_ref_data(ref)
    print(f"Reference: {ref.max_phase} steps, {ref.duration:.2f}s")

    # Load MJX model
    print("Loading MJX model...")
    mj_model, mjx_model = load_mjx_model(sim_dt=args.sim_dt)
    foot_body_ids = _get_foot_body_ids(mj_model)
    ground_geom_id = _get_ground_geom_id(mj_model)

    # PPO hyperparameters
    num_envs = args.num_envs
    samples_per_update = 20000
    n_steps = max(1, samples_per_update // num_envs)
    n_updates = args.total_timesteps // samples_per_update
    batch_size = min(5000, n_steps * num_envs)
    n_epochs = args.n_epochs
    gamma = 0.995
    gae_lambda = 0.95
    clip_range = 0.2
    ent_coef = 0.005
    vf_coef = 0.5
    max_grad_norm = 0.5

    print(f"Training: {args.total_timesteps} steps, {num_envs} envs, "
          f"{n_steps} steps/update, {n_updates} updates, {n_epochs} epochs")

    # Initialize network
    network = ActorCritic(action_dim=12)
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros(39)
    params = network.init(init_rng, dummy_obs)

    # Optimizer with LR schedule (exponential decay 0.99 per update)
    def lr_schedule(step):
        # step = update_idx * n_epochs * n_minibatches + ...
        # We approximate: lr decays per outer update
        update_approx = step / (n_epochs * max(1, (n_steps * num_envs) // batch_size))
        return 1e-3 * (0.99 ** update_approx)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.zero_nans(),
        optax.adam(learning_rate=lr_schedule),
    )
    opt_state = optimizer.init(params)

    # Observation normalizer
    normalizer = init_normalizer(39)

    # Initialize environments
    print("Initializing environments...")
    import mujoco as mj
    mj_data_template = mj.MjData(mj_model)
    mjx_data_template = jax.jit(lambda: __import__('mujoco').mjx.put_data(mj_model, mj_data_template))()

    rng, *env_rngs = jax.random.split(rng, num_envs + 1)
    env_rngs = jnp.stack(env_rngs)

    # Vectorized reset
    def reset_one(rng):
        return reset_fast(rng, mjx_model, mjx_data_template, ref_data, randomize=True)

    print("Compiling reset...")
    t0 = time.time()
    batched_reset = jax.jit(jax.vmap(reset_one))
    env_states = batched_reset(env_rngs)
    env_states.phase.block_until_ready()
    print(f"Reset compiled in {time.time() - t0:.1f}s")

    # Compile step function
    def step_one(state, action):
        return env_step(state, action, mjx_model, ref_data, foot_body_ids, ground_geom_id)

    batched_step = jax.vmap(step_one)

    def get_obs_one(state):
        return get_obs(state, ref_data)

    batched_get_obs = jax.vmap(get_obs_one)

    # Compile a single step to warm up
    print("Compiling step...")
    t0 = time.time()
    dummy_actions = jnp.zeros((num_envs, 12))
    _test_out = jax.jit(batched_step)(env_states, dummy_actions)
    _test_out[0].phase.block_until_ready()
    print(f"Step compiled in {time.time() - t0:.1f}s")

    # Apply function
    apply_fn = network.apply

    # JIT-compiled rollout collection
    @jax.jit
    def collect_rollout(params, normalizer, env_states, rng):
        """Collect n_steps of experience from all envs."""
        def scan_step(carry, _):
            states, norm, rng = carry
            rng, act_rng, reset_rng = jax.random.split(rng, 3)

            obs = batched_get_obs(states)
            obs_norm = jax.vmap(lambda o: normalize_obs(norm, o))(obs)

            # Sample actions
            act_rngs = jax.random.split(act_rng, num_envs)
            actions, log_probs, values = jax.vmap(
                lambda p, o, r: sample_action(p, apply_fn, o, r),
                in_axes=(None, 0, 0)
            )(params, obs_norm, act_rngs)

            # Step all envs
            new_states, next_obs, rewards, dones = batched_step(states, actions)

            # Auto-reset done envs
            reset_rngs = jax.random.split(reset_rng, num_envs)

            def maybe_reset(state, done, rng):
                new_state = reset_fast(rng, mjx_model, state.mjx_data, ref_data, randomize=True)
                return jax.tree.map(
                    lambda n, o: jnp.where(done, n, o), new_state, state
                )

            new_states = jax.vmap(maybe_reset)(new_states, dones, reset_rngs)

            # Update normalizer with raw obs
            norm = update_normalizer(norm, obs)

            transition = {
                "obs": obs_norm,
                "actions": actions,
                "log_probs": log_probs,
                "values": values,
                "rewards": rewards,
                "dones": dones,
            }

            return (new_states, norm, rng), transition

        (env_states_out, normalizer_out, rng_out), rollout = jax.lax.scan(
            scan_step, (env_states, normalizer, rng), None, length=n_steps
        )

        # Last value for GAE
        last_obs = batched_get_obs(env_states_out)
        last_obs_norm = jax.vmap(lambda o: normalize_obs(normalizer_out, o))(last_obs)
        _, _, last_values = jax.vmap(
            lambda p, o: apply_fn(p, o), in_axes=(None, 0)
        )(params, last_obs_norm)

        return env_states_out, normalizer_out, rng_out, rollout, last_values

    # JIT-compiled PPO update
    @jax.jit
    def ppo_update(params, opt_state, rollout, last_values, rng):
        """Run n_epochs of minibatch PPO updates."""
        obs = rollout["obs"]  # (T, N, 39)
        actions = rollout["actions"]  # (T, N, 12)
        old_log_probs = rollout["log_probs"]  # (T, N)
        values = rollout["values"]  # (T, N)
        rewards = rollout["rewards"]  # (T, N)
        dones = rollout["dones"]  # (T, N)

        # GAE
        advantages, returns = compute_gae(rewards, values, dones, last_values, gamma, gae_lambda)

        # Flatten
        T, N = obs.shape[0], obs.shape[1]
        total_samples = T * N
        obs_flat = obs.reshape(total_samples, -1)
        actions_flat = actions.reshape(total_samples, -1)
        old_lp_flat = old_log_probs.reshape(total_samples)
        adv_flat = advantages.reshape(total_samples)
        ret_flat = returns.reshape(total_samples)

        def epoch_step(carry, rng_epoch):
            params, opt_state = carry
            # Shuffle
            perm = jax.random.permutation(rng_epoch, total_samples)
            n_batches = max(1, total_samples // batch_size)

            def batch_step(carry, batch_idx):
                params, opt_state = carry
                start = batch_idx * batch_size
                idx = jax.lax.dynamic_slice(perm, (start,), (batch_size,))

                mb_obs = obs_flat[idx]
                mb_actions = actions_flat[idx]
                mb_old_lp = old_lp_flat[idx]
                mb_adv = adv_flat[idx]
                mb_ret = ret_flat[idx]

                grad_fn = jax.grad(ppo_loss, has_aux=True)
                grads, info = grad_fn(
                    params, apply_fn, mb_obs, mb_actions, mb_old_lp,
                    mb_adv, mb_ret, clip_range, vf_coef, ent_coef,
                )
                updates, new_opt_state = optimizer.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                return (new_params, new_opt_state), info

            (params, opt_state), infos = jax.lax.scan(
                batch_step, (params, opt_state), jnp.arange(n_batches)
            )
            return (params, opt_state), infos

        epoch_rngs = jax.random.split(rng, n_epochs)
        (params, opt_state), all_infos = jax.lax.scan(
            epoch_step, (params, opt_state), epoch_rngs
        )

        # Average metrics across last epoch's batches
        last_info = jax.tree.map(lambda x: jnp.mean(x[-1]), all_infos)
        mean_reward = jnp.mean(rewards)
        last_info["mean_reward"] = mean_reward
        last_info["mean_ep_length"] = jnp.mean(jnp.sum(1.0 - dones.astype(jnp.float32), axis=0))

        return params, opt_state, last_info

    # Training loop
    log = get_log()
    write_training_header_jax(args.total_timesteps, num_envs, ref)

    reward_history = []
    component_history = {"rw_pos": [], "rw_ori": [], "rw_joint": [], "rw_smooth": [], "rw_torque": []}
    timestep_history = []

    total_steps = 0
    print(f"\nStarting training ({n_updates} updates)...")
    t_start = time.time()

    for update_idx in range(n_updates):
        t_update = time.time()

        # Collect rollout
        rng, rollout_rng = jax.random.split(rng)
        env_states, normalizer, rng, rollout, last_values = collect_rollout(
            params, normalizer, env_states, rollout_rng
        )

        # PPO update
        rng, ppo_rng = jax.random.split(rng)
        params, opt_state, update_info = ppo_update(
            params, opt_state, rollout, last_values, ppo_rng
        )

        total_steps += n_steps * num_envs
        dt = time.time() - t_update

        # Log every update
        mean_reward = float(update_info["mean_reward"])
        reward_history.append(mean_reward)
        timestep_history.append(total_steps)

        if update_idx % 1 == 0:
            pg_loss = float(update_info["pg_loss"])
            vf_loss = float(update_info["vf_loss"])
            entropy = float(update_info["entropy"])
            kl = float(update_info["approx_kl"])
            clip_frac = float(update_info["clip_fraction"])

            msg = (
                f"[step {total_steps:>9,}  update {update_idx + 1:>5}]  "
                f"reward={mean_reward:.4f}  pg_loss={pg_loss:.5g}  "
                f"vf_loss={vf_loss:.5g}  kl={kl:.5g}  clip={clip_frac:.3f}  "
                f"entropy={entropy:.3f}  dt={dt:.1f}s"
            )
            log.info(msg)
            print(msg)

        # Save checkpoint periodically
        if (update_idx + 1) % max(1, n_updates // 10) == 0 or update_idx == n_updates - 1:
            ckpt_path = str(output_dir / "checkpoints" / f"step_{total_steps}")
            save_checkpoint(ckpt_path, params, normalizer, total_steps)

        # Save plots periodically
        if (update_idx + 1) % max(1, n_updates // 5) == 0 or update_idx == n_updates - 1:
            save_reward_curve(
                np.array(reward_history), np.array(timestep_history),
                str(output_dir), total_steps
            )

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed:.1f}s ({total_steps:,} steps)")

    # Save final checkpoint as "best_model"
    best_dir = str(output_dir / "best_model")
    save_checkpoint(best_dir, params, normalizer, total_steps)
    print(f"Model saved to {best_dir}")

    # Save normalizer stats separately for backward compatibility
    np.savez(
        str(output_dir / "normalizer.npz"),
        mean=np.array(normalizer.mean),
        var=np.array(normalizer.var),
        count=np.array(normalizer.count),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OPT-Mimic tracking policy (JAX/MJX)")
    parser.add_argument("--state-traj", type=str, required=True)
    parser.add_argument("--grf-traj", type=str, required=True)
    parser.add_argument("--joint-vel-traj", type=str, required=True)
    parser.add_argument("--contact-sequence", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="rl/trained_models")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--sim-dt", type=float, default=0.001)
    parser.add_argument("--control-dt", type=float, default=0.02)
    args = parser.parse_args()
    train(args)
