"""JAX/Flax PPO implementation for OPT-Mimic tracking.

ActorCritic network, GAE computation, PPO loss, observation normalization,
and checkpoint save/load utilities. All JIT-compatible.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """Separate actor [128,128] and critic [512,512] with ReLU + orthogonal init."""

    action_dim: int = 12

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Actor: 39 -> 128 -> 128 -> 12 (tanh output)
        a = nn.Dense(128, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        a = nn.relu(a)
        a = nn.Dense(128, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(a)
        a = nn.relu(a)
        mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(
            a
        )
        mean = nn.tanh(mean)
        log_std = self.param(
            "log_std", nn.initializers.constant(-0.7), (self.action_dim,)
        )
        log_std = jnp.clip(log_std, -2.0, 0.0)

        # Critic: 39 -> 512 -> 512 -> 1
        v = nn.Dense(512, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        v = nn.relu(v)
        v = nn.Dense(512, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(v)
        v = nn.relu(v)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(v)

        return mean, log_std, value.squeeze(-1)


# ---------------------------------------------------------------------------
# Observation normalization (running mean/var)
# ---------------------------------------------------------------------------


class NormalizerState(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


def init_normalizer(obs_dim: int) -> NormalizerState:
    return NormalizerState(
        mean=jnp.zeros(obs_dim),
        var=jnp.ones(obs_dim),
        count=jnp.array(1e-4),
    )


def update_normalizer(state: NormalizerState, batch: jnp.ndarray) -> NormalizerState:
    """Welford online update with a batch of observations (N, obs_dim)."""
    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = jnp.array(batch.shape[0], dtype=jnp.float32)

    delta = batch_mean - state.mean
    total_count = state.count + batch_count
    new_mean = state.mean + delta * batch_count / total_count
    m_a = state.var * state.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta**2 * state.count * batch_count / total_count
    new_var = m2 / total_count
    return NormalizerState(mean=new_mean, var=new_var, count=total_count)


def normalize_obs(
    state: NormalizerState, obs: jnp.ndarray, clip: float = 10.0
) -> jnp.ndarray:
    return jnp.clip((obs - state.mean) / jnp.sqrt(state.var + 1e-8), -clip, clip)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: jnp.ndarray,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute GAE advantages and returns.

    Args:
        rewards: (T, N) rewards
        values: (T, N) value estimates
        dones: (T, N) done flags
        last_value: (N,) value estimate for the state after the last step
        gamma: discount factor
        gae_lambda: GAE lambda

    Returns:
        advantages: (T, N)
        returns: (T, N)
    """
    T = rewards.shape[0]

    def _scan_fn(gae, t):
        idx = T - 1 - t
        done = dones[idx]
        next_val = jnp.where(t == 0, last_value, values[idx + 1])
        delta = rewards[idx] + gamma * next_val * (1.0 - done) - values[idx]
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae
        return gae, gae

    _, advantages_rev = jax.lax.scan(
        _scan_fn, jnp.zeros_like(last_value), jnp.arange(T)
    )
    advantages = jnp.flip(advantages_rev, axis=0)
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO loss
# ---------------------------------------------------------------------------


def ppo_loss(
    params: Any,
    apply_fn: Any,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    clip_range: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.005,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """PPO clipped objective with value and entropy losses."""
    mean, log_std, values = apply_fn(params, obs)
    std = jnp.exp(log_std)

    # Gaussian log prob
    log_probs = -0.5 * jnp.sum(
        ((actions - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1
    )

    # Policy loss (clipped)
    ratio = jnp.exp(log_probs - old_log_probs)
    adv_norm = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    pg_loss1 = -adv_norm * ratio
    pg_loss2 = -adv_norm * jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)
    pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

    # Value loss
    vf_loss = 0.5 * jnp.mean((values - returns) ** 2)

    # Entropy bonus
    entropy = 0.5 * jnp.sum(jnp.log(2 * jnp.pi * jnp.e * std**2), axis=-1)
    entropy_loss = -jnp.mean(entropy)

    total_loss = pg_loss + vf_coef * vf_loss + ent_coef * entropy_loss

    info = {
        "pg_loss": pg_loss,
        "vf_loss": vf_loss,
        "entropy": jnp.mean(entropy),
        "approx_kl": jnp.mean(0.5 * (log_probs - old_log_probs) ** 2),
        "clip_fraction": jnp.mean(jnp.abs(ratio - 1.0) > clip_range),
    }
    return total_loss, info


# ---------------------------------------------------------------------------
# Action sampling
# ---------------------------------------------------------------------------


def sample_action(
    params: Any, apply_fn: Any, obs: jnp.ndarray, rng: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample action from policy, return (action, log_prob, value)."""
    mean, log_std, value = apply_fn(params, obs)
    std = jnp.exp(log_std)
    noise = jax.random.normal(rng, mean.shape)
    action = mean + std * noise
    action = jnp.clip(action, -1.0, 1.0)
    log_prob = -0.5 * jnp.sum(
        ((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1
    )
    return action, log_prob, value


def deterministic_action(
    params: Any, apply_fn: Any, obs: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Deterministic action (mean) for evaluation."""
    mean, _, value = apply_fn(params, obs)
    return mean, value


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str, params: Any, normalizer: NormalizerState, step: int
) -> None:
    """Save params + normalizer to disk as numpy pickle (synchronous, reliable)."""
    import os
    import pickle

    os.makedirs(path, exist_ok=True)
    ckpt = {
        "params": jax.tree.map(lambda x: np.array(x), jax.device_get(params)),
        "normalizer_mean": np.array(normalizer.mean),
        "normalizer_var": np.array(normalizer.var),
        "normalizer_count": np.array(normalizer.count),
        "step": int(step),
    }
    ckpt_file = os.path.join(path, "checkpoint.pkl")
    with open(ckpt_file, "wb") as f:
        pickle.dump(ckpt, f)


def load_checkpoint(path: str) -> tuple[Any, NormalizerState, int]:
    """Load params + normalizer from numpy pickle checkpoint."""
    import os
    import pickle

    ckpt_file = os.path.join(path, "checkpoint.pkl")
    if not os.path.exists(ckpt_file):
        # Try path directly as a file
        ckpt_file = path
    with open(ckpt_file, "rb") as f:
        ckpt = pickle.load(f)
    params = jax.tree.map(lambda x: jnp.array(x), ckpt["params"])
    normalizer = NormalizerState(
        mean=jnp.array(ckpt["normalizer_mean"]),
        var=jnp.array(ckpt["normalizer_var"]),
        count=jnp.array(ckpt["normalizer_count"]),
    )
    return params, normalizer, int(ckpt["step"])
