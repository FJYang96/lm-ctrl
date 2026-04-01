"""PPO algorithm: GAE computation and minibatch update.

Pure RL math — no environment coupling.
"""

from __future__ import annotations

import torch

from .network import OPTMimicActorCritic
from .train_cfg import OPTMimicPPOCfg


def compute_gae(
    rew_buf: torch.Tensor, val_buf: torch.Tensor, done_buf: torch.Tensor,
    last_values: torch.Tensor, gamma: float, lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation (reverse-pass).

    Returns (advantages, returns), both shape (n_steps, num_envs).
    """
    n_steps = rew_buf.shape[0]
    num_envs = rew_buf.shape[1]
    advantages = torch.zeros_like(rew_buf)
    last_gae = torch.zeros(num_envs, device=rew_buf.device)
    for t in reversed(range(n_steps)):
        next_values = last_values if t == n_steps - 1 else val_buf[t + 1]
        non_terminal = (~done_buf[t]).float()
        delta = rew_buf[t] + gamma * next_values * non_terminal - val_buf[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    return advantages, advantages + val_buf


def ppo_update(
    actor_critic: OPTMimicActorCritic,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    obs_flat: torch.Tensor, act_flat: torch.Tensor, lp_flat: torch.Tensor,
    adv_flat: torch.Tensor, ret_flat: torch.Tensor,
    cfg: OPTMimicPPOCfg, ent_coef: float,
) -> dict[str, float]:
    """Run PPO epochs over collected rollout data. Returns averaged metrics."""
    total_samples = obs_flat.shape[0]
    n_minibatches = cfg.num_mini_batches
    batch_size = total_samples // n_minibatches
    device = obs_flat.device
    metrics = {"pg_loss": 0.0, "vf_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
    n_updates = 0

    for _ in range(cfg.num_learning_epochs):
        perm = torch.randperm(total_samples, device=device)
        for mb in range(n_minibatches):
            idx = perm[mb * batch_size:(mb + 1) * batch_size]
            mb_obs, mb_act = obs_flat[idx], act_flat[idx]
            mb_old_lp, mb_ret = lp_flat[idx], ret_flat[idx]
            mb_adv = adv_flat[idx]
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            actor_critic._update_distribution(mb_obs)
            new_lp = actor_critic.get_actions_log_prob(mb_act)
            new_values = actor_critic.evaluate(mb_obs)
            entropy = actor_critic.entropy.mean()

            ratio = torch.exp(new_lp - mb_old_lp)
            pg_loss = torch.max(
                -mb_adv * ratio,
                -mb_adv * ratio.clamp(1.0 - cfg.clip_param, 1.0 + cfg.clip_param),
            ).mean()
            vf_loss = 0.5 * ((new_values - mb_ret) ** 2).mean()
            loss = pg_loss + cfg.value_loss_coef * vf_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                metrics["pg_loss"] += pg_loss.item()
                metrics["vf_loss"] += vf_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["approx_kl"] += (0.5 * ((new_lp - mb_old_lp) ** 2).mean()).item()
                metrics["clip_frac"] += ((ratio - 1.0).abs() > cfg.clip_param).float().mean().item()
            n_updates += 1

    for k in metrics:
        metrics[k] /= max(1, n_updates)
    return metrics
