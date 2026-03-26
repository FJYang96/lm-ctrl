"""OPT-Mimic ActorCritic network for RSL-RL.

Architecture matches rl/ppo.py exactly:
  Actor:  39 -> 128 (ReLU) -> 128 (ReLU) -> 12 (Tanh)
  Critic: 39 -> 512 (ReLU) -> 512 (ReLU) -> 1
  log_std: learnable param, init=-0.7, clamped [-2.0, 0.0]
  Orthogonal init: sqrt(2) for hidden, 0.01 for actor head, 1.0 for critic head
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal


def _ortho_init(module: nn.Linear, gain: float):
    """Orthogonal initialization matching Flax's nn.initializers.orthogonal(gain)."""
    nn.init.orthogonal_(module.weight, gain=gain)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class OPTMimicActorCritic(nn.Module):
    """Actor-Critic matching rl/ppo.py ActorCritic (Flax) exactly.

    Implements the RSL-RL ActorCritic interface:
    - act(obs, **kwargs) -> actions
    - act_inference(obs) -> actions
    - evaluate(obs, actions) -> (values, log_probs, entropy)
    - get_actions_log_prob(actions) -> log_probs
    """

    is_recurrent = False

    def __init__(
        self,
        num_obs: int,
        num_privileged_obs: int,  # unused, required by RSL-RL
        num_actions: int,
        actor_hidden_dims: list[int] = [128, 128],
        critic_hidden_dims: list[int] = [512, 512],
        init_noise_std: float = math.exp(-0.7),  # exp(-0.7) ≈ 0.497
        **kwargs,
    ):
        super().__init__()

        # Actor: 39 -> 128 -> 128 -> 12 -> tanh
        self.actor = nn.Sequential(
            nn.Linear(num_obs, actor_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(actor_hidden_dims[0], actor_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(actor_hidden_dims[1], num_actions),
            nn.Tanh(),
        )

        # Critic: 39 -> 512 -> 512 -> 1
        self.critic = nn.Sequential(
            nn.Linear(num_obs, critic_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(critic_hidden_dims[0], critic_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(critic_hidden_dims[1], 1),
        )

        # Learnable log_std, init=-0.7, clamped to [-2.0, 0.0]
        self.log_std = nn.Parameter(torch.ones(num_actions) * -0.7)

        # Orthogonal initialization (matching Flax)
        _ortho_init(self.actor[0], gain=math.sqrt(2))
        _ortho_init(self.actor[2], gain=math.sqrt(2))
        _ortho_init(self.actor[4], gain=0.01)
        _ortho_init(self.critic[0], gain=math.sqrt(2))
        _ortho_init(self.critic[2], gain=math.sqrt(2))
        _ortho_init(self.critic[4], gain=1.0)

        # Internal state for action distribution (set by forward pass)
        self._distribution: Normal | None = None
        self._mean: torch.Tensor | None = None

    @property
    def action_mean(self) -> torch.Tensor:
        return self._mean

    @property
    def action_std(self) -> torch.Tensor:
        return torch.exp(self.log_std.clamp(-2.0, 0.0))

    @property
    def entropy(self) -> torch.Tensor:
        if self._distribution is not None:
            return self._distribution.entropy().sum(dim=-1)
        return torch.tensor(0.0)

    @property
    def std(self) -> torch.Tensor:
        return self.action_std

    def reset(self, dones=None):
        """No-op for non-recurrent networks."""
        pass

    def forward(self):
        raise NotImplementedError("Use act(), act_inference(), or evaluate()")

    def _update_distribution(self, obs: torch.Tensor):
        mean = self.actor(obs)
        self._mean = mean
        std = self.action_std
        self._distribution = Normal(mean, std)

    def act(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        """Sample action from policy. Returns clipped action in [-1, 1]."""
        self._update_distribution(obs)
        actions = self._distribution.sample()
        actions = actions.clamp(-1.0, 1.0)
        return actions

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action (mean) for evaluation."""
        return self.actor(obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions under current distribution.

        IMPORTANT: No tanh-squashing correction, matching JAX ppo.py exactly.
        The Flax code computes log_prob = -0.5 * sum(((a-mu)/std)^2 + 2*log_std + log(2*pi))
        which is the standard Gaussian log-prob without any correction.
        """
        if self._distribution is None:
            raise RuntimeError("Call act() first to set up distribution")
        return self._distribution.log_prob(actions).sum(dim=-1)

    def evaluate(
        self, obs: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute value estimate. Returns (value, None) — second element unused by RSL-RL for non-privileged."""
        value = self.critic(obs).squeeze(-1)
        return value
