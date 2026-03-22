"""RSL-RL training configuration for OPT-Mimic.

All hyperparameters match rl/train.py exactly:
  gamma=0.995, lambda=0.95, clip=0.2, vf_coef=0.5, max_grad_norm=0.5
  LR: 1e-3 with 0.999^(step/denom) decay
  Entropy: 0.002 -> 0.0005 (annealed)
  Network: actor [128,128], critic [512,512]
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OPTMimicPPOCfg:
    """PPO configuration matching MJX implementation."""

    seed: int = 42

    # Runner
    num_steps_per_env: int = 80  # overridden to max_phase at runtime
    max_iterations: int = 1000   # overridden at runtime
    save_interval: int = 100
    experiment_name: str = "opt_mimic_go2"
    run_name: str = ""
    log_interval: int = 1

    # Algorithm
    value_loss_coef: float = 0.5
    use_clipped_value_loss: bool = False  # MJX code does NOT clip value loss
    clip_param: float = 0.2
    entropy_coef: float = 0.002  # start value, annealed to 0.0005
    entropy_coef_end: float = 0.0005
    num_learning_epochs: int = 10
    num_mini_batches: int = 0  # computed at runtime: per_device_samples // 5000
    learning_rate: float = 1e-3
    gamma: float = 0.995
    lam: float = 0.95
    max_grad_norm: float = 0.5
    desired_kl: float = 0.0  # no adaptive KL
    schedule: str = "fixed"  # we handle LR schedule manually

    # Network
    actor_hidden_dims: list = None
    critic_hidden_dims: list = None

    # Normalization
    empirical_normalization: bool = True
    normalize_value: bool = False

    def __post_init__(self):
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [128, 128]
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [512, 512]
