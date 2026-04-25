"""OPT-Mimic reward, tracking error, and termination functions.

All constants and math from the OPT-Mimic paper (Eq. 14-16, Sec III-C.4).
Pure functions — take tensor arguments, no environment coupling.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Actuation constants (Go2-tuned)
# ---------------------------------------------------------------------------
KP = 50.0
KD = 3.0
TORQUE_LIMITS = torch.tensor([
    23.7, 23.7, 45.43,  # FL
    23.7, 23.7, 45.43,  # FR
    23.7, 23.7, 45.43,  # RL
    23.7, 23.7, 45.43,  # RR
], dtype=torch.float32)
ACTION_LIMIT = 0.4

# ---------------------------------------------------------------------------
# Reward sigmas and weights (OPT-Mimic Eq. 16, Go2-tuned)
# ---------------------------------------------------------------------------
SIGMA_POS = 0.10
SIGMA_ORI = 0.25
SIGMA_JOINT = 0.5
SIGMA_SMOOTH = 1.0
SIGMA_TORQUE = 40.0

W_POS = 0.3
W_ORI = 0.3
W_JOINT = 0.2
W_SMOOTH = 0.1
W_TORQUE = 0.1

# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------
TERM_MULTIPLIER = 2.5
CONTACT_GRACE_WINDOW = 12  # 240ms at 50Hz

# Phase-0 instrumentation: stable order for CSV column layout.
TERM_CAUSE_NAMES = (
    "thresh_pos", "thresh_ori", "thresh_joint", "thresh_rate", "thresh_torque",
    "body", "contact", "nan", "trunc",
)


def quat_error_vec(q_ref: torch.Tensor, q_actual: torch.Tensor) -> torch.Tensor:
    """Vector part of q_ref * q_actual^{-1}.  Input (N,4) [w,x,y,z] → (N,3)."""
    q_inv = q_actual.clone()
    q_inv[:, 1:] = -q_inv[:, 1:]
    w1, x1, y1, z1 = q_ref[:, 0], q_ref[:, 1], q_ref[:, 2], q_ref[:, 3]
    w2, x2, y2, z2 = q_inv[:, 0], q_inv[:, 1], q_inv[:, 2], q_inv[:, 3]
    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def compute_tracking_errors(
    ref_pos: torch.Tensor, ref_quat: torch.Tensor, ref_joint: torch.Tensor,
    actual_pos: torch.Tensor, actual_quat: torch.Tensor, actual_joint: torch.Tensor,
    action_scaled: torch.Tensor, prev_action: torch.Tensor,
    last_torque: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute all tracking errors used by rewards and termination."""
    pos_err_sq = ((ref_pos - actual_pos) ** 2).sum(dim=-1)
    ori_err = quat_error_vec(ref_quat, actual_quat)
    ori_err_sq = (ori_err ** 2).sum(dim=-1)
    joint_err_sq = ((ref_joint - actual_joint) ** 2).sum(dim=-1)
    rate_sq = ((action_scaled - prev_action) ** 2).sum(dim=-1)
    max_torque = last_torque.abs().max(dim=-1).values
    return {
        "pos_err_sq": pos_err_sq,  "ori_err_sq": ori_err_sq,
        "joint_err_sq": joint_err_sq, "rate_sq": rate_sq,
        "max_torque": max_torque,
        "pos_error": pos_err_sq.sqrt(), "ori_error": ori_err_sq.sqrt(),
        "joint_error": joint_err_sq.sqrt(), "action_rate": rate_sq.sqrt(),
    }


def compute_rewards(te: dict[str, torch.Tensor]) -> torch.Tensor:
    """OPT-Mimic Eq. 15-16: weighted sum of Gaussian reward terms."""
    r_pos = torch.exp(-te["pos_err_sq"] / (2.0 * SIGMA_POS ** 2))
    r_ori = torch.exp(-te["ori_err_sq"] / (2.0 * SIGMA_ORI ** 2))
    r_joint = torch.exp(-te["joint_err_sq"] / (2.0 * SIGMA_JOINT ** 2))
    r_smooth = torch.exp(-te["rate_sq"] / (2.0 * SIGMA_SMOOTH ** 2))
    r_torque = torch.exp(-(te["max_torque"] ** 2) / (2.0 * SIGMA_TORQUE ** 2))
    total = W_POS * r_pos + W_ORI * r_ori + W_JOINT * r_joint + W_SMOOTH * r_smooth + W_TORQUE * r_torque
    return torch.nan_to_num(total, nan=0.0), {
        "r_pos": r_pos.mean().item(), "r_ori": r_ori.mean().item(),
        "r_joint": r_joint.mean().item(), "r_smooth": r_smooth.mean().item(),
        "r_torque": r_torque.mean().item(),
    }


def tracking_termination_breakdown(
    te: dict[str, torch.Tensor], first_step: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Per-cause masks for the OPT-Mimic threshold terminations."""
    return {
        "thresh_pos": te["pos_error"] > TERM_MULTIPLIER * SIGMA_POS,
        "thresh_ori": te["ori_error"] > TERM_MULTIPLIER * SIGMA_ORI,
        "thresh_joint": te["joint_error"] > TERM_MULTIPLIER * SIGMA_JOINT,
        "thresh_rate": (~first_step) & (te["action_rate"] > TERM_MULTIPLIER * SIGMA_SMOOTH),
        "thresh_torque": te["max_torque"] > TERM_MULTIPLIER * SIGMA_TORQUE,
    }


def check_tracking_termination(
    te: dict[str, torch.Tensor], first_step: torch.Tensor,
) -> torch.Tensor:
    """OPT-Mimic Sec III-C.4: terminate if any error > 2.5 * sigma."""
    m = tracking_termination_breakdown(te, first_step)
    return m["thresh_pos"] | m["thresh_ori"] | m["thresh_joint"] | m["thresh_rate"] | m["thresh_torque"]


def check_body_contact(
    net_forces: torch.Tensor, non_foot_ids: torch.Tensor,
) -> torch.Tensor:
    """Terminate if any non-foot body contacts the ground."""
    if len(non_foot_ids) == 0:
        return torch.zeros(net_forces.shape[0], dtype=torch.bool, device=net_forces.device)
    non_foot = torch.max(torch.norm(net_forces[:, :, non_foot_ids], dim=-1), dim=1)[0]
    return (non_foot > 1.0).any(dim=-1)


def contact_mismatch_diagnostics(
    net_forces: torch.Tensor, foot_ids: torch.Tensor,
    ref_contact: torch.Tensor, near_transition: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Per-foot contact-mismatch view used by both termination and slip logging.

    Returns dict with:
      mismatch_per_foot: (n_envs, 4) bool — true where actual ≠ expected and
                        we are NOT inside the ±CONTACT_GRACE_WINDOW around a
                        scheduled transition. Gated by has_info.
      actual_force_per_foot: (n_envs, 4) float — peak |force| per foot over
                        the contact-sensor history window.
      any_mismatch: (n_envs,) bool — true if any foot mismatched (used for
                        termination).
    """
    n_envs = net_forces.shape[0]
    device = net_forces.device
    has_info = ref_contact[:, 0] >= 0
    if not has_info.any():
        zeros_b = torch.zeros(n_envs, 4, dtype=torch.bool, device=device)
        zeros_f = torch.zeros(n_envs, 4, dtype=net_forces.dtype, device=device)
        return {
            "mismatch_per_foot": zeros_b,
            "actual_force_per_foot": zeros_f,
            "any_mismatch": torch.zeros(n_envs, dtype=torch.bool, device=device),
        }
    foot_forces = torch.max(torch.norm(net_forces[:, :, foot_ids], dim=-1), dim=1)[0]
    actual = foot_forces > 1.0
    expected = ref_contact > 0.5
    mismatch = (actual != expected) & (near_transition < 0.5) & has_info.unsqueeze(-1)
    return {
        "mismatch_per_foot": mismatch,
        "actual_force_per_foot": foot_forces,
        "any_mismatch": mismatch.any(dim=-1),
    }


def check_contact_mismatch(
    net_forces: torch.Tensor, foot_ids: torch.Tensor,
    ref_contact: torch.Tensor, near_transition: torch.Tensor,
) -> torch.Tensor:
    """Terminate on foot contact mismatch outside the grace window."""
    return contact_mismatch_diagnostics(
        net_forces, foot_ids, ref_contact, near_transition,
    )["any_mismatch"]
