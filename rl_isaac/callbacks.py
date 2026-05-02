"""Logging and plotting callbacks for Isaac Lab OPT-Mimic training.

Matches rl/callbacks.py output format: experiment.log with same metrics,
reward_curve.png with same smoothing.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .rewards import TERM_CAUSE_NAMES


class TrainingLogger:
    """Logs training metrics to experiment.log and tracks reward history."""

    def __init__(self, output_dir: Path, total_timesteps: int, num_envs: int, max_phase: int):
        self.output_dir = Path(output_dir)
        self.log_path = self.output_dir / "experiment.log"
        self.term_csv_path = self.output_dir / "term_breakdown.csv"
        self._term_csv_initialised = False
        self.phase_err_csv_path = self.output_dir / "phase_errors.csv"
        self._phase_err_csv_initialised = False
        self.slip_csv_path = self.output_dir / "slip_log.csv"
        self._slip_csv_initialised = False
        self.reward_history: list[float] = []
        self.timestep_history: list[int] = []

        # Setup logger
        self._logger = logging.getLogger("rl_isaac.diagnostics")
        for h in self._logger.handlers[:]:
            self._logger.removeHandler(h)
            h.close()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(self.log_path, mode="a")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)

        # Also log to stdout
        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(stdout_handler)

        # Header
        if max_phase > 0:
            self._write_header(total_timesteps, num_envs, max_phase)

    def _write_header(self, total_timesteps, num_envs, max_phase):
        self._logger.info("")
        self._logger.info("=" * 60)
        self._logger.info("TRAINING: ISAAC LAB PPO LEARNING LOG (OPT-MIMIC)")
        self._logger.info("=" * 60)
        self._logger.info(f"Timesteps: {total_timesteps}  Envs: {num_envs}")
        self._logger.info(f"Trajectory: {max_phase} steps, {max_phase * 0.02:.2f}s")
        self._logger.info("=" * 60)

    def update_header(self, total_timesteps, num_envs, max_phase):
        """Rewrite header once we know the actual max_phase from the env."""
        self._write_header(total_timesteps, num_envs, max_phase)

    def info(self, msg: str):
        """Log an info message to both file and stdout."""
        self._logger.info(msg)

    def log_update(
        self,
        update_idx: int,
        total_steps: int,
        ep_return: float,
        ep_length: float,
        r_pos: float,
        r_ori: float,
        r_joint: float,
        r_smooth: float,
        r_torque: float,
        mean_std: float,
        grad_norm: float,
        pg_loss: float,
        vf_loss: float,
        approx_kl: float,
        clip_frac: float,
        entropy: float,
        ent_coef: float,
        dt: float,
        term_info: dict | None = None,
        episodes_finished: int | None = None,
        term_trunc_rate: float | None = None,
    ):
        self.reward_history.append(ep_return)
        self.timestep_history.append(total_steps)

        # Termination breakdown
        term_str = ""
        if term_info:
            t_thresh = term_info.get("term_thresh", 0)
            t_body = term_info.get("term_body", 0)
            t_contact = term_info.get("term_contact", 0)
            t_nan = term_info.get("term_nan", 0)
            t_trunc = term_info.get("term_trunc", 0)
            term_str = (
                f"term: thresh={t_thresh:.0f} body={t_body:.0f} "
                f"cntct={t_contact:.0f} nan={t_nan:.0f} trunc={t_trunc:.0f}"
            )
            if episodes_finished is not None and term_trunc_rate is not None:
                term_str += f" done={episodes_finished:d} trunc_rate={term_trunc_rate:.3f}"

        msg = (
            f"[step {total_steps:>9,}  update {update_idx + 1:>5}]  "
            f"ep_return={ep_return:.2f}  ep_len={ep_length:.1f}  "
            f"r_pos={r_pos:.3f}  r_ori={r_ori:.3f}  r_joint={r_joint:.3f}  "
            f"r_smooth={r_smooth:.3f}  r_torque={r_torque:.3f}  "
            f"std={mean_std:.3f}  grad_norm={grad_norm:.3f}  "
            f"pg={pg_loss:.4g}  vf={vf_loss:.4g}  kl={approx_kl:.4g}  "
            f"clip={clip_frac:.3f}  ent={entropy:.2f}  "
            f"ent_c={ent_coef:.4f}  dt={dt:.1f}s  "
            f"{term_str}"
        )
        self._logger.info(msg)

    def log_term_breakdown(self, update_idx: int, total_steps: int, hist):
        """Append per-cause × per-phase termination histogram for one update.

        `hist`: int tensor of shape (n_causes, max_phase+1). Rows with zero
        events are skipped to keep the CSV compact.
        """
        try:
            arr = hist.cpu().numpy()
        except AttributeError:
            arr = np.asarray(hist)
        if arr.size == 0 or not arr.any():
            return
        if not self._term_csv_initialised:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            new_file = not self.term_csv_path.exists()
            with self.term_csv_path.open("a", newline="") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["update_idx", "total_steps", "cause", "phase", "count"])
            self._term_csv_initialised = True
        with self.term_csv_path.open("a", newline="") as f:
            w = csv.writer(f)
            for c_idx, name in enumerate(TERM_CAUSE_NAMES):
                row = arr[c_idx]
                nz = np.nonzero(row)[0]
                for p in nz:
                    w.writerow([update_idx, total_steps, name, int(p), int(row[p])])

    def log_phase_errors(self, update_idx: int, total_steps: int,
                         sums, counts, metric_names: tuple[str, ...]):
        """Append one row per phase index for this update. Skips phases with
        zero samples in the rollout window."""
        try:
            sums_np = sums.cpu().numpy()
            counts_np = counts.cpu().numpy()
        except AttributeError:
            sums_np = np.asarray(sums)
            counts_np = np.asarray(counts)
        if counts_np.sum() == 0:
            return
        if not self._phase_err_csv_initialised:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            new_file = not self.phase_err_csv_path.exists()
            with self.phase_err_csv_path.open("a", newline="") as f:
                w = csv.writer(f)
                if new_file:
                    header = ["update_idx", "total_steps", "phase", "n_samples"]
                    header.extend(f"mean_{m}" for m in metric_names)
                    w.writerow(header)
            self._phase_err_csv_initialised = True
        with self.phase_err_csv_path.open("a", newline="") as f:
            w = csv.writer(f)
            for p_idx in range(counts_np.shape[0]):
                n = int(counts_np[p_idx])
                if n == 0:
                    continue
                means = sums_np[p_idx] / n
                w.writerow([update_idx, total_steps, p_idx, n,
                            *[f"{v:.6g}" for v in means]])

    def log_slip_diagnostics(self, update_idx: int, total_steps: int,
                             counts, force_sum, offset_sum):
        """Append per-foot mismatch stats. Skips foot rows with zero events."""
        try:
            c = counts.cpu().numpy()
            f = force_sum.cpu().numpy()
            o = offset_sum.cpu().numpy()
        except AttributeError:
            c = np.asarray(counts); f = np.asarray(force_sum); o = np.asarray(offset_sum)
        if int(c.sum()) == 0:
            return
        if not self._slip_csv_initialised:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            new_file = not self.slip_csv_path.exists()
            with self.slip_csv_path.open("a", newline="") as fh:
                w = csv.writer(fh)
                if new_file:
                    w.writerow(["update_idx", "total_steps", "foot",
                                "n_mismatches", "mean_force", "mean_abs_offset"])
            self._slip_csv_initialised = True
        with self.slip_csv_path.open("a", newline="") as fh:
            w = csv.writer(fh)
            for foot_idx in range(c.shape[0]):
                n = int(c[foot_idx])
                if n == 0:
                    continue
                w.writerow([update_idx, total_steps, foot_idx, n,
                            f"{f[foot_idx] / n:.6g}",
                            f"{o[foot_idx] / n:.6g}"])

    def save_reward_curve(self, plot_dir: str, total_steps: int):
        """Save reward curve plot (same as rl/callbacks.py)."""
        rewards = np.array(self.reward_history)
        timesteps = np.array(self.timestep_history)
        if len(rewards) < 2:
            return

        plot_path = Path(plot_dir)
        plot_path.mkdir(parents=True, exist_ok=True)

        window = max(1, min(50, len(rewards) // 4))
        kernel = np.ones(window) / window
        smooth_r = np.convolve(rewards, kernel, mode="valid")
        smooth_s = timesteps[window - 1:]

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(smooth_s, smooth_r, linewidth=0.8)
        ax.set_ylabel("Episode Return")
        ax.set_xlabel("Timesteps")
        ax.set_title(f"Training Progress ({total_steps:,} steps) [Isaac Lab]")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path / "reward_curve.png", dpi=100)
        plt.close(fig)
