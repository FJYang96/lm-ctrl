"""Reference trajectory wrapper for OPT-Mimic tracking.

Wraps a single MPC solution into a phase-indexed reference. The policy
doesn't see this directly — it only gets a phase encoding in its obs.
The environment uses this internally for reward computation, PD targets,
and feedforward torques.
"""

from __future__ import annotations

import numpy as np

from utils.conversion import (
    MPC_X_BASE_ANG,
    MPC_X_BASE_EUL,
    MPC_X_BASE_POS,
    MPC_X_BASE_VEL,
    MPC_X_Q_JOINTS,
    euler_to_quaternion,
)


class ReferenceTrajectory:
    """Phase-indexed reference for a single MPC trajectory.

    Phase 0 = start, phase N-1 = end. Clamped past the end.

    Attributes:
        state_traj: MPC state trajectory (N+1, 30).
        joint_vel_traj: Joint velocities (N, 12).
        grf_traj: Ground reaction forces (N, 12).
        max_phase: Number of control steps (N).
        control_dt: Policy timestep in seconds.
        duration: Total trajectory time in seconds.
    """

    def __init__(
        self,
        state_traj: np.ndarray,
        joint_vel_traj: np.ndarray,
        grf_traj: np.ndarray,
        feedforward_torques: np.ndarray | None = None,
        control_dt: float = 0.02,
    ):
        self.state_traj = state_traj.copy()
        self.joint_vel_traj = joint_vel_traj.copy()
        self.grf_traj = grf_traj.copy()
        self.control_dt = control_dt
        self.max_phase = joint_vel_traj.shape[0]
        self.duration = self.max_phase * control_dt

        # Precompute quaternions from MPC Euler angles
        self._body_quats = np.zeros((self.max_phase + 1, 4))
        for k in range(self.max_phase + 1):
            self._body_quats[k] = euler_to_quaternion(state_traj[k, MPC_X_BASE_EUL])

        # Feedforward torques (J^T·F), set via set_feedforward() if not provided
        if feedforward_torques is not None:
            self._ff_torques = feedforward_torques.copy()
        else:
            self._ff_torques = np.zeros((self.max_phase, 12))

    def set_feedforward(self, feedforward_torques: np.ndarray) -> None:
        """Set precomputed J^T·F feedforward torques (N, 12)."""
        assert feedforward_torques.shape == (self.max_phase, 12)
        self._ff_torques = feedforward_torques.copy()

    def _clamp_phase(self, phase: int) -> int:
        return int(max(0, min(phase, self.max_phase - 1)))

    def get_body_pos(self, phase: int) -> np.ndarray:
        """Reference COM position (3,)."""
        return self.state_traj[self._clamp_phase(phase), MPC_X_BASE_POS].copy()

    def get_body_quat(self, phase: int) -> np.ndarray:
        """Reference quaternion [w,x,y,z] (4,)."""
        return self._body_quats[self._clamp_phase(phase)].copy()

    def get_joint_pos(self, phase: int) -> np.ndarray:
        """Reference joint angles (12,)."""
        return self.state_traj[self._clamp_phase(phase), MPC_X_Q_JOINTS].copy()

    def get_joint_vel(self, phase: int) -> np.ndarray:
        """Reference joint velocities (12,)."""
        return self.joint_vel_traj[self._clamp_phase(phase)].copy()

    def get_body_vel(self, phase: int) -> np.ndarray:
        """Reference body linear velocity (3,)."""
        return self.state_traj[self._clamp_phase(phase), MPC_X_BASE_VEL].copy()

    def get_body_ang_vel(self, phase: int) -> np.ndarray:
        """Reference body angular velocity (3,)."""
        return self.state_traj[self._clamp_phase(phase), MPC_X_BASE_ANG].copy()

    def get_feedforward_torque(self, phase: int) -> np.ndarray:
        """Precomputed feedforward torque J^T·F (12,) in Nm."""
        return self._ff_torques[self._clamp_phase(phase)].copy()

    def get_phase_encoding(self, phase: int) -> np.ndarray:
        """Sinusoidal phase encoding [cos, sin] (2,). Matches OPT-Mimic."""
        angle = 2.0 * np.pi * phase / self.max_phase
        return np.array([np.cos(angle), np.sin(angle)])

    @classmethod
    def from_files(
        cls,
        state_traj_path: str,
        joint_vel_traj_path: str,
        grf_traj_path: str,
        control_dt: float = 0.02,
    ) -> ReferenceTrajectory:
        """Load from .npy files (e.g. results/ directory)."""
        return cls(
            state_traj=np.load(state_traj_path),
            joint_vel_traj=np.load(joint_vel_traj_path),
            grf_traj=np.load(grf_traj_path),
            control_dt=control_dt,
        )
