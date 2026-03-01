"""Go2 trajectory tracking environment following OPT-Mimic.

Trains an RL policy to track a single MPC trajectory in closed loop.
Adapted from Solo 8 (8 joints, 1.7kg) to Go2 (12 joints, 15kg).

Observation (39D): body_pos(3) + body_quat(4) + joints(12) + body_vel(6) + joint_vel(12) + phase(2)
Action (12D): residual joint position corrections
Actuation: PD controller + J^T·F feedforward
Reward: Additive weighted sum of 5 Gaussian terms (OPT-Mimic Eq. 15-16, Table I)
"""

from __future__ import annotations

from typing import Any

import gymnasium
import mujoco
import numpy as np
from gym_quadruped.quadruped_env import QuadrupedEnv
from gymnasium import spaces

from .reference import ReferenceTrajectory


class Go2TrackingEnv(gymnasium.Env):  # type: ignore[misc]
    """OPT-Mimic tracking environment for Go2."""

    # PD gains — scaled from OPT-Mimic (Kp=3, Kd=0.3) by mass ratio ~9×
    KP = 25.0
    KD = 1.5

    TORQUE_LIMIT = 55.0  # Nm — must exceed max feedforward (49.2 Nm at landing)
    ACTION_LIMIT = 0.5  # rad — Kp×0.5=12.5Nm residual for takeoff compensation

    # Reward sigmas — OPT-Mimic Table I, joint/smooth/torque scaled for Go2
    SIGMA_POS = 0.10  # tighter: with correct torques the robot should actually hop
    SIGMA_ORI = 0.14  # Solo 8 value
    SIGMA_JOINT = 0.5  # was 0.3; Go2 joint errors ~2× larger (heavier limbs)
    SIGMA_SMOOTH = 1.0  # scaled for ACTION_LIMIT=0.5: max action rate ≈ 1.0
    SIGMA_TORQUE = 40.0  # scaled for 55Nm limit; Gaussian=0.36 at clamp, gives gradient

    # Reward weights for additive weighted sum (OPT-Mimic Table I, sum to 1.0)
    W_POS = 0.3
    W_ORI = 0.3
    W_JOINT = 0.2
    W_SMOOTH = 0.1
    W_TORQUE = 0.1

    TERM_MULTIPLIER = 2.5  # early termination threshold
    SENSOR_DIM = 37  # body_pos(3) + quat(4) + joints(12) + body_vel(6) + joint_vel(12)
    OBS_DIM = 39  # sensor(37) + phase(2) — full OPT-Mimic obs

    def __init__(
        self,
        ref: ReferenceTrajectory,
        sim_dt: float = 0.001,
        control_dt: float = 0.02,
        randomize: bool = True,
    ):
        super().__init__()
        self.ref = ref
        self.sim_dt = sim_dt
        self.control_dt = control_dt
        self.substeps = int(control_dt / sim_dt)
        self.randomize = randomize

        # Friction randomization: tuple enables per-reset sampling (OPT-Mimic Table I)
        # Centered on config μ=0.5 with σ=0.25 → ±1σ range ≈ [0.25, 0.75]
        ground_friction = (0.25, 0.75) if randomize else 0.5
        self._quad_env = QuadrupedEnv(
            robot="go2",
            scene="flat",
            ground_friction_coeff=ground_friction,
            state_obs_names=QuadrupedEnv._DEFAULT_OBS + ("contact_forces:base",),
            sim_dt=sim_dt,
        )

        # Cache ground geom IDs for restitution randomization
        self._ground_geom_ids = self._find_ground_geom_ids()

        # [-1, 1] action space, scaled by ACTION_LIMIT in step() (OPT-Mimic tanh output)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.OBS_DIM,),
            dtype=np.float32,
        )

        # Runtime state (set in reset)
        self._phase: int = 0
        self._prev_action = np.zeros(12)
        self._last_torque = np.zeros(12)
        self._joint_offset = np.zeros(12)
        self._torque_scale = 1.0
        # Store default solref for restoring when randomize=False
        self._default_solref = self._quad_env.mjModel.geom_solref.copy()

    def _find_ground_geom_ids(self) -> list[int]:
        """Find MuJoCo geom IDs for the ground surface."""
        ground_names = {"ground", "floor", "hfield", "terrain"}
        ids = []
        model = self._quad_env.mjModel
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if name and name.lower() in ground_names:
                ids.append(gid)
        return ids

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # Domain randomization
        if self.randomize and self.np_random is not None:
            self._joint_offset = self.np_random.normal(0.0, 0.02, size=12)
            self._torque_scale = float(
                np.clip(self.np_random.normal(1.0, 0.1), 0.5, 1.5)
            )
            # Restitution randomization (OPT-Mimic Table I: μ=0.0, σ=0.25, clipped [0,1])
            # MuJoCo solref[1] is damping ratio: 1.0=no bounce, lower=bouncier
            restitution = float(np.clip(self.np_random.normal(0.0, 0.25), 0.0, 1.0))
            damping_ratio = 1.0 - restitution  # map restitution→damping
            for gid in self._ground_geom_ids:
                self._quad_env.mjModel.geom_solref[gid, 1] = damping_ratio
            # Random phase initialization — restrict to stance phases (0-14)
            # Starting mid-flight is unrecoverable for hop trajectories
            max_start = min(15, self.ref.max_phase)
            start_phase = int(self.np_random.integers(0, max_start))
        else:
            self._joint_offset = np.zeros(12)
            self._torque_scale = 1.0
            # Restore default solref
            self._quad_env.mjModel.geom_solref[:] = self._default_solref
            start_phase = 0

        # Reset MuJoCo to reference state at start_phase
        init_qpos = np.zeros(19)
        init_qpos[0:3] = self.ref.get_body_pos(start_phase)
        init_qpos[3:7] = self.ref.get_body_quat(start_phase)
        init_qpos[7:19] = self.ref.get_joint_pos(start_phase)

        init_qvel = np.zeros(18)
        init_qvel[0:3] = self.ref.get_body_vel(start_phase)
        init_qvel[3:6] = self.ref.get_body_ang_vel(start_phase)
        init_qvel[6:18] = self.ref.get_joint_vel(start_phase)

        self._quad_env.reset(qpos=init_qpos, qvel=init_qvel)

        self._phase = start_phase
        self._prev_action = np.zeros(12)
        self._first_step = True  # skip action_rate termination on first step
        self._last_torque = np.zeros(12)

        return self._build_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Scale [-1, 1] to [-ACTION_LIMIT, ACTION_LIMIT] (OPT-Mimic tanh output)
        action = action * self.ACTION_LIMIT

        # PD + feedforward torque computation
        ref_joint_pos = self.ref.get_joint_pos(self._phase)
        ref_joint_vel = self.ref.get_joint_vel(self._phase)
        ff_torque = self.ref.get_feedforward_torque(self._phase)

        sim_obs = self._quad_env._get_obs()
        actual_joint_pos = sim_obs["qpos"][7:19]
        actual_joint_vel = sim_obs["qvel"][6:18]

        target_pos = ref_joint_pos + action + self._joint_offset
        torque = (
            self.KP * (target_pos - actual_joint_pos)
            + self.KD * (ref_joint_vel - actual_joint_vel)
            + ff_torque
        )
        torque = np.clip(torque, -self.TORQUE_LIMIT, self.TORQUE_LIMIT)
        torque = torque * self._torque_scale
        self._last_torque = torque.copy()

        # Step MuJoCo (substeps per policy step)
        for _ in range(self.substeps):
            sim_obs, _, _, _, _ = self._quad_env.step(action=torque)

        # Reward and termination
        reward, reward_info = self._compute_reward(sim_obs, action)
        terminated = self._check_termination(reward_info)

        self._first_step = False
        self._phase += 1
        truncated = self._phase >= self.ref.max_phase

        self._prev_action = action.copy()
        obs = self._build_obs()

        return obs, reward, terminated, truncated, {"phase": self._phase, **reward_info}

    def _get_sensor(self) -> np.ndarray:
        """Current sensor: [body_pos(3), body_quat(4), joints(12), body_vel(6), joint_vel(12)] = 37D."""
        sim_obs = self._quad_env._get_obs()
        return np.concatenate(
            [
                sim_obs["qpos"][0:3],  # body position (xyz)
                sim_obs["qpos"][3:7],  # body quat
                sim_obs["qpos"][7:19] + self._joint_offset,  # joints + offset
                sim_obs["qvel"][0:6],  # body velocity (lin 3 + ang 3)
                sim_obs["qvel"][6:18],  # joint vel
            ]
        )

    def _build_obs(self) -> np.ndarray:
        """Build 39D obs: sensor(37) + phase(2) — full OPT-Mimic."""
        return np.concatenate(
            [
                self._get_sensor(),  # 37
                self.ref.get_phase_encoding(self._phase),  # 2
            ]
        ).astype(np.float32)

    def _compute_reward(
        self, sim_obs: dict[str, Any], action: np.ndarray
    ) -> tuple[float, dict[str, Any]]:
        """Additive weighted sum of 5 Gaussian terms (OPT-Mimic Eq. 15). Total in [0, 1]."""
        qpos = sim_obs["qpos"]

        # Position tracking
        pos_err_sq = np.sum((self.ref.get_body_pos(self._phase) - qpos[0:3]) ** 2)
        r_pos = np.exp(-pos_err_sq / (2.0 * self.SIGMA_POS**2))

        # Orientation tracking (quaternion error vector part)
        ori_err = self._quat_error(self.ref.get_body_quat(self._phase), qpos[3:7])
        ori_err_sq = np.sum(ori_err**2)
        r_ori = np.exp(-ori_err_sq / (2.0 * self.SIGMA_ORI**2))

        # Joint tracking
        joint_err_sq = np.sum((self.ref.get_joint_pos(self._phase) - qpos[7:19]) ** 2)
        r_joint = np.exp(-joint_err_sq / (2.0 * self.SIGMA_JOINT**2))

        # Action smoothness
        rate_sq = np.sum((action - self._prev_action) ** 2)
        r_smooth = np.exp(-rate_sq / (2.0 * self.SIGMA_SMOOTH**2))

        # Max torque penalty (Gaussian, same as other terms)
        max_torque = np.max(np.abs(self._last_torque))
        r_torque = np.exp(-(max_torque**2) / (2.0 * self.SIGMA_TORQUE**2))

        # Additive weighted sum (OPT-Mimic Eq. 15)
        total = (
            self.W_POS * r_pos
            + self.W_ORI * r_ori
            + self.W_JOINT * r_joint
            + self.W_SMOOTH * r_smooth
            + self.W_TORQUE * r_torque
        )

        info = {
            "pos_error": np.sqrt(pos_err_sq),
            "ori_error": np.sqrt(ori_err_sq),
            "joint_error": np.sqrt(joint_err_sq),
            "action_rate": np.sqrt(rate_sq),
            "max_torque": max_torque,
            # Individual reward components (before weighting)
            "rw_pos": float(r_pos),
            "rw_ori": float(r_ori),
            "rw_joint": float(r_joint),
            "rw_smooth": float(r_smooth),
            "rw_torque": float(r_torque),
        }
        return float(total), info

    # GRF z-component indices for each foot in the 12D contact_forces:base vector
    _GRF_Z_INDICES = [2, 5, 8, 11]  # FL_z, FR_z, RL_z, RR_z
    CONTACT_FORCE_THRESHOLD = (
        1.0  # Newtons — foot considered in contact if GRF_z > this
    )
    CONTACT_GRACE_WINDOW = 6  # steps (120ms at 50Hz) — tolerance near transitions

    def _check_termination(self, info: dict[str, Any]) -> bool:
        """Early termination if any error exceeds 2.5× its sigma.

        Sets info["termination_reason"] to the cause for diagnostics.
        """
        thresh_pos = self.TERM_MULTIPLIER * self.SIGMA_POS
        thresh_ori = self.TERM_MULTIPLIER * self.SIGMA_ORI
        thresh_joint = self.TERM_MULTIPLIER * self.SIGMA_JOINT
        thresh_smooth = self.TERM_MULTIPLIER * self.SIGMA_SMOOTH
        thresh_torque = self.TERM_MULTIPLIER * self.SIGMA_TORQUE

        if info["pos_error"] > thresh_pos:
            info["termination_reason"] = (
                f"pos_error {info['pos_error']:.4f} > {thresh_pos:.4f}"
            )
            return True
        if info["ori_error"] > thresh_ori:
            info["termination_reason"] = (
                f"ori_error {info['ori_error']:.4f} > {thresh_ori:.4f}"
            )
            return True
        if info["joint_error"] > thresh_joint:
            info["termination_reason"] = (
                f"joint_error {info['joint_error']:.4f} > {thresh_joint:.4f}"
            )
            return True
        if not self._first_step and info["action_rate"] > thresh_smooth:
            info["termination_reason"] = (
                f"action_rate {info['action_rate']:.4f} > {thresh_smooth:.4f}"
            )
            return True
        if info["max_torque"] > thresh_torque:
            info["termination_reason"] = (
                f"max_torque {info['max_torque']:.4f} > {thresh_torque:.4f}"
            )
            return True

        # Fall detection
        body_height = self._quad_env._get_obs()["qpos"][2]
        if body_height < 0.05:
            info["termination_reason"] = f"fall: height {body_height:.4f} < 0.05"
            return True

        # Contact consistency termination (OPT-Mimic Section III-C.4)
        if self.ref.contact_sequence is not None:
            sim_obs = self._quad_env._get_obs()
            grf = sim_obs["contact_forces:base"]
            actual_contact = np.array(
                [grf[i] > self.CONTACT_FORCE_THRESHOLD for i in self._GRF_Z_INDICES]
            )
            expected_contact = self.ref.get_contact_state(self._phase) > 0.5
            for foot in range(4):
                if actual_contact[foot] != expected_contact[foot]:
                    # Allow mismatch near contact transitions
                    if not self.ref.is_near_contact_transition(
                        self._phase, foot, self.CONTACT_GRACE_WINDOW
                    ):
                        foot_names = ["FL", "FR", "RL", "RR"]
                        info["termination_reason"] = (
                            f"contact_mismatch: {foot_names[foot]} "
                            f"actual={actual_contact[foot]} "
                            f"expected={expected_contact[foot]}"
                        )
                        return True

        return False

    @staticmethod
    def _quat_error(q_ref: np.ndarray, q_actual: np.ndarray) -> np.ndarray:
        """Quaternion error: vector part of q_ref * q_actual^{-1}. Returns (3,)."""
        q_inv = np.array([q_actual[0], -q_actual[1], -q_actual[2], -q_actual[3]])
        w1, x1, y1, z1 = q_ref
        w2, x2, y2, z2 = q_inv
        return np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )
