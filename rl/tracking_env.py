"""Go2 trajectory tracking environment following OPT-Mimic.

Trains an RL policy to track a single MPC trajectory in closed loop.
Adapted from Solo 8 (8 joints, 1.7kg) to Go2 (12 joints, 15kg).

Observation (150D): sensor(30) + sensor_history(84) + action_history(36)
Action (12D): residual joint position corrections
Actuation: PD controller + J^T·F feedforward
Reward: 5 Gaussian-exponential tracking terms (OPT-Mimic)
"""

from __future__ import annotations

from collections import deque
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

    TORQUE_LIMIT = 33.5  # Nm
    ACTION_LIMIT = 2.7  # rad (OPT-Mimic actuation_limit)

    # Reward sigmas
    SIGMA_POS = np.sqrt(0.1)
    SIGMA_ORI = np.sqrt(0.1)
    SIGMA_JOINT = np.sqrt(0.5)
    SIGMA_SMOOTH = np.sqrt(0.2)
    SIGMA_TORQUE = 15.0

    # Reward weights (sum to 1.0)
    W_POS = 0.3
    W_ORI = 0.3
    W_JOINT = 0.2
    W_SMOOTH = 0.1
    W_TORQUE = 0.1

    TERM_MULTIPLIER = 2.5  # early termination threshold
    SENSOR_DIM = 28  # quat(4) + joints(12) + joint_vel(12)
    OBS_DIM = 150  # sensor(30) + hist(84) + action_hist(36)

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
        # Paper: μ=0.8, σ=0.25 → ±1σ range ≈ [0.55, 1.05]
        ground_friction = (0.55, 1.05) if randomize else 0.8
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
        self._sensor_history: deque[np.ndarray] = deque(maxlen=3)
        self._action_history: deque[np.ndarray] = deque(maxlen=3)
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
            # Random phase initialization (OPT-Mimic Section III-C.4)
            start_phase = int(self.np_random.integers(0, self.ref.max_phase))
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
        self._last_torque = np.zeros(12)

        # Fill history with initial sensor reading
        init_sensor = self._get_sensor()
        self._sensor_history = deque(
            [init_sensor.copy() for _ in range(3)],
            maxlen=3,
        )
        self._action_history = deque(
            [np.zeros(12) for _ in range(3)],
            maxlen=3,
        )

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

        # Update histories
        self._sensor_history.append(self._get_sensor())
        self._action_history.append(action.copy())

        # Reward and termination
        reward, reward_info = self._compute_reward(sim_obs, action)
        terminated = self._check_termination(reward_info)

        self._phase += 1
        truncated = self._phase >= self.ref.max_phase

        self._prev_action = action.copy()
        obs = self._build_obs()

        return obs, reward, terminated, truncated, {"phase": self._phase, **reward_info}

    def _get_sensor(self) -> np.ndarray:
        """Current sensor: [body_quat(4), joint_pos(12)+offset, joint_vel(12)] = 28D."""
        sim_obs = self._quad_env._get_obs()
        return np.concatenate(
            [
                sim_obs["qpos"][3:7],  # body quat
                sim_obs["qpos"][7:19] + self._joint_offset,  # joints + offset
                sim_obs["qvel"][6:18],  # joint vel
            ]
        )

    def _build_obs(self) -> np.ndarray:
        """Build 150D obs: sensor(28) + phase(2) + sensor_hist(84) + action_hist(36)."""
        return np.concatenate(
            [
                self._get_sensor(),  # 28
                self.ref.get_phase_encoding(self._phase),  # 2
                np.concatenate(list(self._sensor_history)),  # 84
                np.concatenate(list(self._action_history)),  # 36
            ]
        ).astype(np.float32)

    def _compute_reward(
        self, sim_obs: dict[str, Any], action: np.ndarray
    ) -> tuple[float, dict[str, float]]:
        """5-term Gaussian-exponential reward (OPT-Mimic). Total in [0, 1]."""
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

        # Max torque penalty
        max_torque = np.max(np.abs(self._last_torque))
        r_torque = np.exp(-max_torque / self.SIGMA_TORQUE)

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
        }
        return float(total), info

    # GRF z-component indices for each foot in the 12D contact_forces:base vector
    _GRF_Z_INDICES = [2, 5, 8, 11]  # FL_z, FR_z, RL_z, RR_z
    CONTACT_FORCE_THRESHOLD = (
        1.0  # Newtons — foot considered in contact if GRF_z > this
    )
    CONTACT_GRACE_WINDOW = 6  # steps (120ms at 50Hz) — tolerance near transitions

    def _check_termination(self, info: dict[str, float]) -> bool:
        """Early termination if any error exceeds 2.5× its sigma."""
        if info["pos_error"] > self.TERM_MULTIPLIER * self.SIGMA_POS:
            return True
        if info["ori_error"] > self.TERM_MULTIPLIER * self.SIGMA_ORI:
            return True
        if info["joint_error"] > self.TERM_MULTIPLIER * self.SIGMA_JOINT:
            return True
        if info["action_rate"] > self.TERM_MULTIPLIER * self.SIGMA_SMOOTH:
            return True
        if info["max_torque"] > self.TERM_MULTIPLIER * self.SIGMA_TORQUE:
            return True

        # Fall detection
        if self._quad_env._get_obs()["qpos"][2] < 0.05:
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
