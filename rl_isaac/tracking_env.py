"""Go2 OPT-Mimic tracking environment for Isaac Lab (DirectRLEnv).

Faithful port of rl/tracking_env.py from MJX to Isaac Lab.
All observations, rewards, termination conditions, actuation, and domain
randomization match the MJX implementation exactly.
"""

from __future__ import annotations

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .env_cfg import Go2TrackingEnvCfg

# ---------------------------------------------------------------------------
# Constants — identical to rl/tracking_env.py
# ---------------------------------------------------------------------------
KP = 25.0
KD = 1.5
TORQUE_LIMITS = torch.tensor([
    23.7, 23.7, 45.43,  # FL
    23.7, 23.7, 45.43,  # FR
    23.7, 23.7, 45.43,  # RL
    23.7, 23.7, 45.43,  # RR
], dtype=torch.float32)
ACTION_LIMIT = 0.2

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

TERM_MULTIPLIER = 2.5
CONTACT_GRACE_WINDOW = 12  # 240ms at 50Hz


def _quat_error_vec(q_ref: torch.Tensor, q_actual: torch.Tensor) -> torch.Tensor:
    """Batched quaternion error: vector part of q_ref * q_actual^{-1}.
    Input: (N, 4) [w, x, y, z]. Output: (N, 3).
    Matches rl/tracking_env.py _quat_error exactly.
    """
    # q_inv = conjugate of q_actual
    q_inv = q_actual.clone()
    q_inv[:, 1:] = -q_inv[:, 1:]

    w1, x1, y1, z1 = q_ref[:, 0], q_ref[:, 1], q_ref[:, 2], q_ref[:, 3]
    w2, x2, y2, z2 = q_inv[:, 0], q_inv[:, 1], q_inv[:, 2], q_inv[:, 3]

    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


class Go2TrackingEnv(DirectRLEnv):
    """Go2 trajectory tracking environment using OPT-Mimic.

    Faithfully replicates rl/tracking_env.py in Isaac Lab.
    """

    cfg: Go2TrackingEnvCfg

    def __init__(self, cfg: Go2TrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._torque_limits = TORQUE_LIMITS.to(self.device)

        # Actions buffer (initialized for _compute_tracking_errors on first call)
        self.actions = torch.zeros(self.num_envs, 12, device=self.device)

        # Per-env state tensors
        self._phase = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, 12, device=self.device)
        self._last_torque = torch.zeros(self.num_envs, 12, device=self.device)
        self._first_step = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._joint_offset = torch.zeros(self.num_envs, 12, device=self.device)
        self._torque_scale = torch.ones(self.num_envs, device=self.device)

        # Tracking errors (computed in _compute_tracking_errors, used by _get_dones and _get_rewards)
        self._tracking_errors: dict[str, torch.Tensor] = {}

        # Load reference trajectory data
        self._load_reference_data()

        # Find body indices for contact termination
        self._setup_contact_indices()

        # Store env origins for position offset correction
        # Isaac Lab root_pos_w includes env_origin offset; our reference is relative to (0,0,0)
        self._env_origins = self._terrain.env_origins.clone()  # (N, 3)

        # Joint ordering verification
        joint_names = self._robot.joint_names
        print(f"[Go2TrackingEnv] Joint names: {joint_names}")
        print(f"[Go2TrackingEnv] Num envs: {self.num_envs}, max_phase: {self._max_phase}")

        # Build joint reorder map if needed (Isaac Lab order -> MPC order)
        self._joint_reorder = self._build_joint_reorder(joint_names)

    def _build_joint_reorder(self, joint_names: list[str]) -> torch.Tensor | None:
        """Build reorder indices if Isaac Lab joint order differs from MPC order.

        MPC order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
                   RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
        """
        mpc_order = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        if joint_names == mpc_order:
            print("[Go2TrackingEnv] Joint ordering matches MPC — no reorder needed.")
            return None

        # Build reorder: isaac_idx[i] = mpc_idx[reorder[i]]
        reorder = []
        for mpc_name in mpc_order:
            try:
                isaac_idx = joint_names.index(mpc_name)
                reorder.append(isaac_idx)
            except ValueError:
                raise RuntimeError(
                    f"MPC joint '{mpc_name}' not found in Isaac Lab joints: {joint_names}"
                )
        idx = torch.tensor(reorder, dtype=torch.long, device=self.device)
        print(f"[Go2TrackingEnv] Joint reorder: MPC->Isaac = {reorder}")
        return idx

    def _to_isaac_order(self, mpc_joints: torch.Tensor) -> torch.Tensor:
        """Convert (N, 12) from MPC order to Isaac Lab joint order.

        reorder[i] = Isaac index for MPC joint i.
        To scatter MPC data into Isaac order: result[:, reorder] = mpc_data
        Equivalently: result = mpc_data[:, inverse_reorder]
        """
        if self._joint_reorder is None:
            return mpc_joints
        inv = torch.argsort(self._joint_reorder)
        return mpc_joints[:, inv]

    def _to_mpc_order(self, isaac_joints: torch.Tensor) -> torch.Tensor:
        """Convert (N, 12) from Isaac Lab joint order to MPC order.

        reorder[i] = Isaac index for MPC joint i.
        To gather Isaac data into MPC order: result[:, i] = isaac_data[:, reorder[i]]
        """
        if self._joint_reorder is None:
            return isaac_joints
        return isaac_joints[:, self._joint_reorder]

    def _load_reference_data(self):
        """Load reference trajectory from numpy files and convert to GPU tensors."""
        import config
        from rl.reference import ReferenceTrajectory
        from rl.feedforward import FeedforwardComputer
        from mpc.dynamics.model import KinoDynamic_Model

        cfg = self.cfg

        ref = ReferenceTrajectory.from_files(
            cfg.state_traj_path,
            cfg.joint_vel_traj_path,
            cfg.grf_traj_path,
            contact_sequence_path=cfg.contact_sequence_path if cfg.contact_sequence_path else None,
            control_dt=0.02,
        )

        # Precompute feedforward torques
        kindyn = KinoDynamic_Model(config)
        ff = FeedforwardComputer(kindyn)
        ref.set_feedforward(ff.precompute_trajectory(ref))

        N = ref.max_phase
        self._max_phase = N

        # Build reference arrays
        body_pos = np.zeros((N + 1, 3), dtype=np.float32)
        body_quat = np.zeros((N + 1, 4), dtype=np.float32)
        joint_pos = np.zeros((N + 1, 12), dtype=np.float32)
        body_vel = np.zeros((N + 1, 3), dtype=np.float32)
        body_ang_vel = np.zeros((N + 1, 3), dtype=np.float32)
        for k in range(N + 1):
            body_pos[k] = ref.get_body_pos(k)
            body_quat[k] = ref.get_body_quat(k)
            joint_pos[k] = ref.get_joint_pos(k)
            body_vel[k] = ref.get_body_vel(k)
            body_ang_vel[k] = ref.get_body_ang_vel(k)

        joint_vel = np.zeros((N, 12), dtype=np.float32)
        ff_torques = np.zeros((N, 12), dtype=np.float32)
        for k in range(N):
            joint_vel[k] = ref.get_joint_vel(k)
            ff_torques[k] = ref.get_feedforward_torque(k)

        # Contact sequence
        if ref.contact_sequence is not None:
            contact_seq = ref.contact_sequence[:, :N].astype(np.float32)
        else:
            contact_seq = -np.ones((4, N), dtype=np.float32)

        # Near-transition mask (120ms grace window)
        near_transition = np.ones((4, N), dtype=np.float32)
        if ref.contact_sequence is not None:
            near_transition = np.zeros((4, N), dtype=np.float32)
            for foot in range(4):
                for k in range(N):
                    current = ref.contact_sequence[foot, k]
                    lo = max(0, k - CONTACT_GRACE_WINDOW)
                    hi = min(N - 1, k + CONTACT_GRACE_WINDOW)
                    if np.any(ref.contact_sequence[foot, lo:hi + 1] != current):
                        near_transition[foot, k] = 1.0

        # Convert to GPU tensors
        self._ref_body_pos = torch.tensor(body_pos, device=self.device)
        self._ref_body_quat = torch.tensor(body_quat, device=self.device)
        self._ref_joint_pos = torch.tensor(joint_pos, device=self.device)
        self._ref_joint_vel = torch.tensor(joint_vel, device=self.device)
        self._ref_body_vel = torch.tensor(body_vel, device=self.device)
        self._ref_body_ang_vel = torch.tensor(body_ang_vel, device=self.device)
        self._ref_ff_torques = torch.tensor(ff_torques, device=self.device)
        self._ref_contact_seq = torch.tensor(contact_seq, device=self.device)
        self._ref_near_transition = torch.tensor(near_transition, device=self.device)

        # Store ref for evaluation
        self._ref = ref

    def _setup_contact_indices(self):
        """Identify foot and non-foot body indices for contact termination."""
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        # Try to find foot bodies in the contact sensor
        self._foot_body_ids, _ = self._contact_sensor.find_bodies("|".join(foot_names))
        # All body IDs
        all_body_ids = list(range(self._contact_sensor.num_bodies))
        foot_set = set(self._foot_body_ids.tolist()) if isinstance(self._foot_body_ids, torch.Tensor) else set(self._foot_body_ids)
        self._non_foot_body_ids = torch.tensor(
            [i for i in all_body_ids if i not in foot_set],
            dtype=torch.long, device=self.device,
        )
        self._foot_body_ids_t = torch.tensor(
            list(foot_set), dtype=torch.long, device=self.device,
        )
        print(f"[Go2TrackingEnv] Foot body IDs: {self._foot_body_ids_t.tolist()}")
        print(f"[Go2TrackingEnv] Non-foot body IDs: {self._non_foot_body_ids.tolist()} ({len(self._non_foot_body_ids)} bodies)")

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Pre-physics step (store actions)
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    # ------------------------------------------------------------------
    # Action application — PD + feedforward + residual (OPT-Mimic Eq. 14)
    # ------------------------------------------------------------------

    def _apply_action(self):
        action_scaled = self.actions * ACTION_LIMIT  # (N, 12) in MPC order

        phase = self._phase.clamp(0, self._max_phase - 1)

        # Reference targets (in MPC joint order)
        ref_joint_pos = self._ref_joint_pos[phase]    # (N, 12)
        ref_joint_vel = self._ref_joint_vel[phase]    # (N, 12)
        ff_torque = self._ref_ff_torques[phase]       # (N, 12)

        # Actual state (in Isaac Lab joint order -> convert to MPC order)
        actual_joint_pos = self._to_mpc_order(self._robot.data.joint_pos)  # (N, 12)
        actual_joint_vel = self._to_mpc_order(self._robot.data.joint_vel)  # (N, 12)

        target_pos = ref_joint_pos + action_scaled + self._joint_offset
        torque = (
            KP * (target_pos - actual_joint_pos)
            + KD * (ref_joint_vel - actual_joint_vel)
            + ff_torque
        )
        torque = torque.clamp(-self._torque_limits, self._torque_limits)
        torque = torque * self._torque_scale.unsqueeze(-1)

        self._last_torque = torque.clone()

        # Convert to Isaac Lab joint order and apply
        torque_isaac = self._to_isaac_order(torque)
        self._robot.set_joint_effort_target(torque_isaac)

    # ------------------------------------------------------------------
    # Observations — 39D (matches rl/tracking_env.py get_obs exactly)
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        phase = self._phase.clamp(0, self._max_phase - 1)

        root_pos = self._robot.data.root_pos_w - self._env_origins  # (N, 3) local position
        root_quat = self._robot.data.root_quat_w                    # (N, 4) [w,x,y,z]
        joint_pos = self._to_mpc_order(self._robot.data.joint_pos)  # (N, 12)
        root_lin_vel = self._robot.data.root_lin_vel_w           # (N, 3)
        root_ang_vel = self._robot.data.root_ang_vel_w           # (N, 3)
        joint_vel = self._to_mpc_order(self._robot.data.joint_vel)  # (N, 12)

        # Apply joint offset (domain randomization)
        joint_pos_observed = joint_pos + self._joint_offset

        # Phase encoding
        angle = 2.0 * torch.pi * phase.float() / float(self._max_phase)
        phase_enc = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)

        obs = torch.cat([
            root_pos, root_quat, joint_pos_observed,
            root_lin_vel, root_ang_vel, joint_vel,
            phase_enc,
        ], dim=-1)  # (N, 39)

        # Replace NaN with 0
        obs = torch.nan_to_num(obs, nan=0.0)

        return {"policy": obs}

    # ------------------------------------------------------------------
    # Rewards — 5 Gaussian terms (OPT-Mimic Eq. 15-16)
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        # Use precomputed tracking errors from _compute_tracking_errors()
        te = self._tracking_errors

        r_pos = torch.exp(-te["pos_err_sq"] / (2.0 * SIGMA_POS ** 2))
        r_ori = torch.exp(-te["ori_err_sq"] / (2.0 * SIGMA_ORI ** 2))
        r_joint = torch.exp(-te["joint_err_sq"] / (2.0 * SIGMA_JOINT ** 2))
        r_smooth = torch.exp(-te["rate_sq"] / (2.0 * SIGMA_SMOOTH ** 2))
        r_torque = torch.exp(-(te["max_torque"] ** 2) / (2.0 * SIGMA_TORQUE ** 2))

        total = W_POS * r_pos + W_ORI * r_ori + W_JOINT * r_joint + W_SMOOTH * r_smooth + W_TORQUE * r_torque

        # Force reward to 0 if NaN
        total = torch.nan_to_num(total, nan=0.0)

        # (tracking errors already stored in self._tracking_errors)

        # Log reward components via extras
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update({
            "r_pos": r_pos.mean().item(),
            "r_ori": r_ori.mean().item(),
            "r_joint": r_joint.mean().item(),
            "r_smooth": r_smooth.mean().item(),
            "r_torque": r_torque.mean().item(),
        })

        return total

    # ------------------------------------------------------------------
    # Compute tracking errors (shared by rewards and dones)
    # ------------------------------------------------------------------

    def _compute_tracking_errors(self):
        """Compute tracking errors used by both rewards and termination.
        Called once per step before _get_dones and _get_rewards.

        Also increments phase and updates prev_action (matching MJX step() where
        phase is incremented BEFORE reward/done computation).
        """
        # Increment phase (MJX: new_state.phase = state.phase + 1)
        self._phase += 1
        self._first_step[:] = False

        phase = self._phase.clamp(0, self._max_phase - 1)

        ref_pos = self._ref_body_pos[phase]
        ref_quat = self._ref_body_quat[phase]
        ref_joint = self._ref_joint_pos[phase]

        # Subtract env origins from world position to get local position
        actual_pos = self._robot.data.root_pos_w - self._env_origins
        actual_quat = self._robot.data.root_quat_w
        actual_joint = self._to_mpc_order(self._robot.data.joint_pos)

        pos_err_sq = ((ref_pos - actual_pos) ** 2).sum(dim=-1)
        ori_err = _quat_error_vec(ref_quat, actual_quat)
        ori_err_sq = (ori_err ** 2).sum(dim=-1)
        joint_err_sq = ((ref_joint - actual_joint) ** 2).sum(dim=-1)

        # Compute action rate BEFORE updating _prev_action
        action_scaled = self.actions * ACTION_LIMIT
        rate_sq = ((action_scaled - self._prev_action) ** 2).sum(dim=-1)

        # Now update _prev_action for next step
        self._prev_action = action_scaled.clone()

        max_torque = self._last_torque.abs().max(dim=-1).values

        self._tracking_errors = {
            "pos_err_sq": pos_err_sq,
            "ori_err_sq": ori_err_sq,
            "joint_err_sq": joint_err_sq,
            "rate_sq": rate_sq,
            "max_torque": max_torque,
            "pos_error": pos_err_sq.sqrt(),
            "ori_error": ori_err_sq.sqrt(),
            "joint_error": joint_err_sq.sqrt(),
            "action_rate": rate_sq.sqrt(),
        }

    # ------------------------------------------------------------------
    # Done conditions — OPT-Mimic termination
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_tracking_errors()
        ri = self._tracking_errors

        # Condition 1: tracking error thresholds (2.5 * sigma)
        thresh = (
            (ri["pos_error"] > TERM_MULTIPLIER * SIGMA_POS)
            | (ri["ori_error"] > TERM_MULTIPLIER * SIGMA_ORI)
            | (ri["joint_error"] > TERM_MULTIPLIER * SIGMA_JOINT)
            | ((~self._first_step) & (ri["action_rate"] > TERM_MULTIPLIER * SIGMA_SMOOTH))
            | (ri["max_torque"] > TERM_MULTIPLIER * SIGMA_TORQUE)
        )

        # Condition 2: non-foot body ground contact
        non_foot_contact = self._check_body_contact()

        # Condition 3: foot contact mismatch (with 120ms grace window)
        contact_mismatch = self._check_contact_mismatch()

        # Condition 4: NaN check
        has_nan = torch.any(torch.isnan(self._robot.data.root_pos_w), dim=-1)

        terminated = thresh | non_foot_contact | contact_mismatch | has_nan

        # Truncation: episode length exceeded (phase >= max_phase)
        # Phase is already incremented in _compute_tracking_errors above
        truncated = self._phase >= self._max_phase

        # Log termination breakdown
        n_term = terminated.sum().item()
        if n_term > 0:
            if "log" not in self.extras:
                self.extras["log"] = {}
            self.extras["log"]["term_thresh"] = thresh.sum().item()
            self.extras["log"]["term_body"] = non_foot_contact.sum().item()
            self.extras["log"]["term_contact"] = contact_mismatch.sum().item()
            self.extras["log"]["term_nan"] = has_nan.sum().item()
            self.extras["log"]["term_trunc"] = truncated.sum().item()

        return terminated, truncated

    def _check_body_contact(self) -> torch.Tensor:
        """Check if any non-foot body is in contact with the ground."""
        if len(self._non_foot_body_ids) == 0:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        net_forces = self._contact_sensor.data.net_forces_w_history  # (N, history, num_bodies, 3)
        # Take max over history
        non_foot_forces = torch.max(
            torch.norm(net_forces[:, :, self._non_foot_body_ids], dim=-1), dim=1
        )[0]  # (N, num_non_foot)
        return (non_foot_forces > 1.0).any(dim=-1)

    def _check_contact_mismatch(self) -> torch.Tensor:
        """Check foot contact mismatch with reference (120ms grace window)."""
        phase = self._phase.clamp(0, self._max_phase - 1)

        # Reference contact state per foot: (4, N) -> gather -> (N, 4)
        ref_contact = self._ref_contact_seq[:, phase].T  # (N, 4)
        has_info = ref_contact[:, 0] >= 0  # -1 sentinel = unavailable

        if not has_info.any():
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Actual foot contact from sensor
        net_forces = self._contact_sensor.data.net_forces_w_history
        foot_forces = torch.max(
            torch.norm(net_forces[:, :, self._foot_body_ids_t], dim=-1), dim=1
        )[0]  # (N, 4)
        actual_contact = foot_forces > 1.0  # (N, 4) bool
        expected_contact = ref_contact > 0.5  # (N, 4) bool

        # Grace window
        grace = self._ref_near_transition[:, phase].T  # (N, 4)
        mismatch = (actual_contact != expected_contact) & (grace < 0.5)

        return has_info & mismatch.any(dim=-1)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        n = len(env_ids)

        # Domain randomization
        self._joint_offset[env_ids] = torch.randn(n, 12, device=self.device) * 0.02
        self._torque_scale[env_ids] = (1.0 + torch.randn(n, device=self.device) * 0.1).clamp(0.5, 1.5)

        # Random start phase
        start_phase = torch.randint(0, self._max_phase, (n,), device=self.device, dtype=torch.int32)
        self._phase[env_ids] = start_phase

        phase_clamped = start_phase.clamp(0, self._max_phase - 1).long()

        # Set root state from reference
        root_pos = self._ref_body_pos[phase_clamped].clone()
        root_quat = self._ref_body_quat[phase_clamped].clone()
        root_vel = self._ref_body_vel[phase_clamped].clone()
        root_ang_vel = self._ref_body_ang_vel[phase_clamped].clone()
        ref_jpos = self._ref_joint_pos[phase_clamped].clone()
        ref_jvel = self._ref_joint_vel[phase_clamped].clone()

        # Add env origins to position
        root_pos += self._terrain.env_origins[env_ids]

        # Write to sim
        self._robot.write_root_pose_to_sim(
            torch.cat([root_pos, root_quat], dim=-1), env_ids
        )
        self._robot.write_root_velocity_to_sim(
            torch.cat([root_vel, root_ang_vel], dim=-1), env_ids
        )
        # Convert joint data to Isaac Lab order
        self._robot.write_joint_state_to_sim(
            self._to_isaac_order(ref_jpos),
            self._to_isaac_order(ref_jvel),
            None, env_ids,
        )

        # Reset per-env state
        self._prev_action[env_ids] = 0.0
        self._last_torque[env_ids] = 0.0
        self._first_step[env_ids] = True

    # Phase increment and prev_action update are handled in _compute_tracking_errors()
    # which is called at the start of _get_dones() — matching the MJX step order where
    # phase is incremented before reward/done computation.
