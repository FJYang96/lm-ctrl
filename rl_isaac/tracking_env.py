"""Go2 OPT-Mimic tracking environment for Isaac Lab (DirectRLEnv)."""

from __future__ import annotations

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .env_cfg import Go2TrackingEnvCfg
from .rewards import (
    KP, KD, ACTION_LIMIT, CONTACT_GRACE_WINDOW, TERM_CAUSE_NAMES,
    compute_tracking_errors, compute_rewards,
    tracking_termination_breakdown,
    check_body_contact, contact_mismatch_diagnostics,
)

_MPC_JOINT_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]


class Go2TrackingEnv(DirectRLEnv):
    cfg: Go2TrackingEnvCfg

    def __init__(self, cfg: Go2TrackingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        from .rewards import TORQUE_LIMITS
        self._torque_limits = TORQUE_LIMITS.to(self.device)
        self.actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._phase = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, 12, device=self.device)
        self._last_torque = torch.zeros(self.num_envs, 12, device=self.device)
        self._first_step = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._joint_offset = torch.zeros(self.num_envs, 12, device=self.device)
        self._torque_scale = torch.ones(self.num_envs, device=self.device)
        self._tracking_errors: dict[str, torch.Tensor] = {}
        self._load_reference_data()
        self._setup_contact_indices()
        self._env_origins = self._terrain.env_origins.clone()
        self._joint_reorder = self._build_joint_reorder(self._robot.joint_names)
        # Phase-0.1 instrumentation: (cause, phase) histogram, accumulated per
        # rollout. Train.py drains it via flush_term_diagnostics() each update.
        self._term_phase_hist = torch.zeros(
            len(TERM_CAUSE_NAMES), self._max_phase + 1,
            dtype=torch.int64, device=self.device,
        )
        self._term_counts = {n: 0 for n in TERM_CAUSE_NAMES}
        # Phase-0.2 instrumentation: per-phase tracking-error sums for the
        # five metrics tracked by compute_tracking_errors. Drained per update
        # by flush_phase_errors().
        self._phase_err_metrics = (
            "pos_error", "ori_error", "joint_error", "action_rate", "max_torque",
        )
        self._phase_err_sum = torch.zeros(
            self._max_phase + 1, len(self._phase_err_metrics),
            dtype=torch.float64, device=self.device,
        )
        self._phase_err_count = torch.zeros(
            self._max_phase + 1, dtype=torch.int64, device=self.device,
        )
        # Phase-0.3 instrumentation: per-foot contact slip stats.
        self._slip_count_per_foot = torch.zeros(4, dtype=torch.int64, device=self.device)
        self._slip_force_sum_per_foot = torch.zeros(4, dtype=torch.float64, device=self.device)
        self._slip_offset_abs_sum_per_foot = torch.zeros(4, dtype=torch.float64, device=self.device)

    def _build_joint_reorder(self, joint_names: list[str]) -> torch.Tensor | None:
        if joint_names == _MPC_JOINT_ORDER:
            return None
        reorder = []
        for name in _MPC_JOINT_ORDER:
            try:
                reorder.append(joint_names.index(name))
            except ValueError:
                raise RuntimeError(f"MPC joint '{name}' not in Isaac joints: {joint_names}")
        return torch.tensor(reorder, dtype=torch.long, device=self.device)

    def _to_isaac_order(self, mpc_joints: torch.Tensor) -> torch.Tensor:
        if self._joint_reorder is None:
            return mpc_joints
        return mpc_joints[:, torch.argsort(self._joint_reorder)]

    def _to_mpc_order(self, isaac_joints: torch.Tensor) -> torch.Tensor:
        if self._joint_reorder is None:
            return isaac_joints
        return isaac_joints[:, self._joint_reorder]

    def _load_reference_data(self):
        from .reference import ReferenceTrajectory
        from .feedforward import FeedforwardComputer
        from mpc.dynamics.model import KinoDynamic_Model
        from utils.conversion import (
            MPC_X_BASE_POS, MPC_X_BASE_VEL, MPC_X_BASE_EUL,
            MPC_X_BASE_ANG, MPC_X_Q_JOINTS, euler_to_quaternion,
        )
        cfg = self.cfg
        ref = ReferenceTrajectory.from_files(
            cfg.state_traj_path, cfg.joint_vel_traj_path, cfg.grf_traj_path,
            contact_sequence_path=cfg.contact_sequence_path or None, control_dt=0.02,
        )
        ref.set_feedforward(FeedforwardComputer(KinoDynamic_Model()).precompute_trajectory(ref))
        N = ref.max_phase
        self._max_phase = N
        st = ref.state_traj
        body_quat = np.array([euler_to_quaternion(st[k, MPC_X_BASE_EUL]) for k in range(N + 1)], dtype=np.float32)
        if ref.contact_sequence is not None:
            contact_seq = ref.contact_sequence[:, :N].astype(np.float32)
            near_transition = np.zeros((4, N), dtype=np.float32)
            for foot in range(4):
                for k in range(N):
                    lo, hi = max(0, k - CONTACT_GRACE_WINDOW), min(N - 1, k + CONTACT_GRACE_WINDOW)
                    if np.any(ref.contact_sequence[foot, lo:hi + 1] != ref.contact_sequence[foot, k]):
                        near_transition[foot, k] = 1.0
            # Phase-0.3: signed |distance in frames| from each phase to the
            # nearest scheduled contact transition for each foot. Defaults to
            # N (sentinel "no transition exists") for feet with constant
            # schedule.
            transition_offset = np.full((4, N), float(N), dtype=np.float32)
            for foot in range(4):
                cs = ref.contact_sequence[foot, :N]
                trans = [k + 0.5 for k in range(N - 1) if cs[k] != cs[k + 1]]
                if trans:
                    for k in range(N):
                        transition_offset[foot, k] = min(abs(k - t) for t in trans)
        else:
            contact_seq = -np.ones((4, N), dtype=np.float32)
            near_transition = np.ones((4, N), dtype=np.float32)
            transition_offset = np.full((4, N), float(N), dtype=np.float32)
        to_t = lambda a: torch.tensor(a, device=self.device)
        self._ref_body_pos = to_t(st[:N + 1, MPC_X_BASE_POS].astype(np.float32))
        self._ref_body_quat = to_t(body_quat)
        self._ref_joint_pos = to_t(st[:N + 1, MPC_X_Q_JOINTS].astype(np.float32))
        self._ref_joint_vel = to_t(ref.joint_vel_traj[:N].astype(np.float32))
        self._ref_body_vel = to_t(st[:N + 1, MPC_X_BASE_VEL].astype(np.float32))
        self._ref_body_ang_vel = to_t(st[:N + 1, MPC_X_BASE_ANG].astype(np.float32))
        self._ref_ff_torques = to_t(ref._ff_torques[:N].astype(np.float32))
        self._ref_contact_seq = to_t(contact_seq)
        self._ref_near_transition = to_t(near_transition)
        self._ref_transition_offset = to_t(transition_offset)  # (4, N)

    def _setup_contact_indices(self):
        self._foot_body_ids, _ = self._contact_sensor.find_bodies("FL_foot|FR_foot|RL_foot|RR_foot")
        all_ids = set(range(self._contact_sensor.num_bodies))
        foot_set = set(self._foot_body_ids.tolist() if isinstance(self._foot_body_ids, torch.Tensor) else self._foot_body_ids)
        self._non_foot_body_ids = torch.tensor(sorted(all_ids - foot_set), dtype=torch.long, device=self.device)
        self._foot_body_ids_t = torch.tensor(sorted(foot_set), dtype=torch.long, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        """OPT-Mimic Eq. 14: PD + feedforward + residual."""
        action_scaled = self.actions * ACTION_LIMIT
        phase = self._phase.clamp(0, self._max_phase - 1)
        actual_jpos = self._to_mpc_order(self._robot.data.joint_pos)
        actual_jvel = self._to_mpc_order(self._robot.data.joint_vel)
        target = self._ref_joint_pos[phase] + action_scaled + self._joint_offset
        torque = (KP * (target - actual_jpos)
                  + KD * (self._ref_joint_vel[phase] - actual_jvel)
                  + self._ref_ff_torques[phase])
        torque = torque.clamp(-self._torque_limits, self._torque_limits) * self._torque_scale.unsqueeze(-1)
        self._last_torque = torque.clone()
        self._robot.set_joint_effort_target(self._to_isaac_order(torque))

    def _get_observations(self) -> dict:
        """OPT-Mimic §III-C.1: proprioception-only (33 dims).

        quat(4) + joint_pos(12) + base_ang_vel(3) + joint_vel(12) + phase(2) = 33.
        Base position and linear velocity are NOT included — they require a state
        estimator on the real robot, which the paper deliberately avoids.
        """
        phase = self._phase.clamp(0, self._max_phase - 1)
        joint_pos = self._to_mpc_order(self._robot.data.joint_pos) + self._joint_offset
        angle = 2.0 * torch.pi * phase.float() / float(self._max_phase)
        obs = torch.cat([
            self._robot.data.root_quat_w, joint_pos,
            self._robot.data.root_ang_vel_w,
            self._to_mpc_order(self._robot.data.joint_vel),
            torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1),
        ], dim=-1)
        return {"policy": torch.nan_to_num(obs, nan=0.0)}

    def _get_rewards(self) -> torch.Tensor:
        total, components = compute_rewards(self._tracking_errors)
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update(components)
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._phase += 1
        self._first_step[:] = False
        phase = self._phase.clamp(0, self._max_phase - 1)
        action_scaled = self.actions * ACTION_LIMIT
        self._tracking_errors = compute_tracking_errors(
            self._ref_body_pos[phase], self._ref_body_quat[phase], self._ref_joint_pos[phase],
            self._robot.data.root_pos_w - self._env_origins, self._robot.data.root_quat_w,
            self._to_mpc_order(self._robot.data.joint_pos),
            action_scaled, self._prev_action, self._last_torque,
        )
        self._prev_action = action_scaled.clone()

        # ---- Phase-0.2 instrumentation: per-phase error accumulation ----
        # Index by post-step phase (the index whose ref the actual state was
        # measured against). first_step envs contribute 0 to action_rate (it
        # references a stale prev_action) — exclude their action_rate sample.
        ph_idx = self._phase.clamp(0, self._max_phase).long()
        ones_b = torch.ones_like(ph_idx, dtype=self._phase_err_count.dtype)
        self._phase_err_count.scatter_add_(0, ph_idx, ones_b)
        for m_idx, mname in enumerate(self._phase_err_metrics):
            vals = self._tracking_errors[mname].to(torch.float64)
            if mname == "action_rate":
                vals = torch.where(self._first_step, torch.zeros_like(vals), vals)
            self._phase_err_sum[:, m_idx].scatter_add_(0, ph_idx, vals)

        forces = self._contact_sensor.data.net_forces_w_history

        thresh_masks = tracking_termination_breakdown(self._tracking_errors, self._first_step)
        body_mask = check_body_contact(forces, self._non_foot_body_ids)
        cm_info = contact_mismatch_diagnostics(
            forces, self._foot_body_ids_t,
            self._ref_contact_seq[:, phase].T, self._ref_near_transition[:, phase].T,
        )
        contact_mask = cm_info["any_mismatch"]
        nan_mask = torch.any(torch.isnan(self._robot.data.root_pos_w), dim=-1)

        # ---- Phase-0.3 instrumentation: per-foot slip stats ----
        mismatch_pf = cm_info["mismatch_per_foot"]  # (n_envs, 4)
        force_pf = cm_info["actual_force_per_foot"]  # (n_envs, 4)
        if mismatch_pf.any():
            phase_clamped = self._phase.clamp(0, self._max_phase - 1).long()
            for foot_idx in range(4):
                m = mismatch_pf[:, foot_idx]
                if m.any():
                    self._slip_count_per_foot[foot_idx] += int(m.sum().item())
                    self._slip_force_sum_per_foot[foot_idx] += float(force_pf[:, foot_idx][m].sum().item())
                    offsets = self._ref_transition_offset[foot_idx, phase_clamped[m]].to(torch.float64)
                    self._slip_offset_abs_sum_per_foot[foot_idx] += float(offsets.sum().item())
        thresh_any = (
            thresh_masks["thresh_pos"] | thresh_masks["thresh_ori"]
            | thresh_masks["thresh_joint"] | thresh_masks["thresh_rate"]
            | thresh_masks["thresh_torque"]
        )
        terminated = thresh_any | body_mask | contact_mask | nan_mask
        truncated = self._phase >= self._max_phase

        # ---- Phase-0.1 instrumentation: per-cause × per-phase histogram ----
        cause_masks = {
            "thresh_pos": thresh_masks["thresh_pos"],
            "thresh_ori": thresh_masks["thresh_ori"],
            "thresh_joint": thresh_masks["thresh_joint"],
            "thresh_rate": thresh_masks["thresh_rate"],
            "thresh_torque": thresh_masks["thresh_torque"],
            "body": body_mask,
            "contact": contact_mask,
            "nan": nan_mask,
            "trunc": truncated,
        }
        # Phase-0.4: expose latest masks + actual force-per-foot to evaluate.py
        # for per-step termination-cause attribution and contact logging.
        self._last_cause_masks = cause_masks
        self._last_actual_force_per_foot = cm_info["actual_force_per_foot"]
        phase_long = self._phase.clamp(0, self._max_phase).long()
        for idx, name in enumerate(TERM_CAUSE_NAMES):
            m = cause_masks[name]
            if m.any():
                phs = phase_long[m]
                ones = torch.ones_like(phs, dtype=self._term_phase_hist.dtype)
                self._term_phase_hist[idx].scatter_add_(0, phs, ones)
                self._term_counts[name] += int(m.sum().item())

        if "log" not in self.extras:
            self.extras["log"] = {}
        # Backward-compat aggregates expected by callbacks.TrainingLogger.
        self.extras["log"]["term_thresh"] = float(self._term_counts["thresh_pos"]
                                                  + self._term_counts["thresh_ori"]
                                                  + self._term_counts["thresh_joint"]
                                                  + self._term_counts["thresh_rate"]
                                                  + self._term_counts["thresh_torque"])
        self.extras["log"]["term_body"] = float(self._term_counts["body"])
        self.extras["log"]["term_contact"] = float(self._term_counts["contact"])
        self.extras["log"]["term_nan"] = float(self._term_counts["nan"])
        self.extras["log"]["term_trunc"] = float(self._term_counts["trunc"])
        return terminated, truncated

    def flush_term_diagnostics(self) -> tuple[dict[str, int], torch.Tensor]:
        """Return accumulated (counts dict, per-cause × per-phase histogram)
        and reset both. Called once per PPO update by train.py."""
        counts = dict(self._term_counts)
        hist = self._term_phase_hist.clone()
        self._term_phase_hist.zero_()
        for k in self._term_counts:
            self._term_counts[k] = 0
        return counts, hist

    def flush_phase_errors(self) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
        """Return (sum tensor of shape (max_phase+1, n_metrics), count tensor
        of shape (max_phase+1,), metric names) and reset both. Called once per
        PPO update by train.py."""
        sums = self._phase_err_sum.clone()
        counts = self._phase_err_count.clone()
        self._phase_err_sum.zero_()
        self._phase_err_count.zero_()
        return sums, counts, self._phase_err_metrics

    def flush_slip_diagnostics(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-foot (count, force_sum, offset_abs_sum) and reset.
        Caller divides sums by count for means. Called once per PPO update."""
        counts = self._slip_count_per_foot.clone()
        force_sum = self._slip_force_sum_per_foot.clone()
        offset_sum = self._slip_offset_abs_sum_per_foot.clone()
        self._slip_count_per_foot.zero_()
        self._slip_force_sum_per_foot.zero_()
        self._slip_offset_abs_sum_per_foot.zero_()
        return counts, force_sum, offset_sum

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        n = len(env_ids)
        self._joint_offset[env_ids] = torch.randn(n, 12, device=self.device) * 0.02
        self._torque_scale[env_ids] = (1.0 + torch.randn(n, device=self.device) * 0.1).clamp(0.5, 1.5)
        # Friction & restitution DR (OPT-Mimic paper Table I)
        mat = self._robot.root_physx_view.get_material_properties()
        env_ids_cpu = env_ids.cpu() if env_ids.device.type != "cpu" else env_ids
        n_shapes = mat.shape[1]
        friction = (torch.randn(n) * 0.25 + 0.8).clamp(0.1, 1.0)
        restitution = (torch.randn(n) * 0.25).clamp(0.0, 1.0)
        mat[env_ids_cpu, :, 0] = friction.unsqueeze(-1).expand(-1, n_shapes)
        mat[env_ids_cpu, :, 1] = friction.unsqueeze(-1).expand(-1, n_shapes)
        mat[env_ids_cpu, :, 2] = restitution.unsqueeze(-1).expand(-1, n_shapes)
        self._robot.root_physx_view.set_material_properties(mat, env_ids_cpu)
        start = torch.randint(0, self._max_phase, (n,), device=self.device, dtype=torch.int32)
        self._phase[env_ids] = start
        ph = start.clamp(0, self._max_phase - 1).long()
        root_pos = self._ref_body_pos[ph].clone() + self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(torch.cat([root_pos, self._ref_body_quat[ph].clone()], dim=-1), env_ids)
        self._robot.write_root_velocity_to_sim(
            torch.cat([self._ref_body_vel[ph].clone(), self._ref_body_ang_vel[ph].clone()], dim=-1), env_ids,
        )
        self._robot.write_joint_state_to_sim(
            self._to_isaac_order(self._ref_joint_pos[ph].clone()),
            self._to_isaac_order(self._ref_joint_vel[ph].clone()), None, env_ids,
        )
        self._prev_action[env_ids] = 0.0
        self._last_torque[env_ids] = 0.0
        self._first_step[env_ids] = True
