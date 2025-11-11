import imageio
import numpy as np
from gym_quadruped.quadruped_env import QuadrupedEnv

import config
from mpc.dynamics.model import KinoDynamic_Model
from mpc.mpc_opti import QuadrupedMPCOpti
from utils import conversion
from utils.inv_dyn import compute_joint_torques
from utils.logging import color_print
from utils.simulation import (
    create_reference_trajectory,
    save_trajectory_results,
    simulate_trajectory,
)
from utils.visualization import (
    plot_trajectory_comparison,
    render_and_save_planned_trajectory,
)


def main() -> None:
    # ========================================================
    # Stage 0: Setup
    # ========================================================
    color_print("orange", "Stage 0: Setup")
    kinodynamic_model = KinoDynamic_Model(config)
    mpc = QuadrupedMPCOpti(model=kinodynamic_model, config=config, build=True)
    env = QuadrupedEnv(
        robot=config.robot,
        scene="flat",
        ground_friction_coeff=config.experiment.mu_ground,
        state_obs_names=QuadrupedEnv._DEFAULT_OBS + ("contact_forces:base",),
        sim_dt=config.experiment.sim_dt,
    )
    suffix = ""

    # ========================================================
    # Stage 1: Trajectory Optimization
    # ========================================================
    color_print("orange", "Stage 1: Trajectory Optimization")
    initial_state, _ = conversion.sim_to_mpc(
        config.experiment.initial_qpos, config.experiment.initial_qvel
    )

    ref = create_reference_trajectory(config.experiment.initial_qpos)

    state_traj, grf_traj, joint_vel_traj, status = mpc.solve_trajectory(
        initial_state, ref, config.mpc_config.contact_sequence
    )
    input_traj = np.concatenate([joint_vel_traj, grf_traj], axis=1)

    if status != 0:
        color_print("red", f"Optimization failed with status: {status}")
    else:
        color_print("green", "Optimization successful.")

    # Save trajectory results
    save_trajectory_results(
        state_traj, joint_vel_traj, grf_traj, config.mpc_config.contact_sequence, suffix
    )

    # Render planned trajectory if rendering is enabled
    planned_traj_images = None
    if config.experiment.render:
        planned_traj_images = render_and_save_planned_trajectory(
            state_traj, input_traj, env, suffix
        )

    # ========================================================
    # Stage 2: Inverse Dynamics + Simulation
    # ========================================================
    color_print("orange", "Stage 2: Inverse Dynamics + Simulation")
    joint_torques_traj = compute_joint_torques(
        kinodynamic_model,
        state_traj,
        grf_traj,
        config.mpc_config.contact_sequence,
        config.mpc_config.mpc_dt,
    )
    np.save(f"results/joint_torques_traj{suffix}.npy", joint_torques_traj)

    # print joint torque
    # print(f"\n{'='*80}")
    # print(f"JOINT TORQUES ANALYSIS")
    # print(f"{'='*80}")
    # print(f"Torque trajectory shape: {joint_torques_traj.shape}")

    # joint_names = ["FL_hip", "FL_thigh", "FL_calf",
    #                "FR_hip", "FR_thigh", "FR_calf",
    #                "RL_hip", "RL_thigh", "RL_calf",
    #                "RR_hip", "RR_thigh", "RR_calf"]

    # print(f"\nPer-joint torque statistics (Nm):")
    # print(f"{'Joint':<12} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'|Max|':>10}")
    # print(f"{'-'*72}")
    # for j in range(12):
    #     tau_j = joint_torques_traj[:, j]
    #     tau_min = np.min(tau_j)
    #     tau_max = np.max(tau_j)
    #     tau_mean = np.mean(tau_j)
    #     tau_std = np.std(tau_j)
    #     tau_abs_max = np.max(np.abs(tau_j))
    #     print(f"{joint_names[j]:<12} {tau_min:>10.3f} {tau_max:>10.3f} {tau_mean:>10.3f} {tau_std:>10.3f} {tau_abs_max:>10.3f}")

    # print(f"\nOverall statistics:")
    # print(f"  Global max torque: {np.max(joint_torques_traj):.3f} Nm")
    # print(f"  Global min torque: {np.min(joint_torques_traj):.3f} Nm")
    # print(f"  Global |max| torque: {np.max(np.abs(joint_torques_traj)):.3f} Nm")
    # print(f"  Mean |torque|: {np.mean(np.abs(joint_torques_traj)):.3f} Nm")

    # torque_limits = config.robot_data.joint_torque_limits
    # violations_per_joint = np.sum(
    #     np.abs(joint_torques_traj) > torque_limits[np.newaxis, :], axis=0
    # )
    # total_violations = np.sum(violations_per_joint)

    # if total_violations > 0:
    #     color_print("red", f"\n Torque limit violations detected!")
    #     print(f"  Total violations: {total_violations} (out of {joint_torques_traj.size} values)")
    #     print(f"\n  Per-joint violations:")
    #     for j in range(12):
    #         if violations_per_joint[j] > 0:
    #             print(f"    {joint_names[j]:<12}: {violations_per_joint[j]:>3} violations (limit: ±{torque_limits[j]:.2f} Nm)")
    # else:
    #     color_print("green", f"\n All torques within limits!")

    # print(f"{'='*80}\n")

    qpos_traj, qvel_traj, grf_traj, images = simulate_trajectory(
        env, joint_torques_traj, planned_traj_images
    )

    # Plot comparison between planned and simulated trajectories
    plot_trajectory_comparison(
        state_traj,
        input_traj,
        qpos_traj,
        qvel_traj,
        grf_traj,
        quantities=config.plot_quantities,
        mpc_dt=config.mpc_config.mpc_dt,
        sim_dt=config.experiment.sim_dt,
        save_path="results/trajectory_comparison.png",
        show_plot=False,
    )

    # Save simulation video
    if config.experiment.render:
        fps = 1 / config.experiment.sim_dt
        imageio.mimsave(f"results/trajectory{suffix}.mp4", images, fps=fps)


if __name__ == "__main__":
    main()
