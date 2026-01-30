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

    # Compute joint torques using updated inverse dynamics
    color_print("green", "Computing joint torques using updated inverse dynamics")
    joint_torques_traj = compute_joint_torques(
        kinodynamic_model,
        state_traj,
        input_traj,
        config.mpc_config.contact_sequence,
        config.mpc_config.mpc_dt,
    )
    np.save(f"results/joint_torques_traj{suffix}.npy", joint_torques_traj)

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
