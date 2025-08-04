# Description: This script orchestrates a two-stage, decoupled approach for
# generating a hopping trajectory using the new `HoppingMPC` class for
# trajectory optimization and a helper function for inverse dynamics.

import numpy as np
import config

# Import the new HoppingMPC class
from examples.mpc import HoppingMPC
from examples.model import KinoDynamic_Model

# Import the inverse dynamics helper
from examples.util import compute_joint_torques

def main():
    # Define a hopping gait by its duration and discretization step
    T = 2.0 # total duration in seconds
    dt = 0.1 # time step in seconds
    horizon = int(T / dt)

    stance_duration = 0.4 # seconds
    flight_duration = 0.5 # seconds

    steps_per_phase = int(stance_duration / dt)

    # Create a contact sequence for a hop based on T and dt
    contact_sequence = np.ones((4, horizon))
    contact_sequence[:, steps_per_phase:steps_per_phase + int(flight_duration / dt)] = 0.0

    # --- STAGE 1: Trajectory Optimization using HoppingMPC ---
    print("--- Stage 1: Solving Kinodynamic Trajectory Optimization ---")

    # Initialize the new HoppingMPC class with T and dt
    mpc = HoppingMPC(T=T, dt=dt, config=config)

    # Set up the initial state and reference for the hopping motion
    initial_state = {
        'position': np.array([0.0, 0.0, 0.5]),
        'linear_velocity': np.array([0.0, 0.0, 0.0]),
        'orientation': np.array([0.0, 0.0, 0.0]),
        'angular_velocity': np.array([0.0, 0.0, 0.0]),
        'joint_FL': np.zeros(3),
        'joint_FR': np.zeros(3),
        'joint_RL': np.zeros(3),
        'joint_RR': np.zeros(3),
    }

    reference = {
        'ref_position': np.array([0.5, 0.0, 0.5]),
        'ref_linear_velocity': np.array([0.5, 0.0, 0.0]),
        'ref_orientation': np.zeros(3),
        'ref_angular_velocity': np.zeros(3),
        'ref_joints': np.zeros(12),
    }

    state_traj, grf_traj, joint_vel_traj, status = mpc.solve_trajectory(
        initial_state,
        reference,
        contact_sequence
    )

    if status != 0:
        print(f"Optimization failed with status: {status}")
        return

    print("Optimization successful. Extracted trajectory of states and GRFs.")
    np.save("results/state_traj.npy", state_traj)
    np.save("results/grf_traj.npy", grf_traj)
    np.save("results/contact_sequence.npy", contact_sequence)

    state_traj = np.load("results/state_traj.npy")
    grf_traj = np.load("results/grf_traj.npy")
    contact_sequence = np.load("results/contact_sequence.npy")

    # --- STAGE 2: Inverse Dynamics to find Joint Torques ---
    print("\n--- Stage 2: Computing Joint Torques via Inverse Dynamics ---")

    # We create an instance of your original model to access its internal
    # `kindyn` object.
    kinodynamic_model = KinoDynamic_Model(config)

    print(f"state_traj: {state_traj.shape}, grf_traj: {grf_traj.shape}, contact_sequence: {contact_sequence.shape}, dt: {dt}")
    joint_torques_traj = compute_joint_torques(
        kinodynamic_model,
        state_traj,
        grf_traj,
        contact_sequence,
        dt
    )
    
    np.save("results/joint_torques_traj.npy", joint_torques_traj)

if __name__ == "__main__":
    main()
