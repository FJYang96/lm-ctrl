#!/usr/bin/env python3
"""
Compare results between standard constraints and complementary constraints.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_results(suffix=""):
    """Load trajectory results with given suffix."""
    suffix_str = f"_{suffix}" if suffix else ""
    
    try:
        state_traj = np.load(f"results/state_traj{suffix_str}.npy")
        grf_traj = np.load(f"results/grf_traj{suffix_str}.npy") 
        joint_vel_traj = np.load(f"results/joint_vel_traj{suffix_str}.npy")
        contact_sequence = np.load("results/contact_sequence.npy")
        
        return state_traj, grf_traj, joint_vel_traj, contact_sequence
    except FileNotFoundError as e:
        print(f"Could not load results with suffix '{suffix_str}': {e}")
        return None, None, None, None

def plot_comparison():
    """Compare the trajectories from both methods."""
    # Load results
    state_std, grf_std, vel_std, contact = load_results("opti")
    state_comp, grf_comp, vel_comp, _ = load_results("opti_comp")
    
    if state_std is None or state_comp is None:
        print("Error: Could not load trajectory data. Make sure to run both versions first.")
        return
    
    time = np.arange(len(state_std)) * 0.1  # Assuming 0.1s time step
    time_input = np.arange(len(grf_std)) * 0.1
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Comparison: Standard vs Complementary Constraints', fontsize=16)
    
    # Plot 1: Base height
    axes[0, 0].plot(time, state_std[:, 2], 'b-', label='Standard', linewidth=2)
    axes[0, 0].plot(time, state_comp[:, 2], 'r--', label='Complementary', linewidth=2)
    axes[0, 0].set_ylabel('Base Height (m)')
    axes[0, 0].set_title('Center of Mass Height')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Contact forces (sum of all feet)
    total_force_std = np.sum(grf_std.reshape(-1, 4, 3)[:, :, 2], axis=1)  # Sum z-forces
    total_force_comp = np.sum(grf_comp.reshape(-1, 4, 3)[:, :, 2], axis=1)
    
    axes[0, 1].plot(time_input, total_force_std, 'b-', label='Standard', linewidth=2)
    axes[0, 1].plot(time_input, total_force_comp, 'r--', label='Complementary', linewidth=2)
    axes[0, 1].set_ylabel('Total Normal Force (N)')
    axes[0, 1].set_title('Total Ground Reaction Forces')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Contact schedule vs actual forces for one foot (FL)
    fl_force_std = grf_std[:, 2]  # FL foot z-force
    fl_force_comp = grf_comp[:, 2]
    contact_fl = contact[0, :len(fl_force_std)]  # FL contact schedule
    
    axes[0, 2].plot(time_input, contact_fl * 100, 'k-', label='Contact Schedule', linewidth=2)
    axes[0, 2].plot(time_input, fl_force_std, 'b-', label='Standard Force', alpha=0.7)
    axes[0, 2].plot(time_input, fl_force_comp, 'r--', label='Complementary Force', alpha=0.7)
    axes[0, 2].set_ylabel('Force (N) / Contact*100')
    axes[0, 2].set_title('FL Foot: Contact vs Forces')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Plot 4: Base velocity
    axes[1, 0].plot(time, state_std[:, 5], 'b-', label='Standard', linewidth=2) # z-velocity
    axes[1, 0].plot(time, state_comp[:, 5], 'r--', label='Complementary', linewidth=2)
    axes[1, 0].set_ylabel('Z Velocity (m/s)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_title('Vertical Velocity')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 5: Joint velocities (average magnitude)
    joint_vel_mag_std = np.linalg.norm(vel_std, axis=1)
    joint_vel_mag_comp = np.linalg.norm(vel_comp, axis=1)
    
    axes[1, 1].plot(time_input, joint_vel_mag_std, 'b-', label='Standard', linewidth=2)
    axes[1, 1].plot(time_input, joint_vel_mag_comp, 'r--', label='Complementary', linewidth=2)
    axes[1, 1].set_ylabel('Joint Velocity Magnitude (rad/s)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_title('Joint Velocity Magnitude')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Plot 6: Force magnitude comparison
    force_mag_std = np.linalg.norm(grf_std, axis=1)
    force_mag_comp = np.linalg.norm(grf_comp, axis=1)
    
    axes[1, 2].plot(time_input, force_mag_std, 'b-', label='Standard', linewidth=2)
    axes[1, 2].plot(time_input, force_mag_comp, 'r--', label='Complementary', linewidth=2)
    axes[1, 2].set_ylabel('Force Magnitude (N)')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_title('Total Force Magnitude')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('results/comparison_plot.png', dpi=300, bbox_inches='tight')
    print("Comparison plot saved to results/comparison_plot.png")
    plt.show()

def print_summary_stats():
    """Print summary statistics for both methods."""
    # Load results
    state_std, grf_std, vel_std, contact = load_results("opti")
    state_comp, grf_comp, vel_comp, _ = load_results("opti_comp")
    
    if state_std is None or state_comp is None:
        print("Error: Could not load trajectory data.")
        return
    
    print("\n" + "="*60)
    print("TRAJECTORY COMPARISON SUMMARY")
    print("="*60)
    
    # Maximum jump height
    max_height_std = np.max(state_std[:, 2])
    max_height_comp = np.max(state_comp[:, 2])
    
    print(f"Maximum Jump Height:")
    print(f"  Standard:        {max_height_std:.3f} m")
    print(f"  Complementary:   {max_height_comp:.3f} m")
    print(f"  Difference:      {max_height_comp - max_height_std:.3f} m")
    
    # Peak forces
    max_force_std = np.max(np.abs(grf_std))
    max_force_comp = np.max(np.abs(grf_comp))
    
    print(f"\nPeak Ground Reaction Force:")
    print(f"  Standard:        {max_force_std:.1f} N")
    print(f"  Complementary:   {max_force_comp:.1f} N")
    print(f"  Difference:      {max_force_comp - max_force_std:.1f} N")
    
    # Average joint velocities
    avg_vel_std = np.mean(np.abs(vel_std))
    avg_vel_comp = np.mean(np.abs(vel_comp))
    
    print(f"\nAverage Joint Velocity Magnitude:")
    print(f"  Standard:        {avg_vel_std:.3f} rad/s")
    print(f"  Complementary:   {avg_vel_comp:.3f} rad/s")
    print(f"  Difference:      {avg_vel_comp - avg_vel_std:.3f} rad/s")
    
    # Force smoothness (variance)
    force_var_std = np.var(grf_std)
    force_var_comp = np.var(grf_comp)
    
    print(f"\nForce Variance (smoothness indicator):")
    print(f"  Standard:        {force_var_std:.1f}")
    print(f"  Complementary:   {force_var_comp:.1f}")
    print(f"  Ratio:           {force_var_comp / force_var_std:.2f}")
    
    # Contact consistency check (forces near zero when not in contact)
    print(f"\nContact Consistency Analysis:")
    
    # Check all feet during flight phase (assuming middle section is flight)
    flight_start = len(grf_std) // 3
    flight_end = 2 * len(grf_std) // 3
    
    flight_forces_std = grf_std[flight_start:flight_end]
    flight_forces_comp = grf_comp[flight_start:flight_end]
    
    # Forces should be near zero during flight
    avg_flight_force_std = np.mean(np.abs(flight_forces_std))
    avg_flight_force_comp = np.mean(np.abs(flight_forces_comp))
    
    print(f"  Avg force during flight (should be ~0):")
    print(f"    Standard:      {avg_flight_force_std:.3f} N")
    print(f"    Complementary: {avg_flight_force_comp:.3f} N")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Compare MPC results')
    parser.add_argument('--plot', action='store_true', 
                        help='Generate comparison plots')
    parser.add_argument('--stats', action='store_true',
                        help='Print summary statistics')
    
    args = parser.parse_args()
    
    if not args.plot and not args.stats:
        # Default: show both
        print_summary_stats()
        plot_comparison()
    else:
        if args.stats:
            print_summary_stats()
        if args.plot:
            plot_comparison()

if __name__ == "__main__":
    main()
