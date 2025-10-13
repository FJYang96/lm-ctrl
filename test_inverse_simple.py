#!/usr/bin/env python3
"""
Simple test for the fixed inverse dynamics implementation
"""
import numpy as np
from debug_id import *  # Import the working original test setup

# Import the fixed inverse dynamics
from utils.inv_dyn import compute_joint_torques

print("Testing fixed inverse dynamics vs original...")

# Use the same setup as debug_id.py to compare
kinodynamic_model = KinoDynamic_Model(config)
kinodynamic_model.export_robot_model()

# Test the fixed version
try:
    computed_torques_fixed = compute_joint_torques(
        kinodynamic_model,
        mpc_state_traj,
        mpc_input_traj[:-1],
        contact_state_traj[:-1].T,  # Note: transposed contact sequence
        sim_dt,
    )
    print("✅ Fixed inverse dynamics completed successfully!")
    
    # Compute validation metrics
    skip = 20  # Skip initial transient
    rmse_per_joint = []
    for i in range(12):
        rmse = np.sqrt(np.mean((computed_torques_fixed[skip:, i] - torque_traj[skip:, i])**2))
        rmse_per_joint.append(rmse)
        print(f"Joint {i+1:2d}: RMSE = {rmse:.4f} Nm")
    
    overall_rmse = np.sqrt(np.mean((computed_torques_fixed[skip:] - torque_traj[skip:])**2))
    max_error = np.max(np.abs(computed_torques_fixed[skip:] - torque_traj[skip:]))
    
    print(f"\nOverall RMSE: {overall_rmse:.4f} Nm")
    print(f"Max absolute error: {max_error:.4f} Nm")
    
    # Success criteria
    success = overall_rmse < 2.0 and max_error < 10.0
    print(f"\nValidation {'PASSED' if success else 'FAILED'}")
    if success:
        print("✅ Inverse dynamics model is working correctly!")
    else:
        print("❌ Inverse dynamics model needs further debugging")
        
except Exception as e:
    print(f"❌ Error in fixed inverse dynamics: {e}")
    import traceback
    traceback.print_exc()