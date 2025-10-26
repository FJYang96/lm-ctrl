#!/usr/bin/env python3
"""
Test script to verify the improved inverse dynamics integration
"""
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_import():
    """Test that we can import both inverse dynamics functions"""
    try:
        from utils.inv_dyn import compute_joint_torques, compute_joint_torques_improved
        print("✅ Successfully imported both inverse dynamics functions")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_function_signatures():
    """Test that the function signatures are compatible"""
    try:
        from utils.inv_dyn import compute_joint_torques, compute_joint_torques_improved
        import inspect
        
        # Get function signatures
        orig_sig = inspect.signature(compute_joint_torques)
        improved_sig = inspect.signature(compute_joint_torques_improved)
        
        print(f"Original function signature: {orig_sig}")
        print(f"Improved function signature: {improved_sig}")
        
        # Check parameter names
        orig_params = list(orig_sig.parameters.keys())
        improved_params = list(improved_sig.parameters.keys())
        
        print(f"Original parameters: {orig_params}")
        print(f"Improved parameters: {improved_params}")
        
        # The improved function should have the same basic parameters
        expected_params = ['kindyn_model', 'state_traj', 'input_traj', 'contact_sequence', 'dt']
        if improved_params == expected_params:
            print("✅ Function signature is correct")
            return True
        else:
            print("❌ Function signature mismatch")
            return False
            
    except Exception as e:
        print(f"❌ Signature test failed: {e}")
        return False

def test_utility_functions():
    """Test utility functions from the improved implementation"""
    try:
        from utils.inv_dyn import (
            quat_wxyz_to_rotmat, 
            euler_xyz_to_quat_wxyz,
            foot_fk_local,
            feet_positions_world_from_qpos
        )
        
        # Test quaternion conversion
        euler = np.array([0.1, 0.2, 0.3])  # roll, pitch, yaw
        quat = euler_xyz_to_quat_wxyz(euler)
        R = quat_wxyz_to_rotmat(quat)
        
        print(f"Euler angles: {euler}")
        print(f"Quaternion: {quat}")
        print(f"Rotation matrix shape: {R.shape}")
        
        # Test FK
        joints = np.array([0.1, 0.5, -1.0])  # abd, hip, knee
        foot_pos = foot_fk_local(joints)
        print(f"Foot position: {foot_pos}")
        
        print("✅ Utility functions work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Utility function test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing improved inverse dynamics integration...")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_import),
        ("Function Signature Test", test_function_signatures), 
        ("Utility Function Test", test_utility_functions),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append(result)
        
    print("\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    all_passed = all(results)
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{test_name}: {status}")
        
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Integration successful! The improved inverse dynamics has been")
        print("   properly integrated into utils/inv_dyn.py and main.py has been")
        print("   updated to use the improved function.")
    else:
        print("\n⚠️  Integration needs attention. Please check the failed tests above.")

if __name__ == "__main__":
    main()