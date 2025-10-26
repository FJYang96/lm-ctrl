#!/usr/bin/env python3
"""
Quick script to run main.py with different inverse dynamics versions
"""
import subprocess
import os

def modify_main_for_original():
    """Temporarily modify main.py to use original inverse dynamics"""
    with open('main.py', 'r') as f:
        content = f.read()
    
    # Replace the improved call with original call
    modified = content.replace(
        """# Use improved inverse dynamics computation with better numerical stability
    # Create proper input trajectory that combines joint velocities and GRFs
    input_traj = np.concatenate([joint_vel_traj, grf_traj], axis=1)
    joint_torques_traj = compute_joint_torques_improved(
        kinodynamic_model, state_traj, input_traj, config.contact_sequence, config.mpc_dt
    )""",
        """# Use ORIGINAL inverse dynamics computation
    joint_torques_traj = compute_joint_torques(
        kinodynamic_model, state_traj, grf_traj, config.contact_sequence, config.mpc_dt
    )"""
    )
    
    with open('main.py', 'w') as f:
        f.write(modified)
    
    return content  # Return original content for restoration

def restore_main(original_content):
    """Restore main.py to its original state"""
    with open('main.py', 'w') as f:
        f.write(original_content)

def run_with_version(version_name, use_original=False):
    """Run main.py with specified version"""
    print(f"\n{'='*60}")
    print(f"RUNNING WITH {version_name.upper()} INVERSE DYNAMICS")
    print(f"{'='*60}")
    
    if use_original:
        print("Temporarily switching to original implementation...")
        original_content = modify_main_for_original()
    
    try:
        result = subprocess.run(['python', 'main.py', '--solver', 'opti'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {version_name} completed successfully!")
            # Look for specific output patterns
            if "Inverse dynamics computation completed!" in result.stdout:
                print("✅ Inverse dynamics computation detected")
            if "Building state representations with standardized indexing" in result.stdout:
                print("✅ Improved features detected (standardized indexing)")
            if "Computing accelerations using improved numerical methods" in result.stdout:
                print("✅ Improved features detected (numerical methods)")
        else:
            print(f"❌ {version_name} failed with return code: {result.returncode}")
            print("STDERR:", result.stderr[-500:] if result.stderr else "None")
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {version_name} timed out after 5 minutes")
    except Exception as e:
        print(f"❌ Error running {version_name}: {e}")
    
    finally:
        if use_original:
            print("Restoring to improved implementation...")
            restore_main(original_content)

def main():
    print("🚀 Starting comparison of inverse dynamics implementations...")
    
    # Check if we're in the right directory
    if not os.path.exists('main.py'):
        print("❌ main.py not found. Please run this script from the lm-ctrl directory.")
        return
    
    # Run with improved version (current state)
    run_with_version("IMPROVED", use_original=False)
    
    # Run with original version
    run_with_version("ORIGINAL", use_original=True)
    
    print(f"\n🎉 Comparison complete!")
    print("\nTo run individually:")
    print("  Improved (current): python main.py --solver opti")
    print("  Original: python test_both_versions.py --test original")
    print("  Both: python test_both_versions.py --test both")

if __name__ == "__main__":
    main()