#!/usr/bin/env python3
"""
Debug script to investigate why original and improved inverse dynamics
show opposite signs (mirror images)
"""

import matplotlib.pyplot as plt
import numpy as np


def analyze_sign_patterns():
    """Analyze the sign patterns between implementations"""

    # Load the comparison data
    try:
        torques_orig = np.load("results/torques_original_comparison.npy")
        torques_improved = np.load("results/torques_improved_comparison.npy")
    except FileNotFoundError:
        print("❌ Comparison data not found. Run generate_comparison_plots.py first.")
        return

    print("🔍 ANALYZING SIGN PATTERNS")
    print("=" * 50)

    # Overall sign correlation analysis
    print(f"Original shape: {torques_orig.shape}")
    print(f"Improved shape: {torques_improved.shape}")

    # Check sign patterns for each joint
    print("\n📊 Sign Analysis per Joint:")
    for joint in range(12):
        orig_joint = torques_orig[:, joint]
        improved_joint = torques_improved[:, joint]

        # Count sign agreements/disagreements
        same_sign = np.sum(np.sign(orig_joint) == np.sign(improved_joint))
        opposite_sign = np.sum(np.sign(orig_joint) == -np.sign(improved_joint))
        total = len(orig_joint)

        # Correlation coefficient
        correlation = np.corrcoef(orig_joint, improved_joint)[0, 1]

        print(
            f"Joint {joint+1:2d}: Same sign: {same_sign:2d}/{total} ({same_sign/total*100:.1f}%), "
            f"Opposite: {opposite_sign:2d}/{total} ({opposite_sign/total*100:.1f}%), "
            f"Correlation: {correlation:.3f}"
        )

    # Overall statistics
    same_sign_total = np.sum(np.sign(torques_orig) == np.sign(torques_improved))
    opposite_sign_total = np.sum(np.sign(torques_orig) == -np.sign(torques_improved))
    total_elements = torques_orig.size

    print("\n🎯 Overall Analysis:")
    print(
        f"Same sign: {same_sign_total}/{total_elements} ({same_sign_total/total_elements*100:.1f}%)"
    )
    print(
        f"Opposite sign: {opposite_sign_total}/{total_elements} ({opposite_sign_total/total_elements*100:.1f}%)"
    )

    overall_correlation = np.corrcoef(
        torques_orig.flatten(), torques_improved.flatten()
    )[0, 1]
    print(f"Overall correlation: {overall_correlation:.3f}")

    # Check if it's a simple sign flip
    sign_flipped_correlation = np.corrcoef(
        torques_orig.flatten(), -torques_improved.flatten()
    )[0, 1]
    print(f"Correlation with sign flip: {sign_flipped_correlation:.3f}")

    return torques_orig, torques_improved


def investigate_potential_causes():
    """Investigate potential causes of sign differences"""

    print("\n🔍 POTENTIAL CAUSES OF SIGN DIFFERENCES")
    print("=" * 50)

    causes = [
        "1. Quaternion Convention Differences",
        "   - Original uses different quaternion order/convention",
        "   - Improved uses euler_xyz_to_quat_wxyz() conversion",
        "",
        "2. Rotation Matrix Sign Convention",
        "   - Different body/world frame conventions",
        "   - Transpose vs inverse rotation matrices",
        "",
        "3. Jacobian Sign Convention",
        "   - Contact Jacobian transpose sign",
        "   - Different foot frame definitions",
        "",
        "4. Gravity Term Convention",
        "   - Original: C(q,dq) + g(q) - J^T*F",
        "   - Improved: C(q,dq) + g(q) - J^T*F",
        "   - Sign difference in gravity implementation",
        "",
        "5. External Wrench Sign",
        "   - Different sign convention for external forces",
        "   - Ground reaction force direction",
        "",
        "6. Mass Matrix or Bias Forces",
        "   - Different implementation in ADAM library",
        "   - Parameter order differences",
    ]

    for line in causes:
        print(line)


def create_sign_analysis_plot(torques_orig, torques_improved):
    """Create visualization of sign patterns"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sign Pattern Analysis: Original vs Improved", fontsize=16)

    # 1. Scatter plot of original vs improved
    ax1.scatter(torques_orig.flatten(), torques_improved.flatten(), alpha=0.5, s=1)
    ax1.plot([-20, 20], [-20, 20], "r--", label="y=x (same)")
    ax1.plot([-20, 20], [20, -20], "b--", label="y=-x (opposite)")
    ax1.set_xlabel("Original Torques (Nm)")
    ax1.set_ylabel("Improved Torques (Nm)")
    ax1.set_title("Torque Correlation")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Sign agreement per joint
    joint_agreements = []
    for joint in range(12):
        same_sign = np.sum(
            np.sign(torques_orig[:, joint]) == np.sign(torques_improved[:, joint])
        )
        joint_agreements.append(same_sign / len(torques_orig) * 100)

    ax2.bar(range(1, 13), joint_agreements, color="skyblue")
    ax2.set_xlabel("Joint Number")
    ax2.set_ylabel("Same Sign Percentage (%)")
    ax2.set_title("Sign Agreement per Joint")
    ax2.grid(True, alpha=0.3)

    # 3. Correlation per joint
    correlations = []
    for joint in range(12):
        corr = np.corrcoef(torques_orig[:, joint], torques_improved[:, joint])[0, 1]
        correlations.append(corr)

    colors = [
        "red"
        if c < -0.5
        else "orange"
        if c < 0
        else "lightgreen"
        if c < 0.5
        else "green"
        for c in correlations
    ]
    ax3.bar(range(1, 13), correlations, color=colors)
    ax3.set_xlabel("Joint Number")
    ax3.set_ylabel("Correlation Coefficient")
    ax3.set_title("Correlation per Joint")
    ax3.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax3.axhline(y=-0.8, color="red", linestyle="--", alpha=0.5, label="Strong negative")
    ax3.grid(True, alpha=0.3)

    # 4. Time series example for worst joint
    worst_joint = np.argmin(correlations)
    time_steps = np.arange(len(torques_orig))
    ax4.plot(
        time_steps,
        torques_orig[:, worst_joint],
        "b-",
        label=f"Original Joint {worst_joint+1}",
    )
    ax4.plot(
        time_steps,
        torques_improved[:, worst_joint],
        "r-",
        label=f"Improved Joint {worst_joint+1}",
    )
    ax4.plot(
        time_steps,
        -torques_improved[:, worst_joint],
        "g--",
        label=f"-(Improved) Joint {worst_joint+1}",
    )
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Torque (Nm)")
    ax4.set_title(
        f"Example: Joint {worst_joint+1} (Correlation: {correlations[worst_joint]:.3f})"
    )
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/sign_analysis.png", dpi=150, bbox_inches="tight")
    print("Sign analysis plot saved to: results/sign_analysis.png")


def main():
    """Run complete sign analysis"""

    torques_orig, torques_improved = analyze_sign_patterns()
    if torques_orig is None:
        return

    investigate_potential_causes()
    create_sign_analysis_plot(torques_orig, torques_improved)

    print("\n🎯 CONCLUSION")
    print("=" * 50)
    print("The sign differences suggest a systematic issue with:")
    print("1. Frame conventions (body vs world)")
    print("2. Quaternion/rotation matrix sign")
    print("3. Contact Jacobian sign convention")
    print("4. External force direction convention")
    print("\nThis is likely NOT a bug, but a difference in sign conventions")
    print("between the original and improved implementations.")


if __name__ == "__main__":
    main()
