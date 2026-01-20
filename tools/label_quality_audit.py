"""
FEAT NEXUS: LABEL QUALITY AUDIT
===============================
Validates the integrity of soft label distributions for multi-head training.
Ensures no class collapse, proper normalization, and smooth probability curves.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from app.ml.training.labeling import calculate_soft_labels, generate_multi_head_label

def run_label_quality_audit():
    print("=== LABEL QUALITY AUDIT (Phase 7) ===\n")
    
    # 1. TEST: Coverage of all classes
    print("[TEST 1: CLASS COVERAGE]")
    test_scenarios = [
        # (profit, duration, expected_dominant_class)
        (10, 3, "Scalp"),      # Fast, small profit
        (25, 10, "Scalp"),     # Fast, decent profit
        (50, 60, "Day"),       # Medium duration, trend capture
        (80, 180, "Day"),      # Long day trade
        (200, 480, "Swing"),   # Multi-session swing
        (500, 1440, "Swing"),  # Full day swing
    ]
    
    class_names = ["Scalp", "Day", "Swing"]
    coverage = {c: 0 for c in class_names}
    
    for profit, duration, expected in test_scenarios:
        probs, hard_idx = generate_multi_head_label(profit, duration)
        dominant = class_names[hard_idx]
        coverage[dominant] += 1
        
        status = "✅" if dominant == expected else "⚠️"
        print(f"  {status} {profit}p/{duration}m -> {dominant} (expected: {expected}) | Probs: {probs.round(2)}")
    
    print(f"\n  Coverage: {coverage}")
    all_covered = all(v > 0 for v in coverage.values())
    print(f"  All Classes Represented: {'✅ YES' if all_covered else '❌ NO - MODE COLLAPSE RISK'}")
    
    # 2. TEST: Probability Sum = 1.0
    print("\n[TEST 2: PROBABILITY NORMALIZATION]")
    n_samples = 100
    errors = 0
    for _ in range(n_samples):
        profit = np.random.uniform(1, 500)
        duration = np.random.uniform(1, 2000)
        probs, _ = generate_multi_head_label(profit, duration)
        prob_sum = np.sum(probs)
        if not np.isclose(prob_sum, 1.0, atol=1e-6):
            errors += 1
    
    print(f"  Samples tested: {n_samples}")
    print(f"  Normalization errors: {errors}")
    print(f"  Status: {'✅ PASS' if errors == 0 else '❌ FAIL'}")
    
    # 3. TEST: Smooth transition curves
    print("\n[TEST 3: SMOOTH TRANSITION (Scalp -> Day)]")
    transition_probs = []
    for d in range(1, 61, 5):
        probs, _ = generate_multi_head_label(50, d)
        transition_probs.append((d, probs[0], probs[1]))  # Scalp, Day
    
    print("  Duration | Scalp  | Day")
    print("  ---------|--------|------")
    for d, s, dy in transition_probs:
        bar_s = "█" * int(s * 10)
        bar_d = "█" * int(dy * 10)
        print(f"  {d:3d} min  | {s:.2f} {bar_s:10} | {dy:.2f} {bar_d:10}")
    
    # Check monotonicity
    scalp_decreasing = all(transition_probs[i][1] >= transition_probs[i+1][1] for i in range(len(transition_probs)-1))
    print(f"\n  Scalp Monotonically Decreasing: {'✅ YES' if scalp_decreasing else '⚠️ NOT STRICTLY'}")
    
    print("\n=== AUDIT COMPLETE ===")

if __name__ == "__main__":
    run_label_quality_audit()
