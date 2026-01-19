"""
FEAT NEXUS: SOFT LABELING (Neural Purity)
=========================================
Converts hard trade outcomes into probabilistic soft labels.
Prevents neural 'mode collapse' by acknowledging the spectrum between 
Quick Scalps, Day Trends, and Structural Swings.
"""

import numpy as np
from typing import List, Tuple

def calculate_soft_labels(profit_pips: float, duration_mins: float) -> np.ndarray:
    """
    Computes a 3-class probability distribution [P_Scalp, P_Day, P_Swing].
    
    Logic:
    - Scalp: High efficiency, short duration (< 15 mins).
    - Day: Moderate duration (15m - 4h), trend captures.
    - Swing: Long duration (> 4h), structural sweeps.
    """
    
    # 1. Base scores
    s_scalp = 0.0
    s_day = 0.0
    s_swing = 0.0
    
    # --- SCALP LOGIC ---
    if duration_mins <= 15:
        s_scalp = 1.0
    elif duration_mins <= 30:
        s_scalp = max(0.0, 1.0 - (duration_mins - 15) / 15.0)
        
    # --- DAY LOGIC ---
    if 15 < duration_mins <= 240:
        # Peak at 60 mins
        if duration_mins <= 60:
            s_day = (duration_mins - 15) / 45.0
        else:
            s_day = max(0.0, 1.0 - (duration_mins - 60) / 180.0)
            
    # --- SWING LOGIC ---
    if duration_mins > 60:
        if duration_mins <= 240:
            s_swing = (duration_mins - 60) / 180.0
        else:
            s_swing = 1.0

    # 2. Performance Weighting (Profit factor)
    # A massive profit in short time is 'Super-Scalp'
    # A tiny profit in long time is 'Weak-Swing'
    pips_per_min = abs(profit_pips) / (duration_mins + 1e-9)
    
    if pips_per_min > 2.0: # Hyper-efficient
        s_scalp *= 1.2
    
    # 3. Softmax Normalization
    scores = np.array([s_scalp, s_day, s_swing], dtype=np.float64)
    # Add small epsilon to avoid total zeros
    scores += 1e-3
    
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores)
    
    return probs

def generate_multi_head_label(profit_pips: float, duration_mins: float) -> Tuple[np.ndarray, int]:
    """
    Returns (SoftLabels, HardIndex) for both cross-entropy and soft targets.
    """
    probs = calculate_soft_labels(profit_pips, duration_mins)
    hard_idx = int(np.argmax(probs))
    return probs, hard_idx

if __name__ == "__main__":
    # Test cases
    test_cases = [
        (15, 5),   # 15 pips in 5m (Scalp)
        (40, 120), # 40 pips in 2h (Day)
        (150, 600), # 150 pips in 10h (Swing)
        (20, 25),  # Border case Scalp/Day
    ]
    
    print("--- SOFT LABEL AUDIT ---")
    for p, d in test_cases:
        lbl, _ = generate_multi_head_label(p, d)
        print(f"Profit: {p}p | Dur: {d}m -> [S: {lbl[0]:.2f}, D: {lbl[1]:.2f}, Sw: {lbl[2]:.2f}]")
