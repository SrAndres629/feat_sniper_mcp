"""
FEAT NEXUS: TRAINING PIPELINE VERIFICATION
==========================================
End-to-end validation of the complete training data generation pipeline.
Ensures all components work together for institutional-grade model training.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')

from app.ml.feat_processor.spectral import SpectralTensorBuilder
from app.ml.training.labeling import generate_multi_head_label

def generate_mock_trade_data(n_bars: int = 200):
    """Generates realistic OHLCV data with trade outcome."""
    base_price = 2000.0
    prices = base_price + np.cumsum(np.random.randn(n_bars) * 0.5)
    
    df = pd.DataFrame({
        'open': prices + np.random.uniform(-0.2, 0.2, n_bars),
        'high': prices + np.random.uniform(0.5, 1.5, n_bars),
        'low': prices - np.random.uniform(0.5, 1.5, n_bars),
        'close': prices,
        'tick_volume': np.random.uniform(100, 1000, n_bars)
    })
    
    # Simulate trade outcome
    profit_pips = np.random.uniform(5, 200)
    duration_mins = np.random.uniform(5, 500)
    
    return df, profit_pips, duration_mins

def verify_training_pipeline():
    print("=== FEAT NEXUS: TRAINING PIPELINE VERIFICATION ===\n")
    
    builder = SpectralTensorBuilder()
    
    # 1. GENERATE SAMPLE TRAINING DATA
    print("[STEP 1: GENERATE TRAINING SAMPLE]")
    n_samples = 5
    training_data = []
    
    for i in range(n_samples):
        df, profit, duration = generate_mock_trade_data()
        
        # Feature extraction
        features = builder.build_tensors(df)
        
        # Label generation
        soft_labels, hard_idx = generate_multi_head_label(profit, duration)
        
        sample = {
            "features": features,
            "soft_labels": soft_labels,
            "hard_idx": hard_idx,
            "profit": profit,
            "duration": duration
        }
        training_data.append(sample)
        print(f"  Sample {i+1}: {len(features)} features | Label: {['Scalp', 'Day', 'Swing'][hard_idx]}")
    
    # 2. VALIDATE FEATURE COMPLETENESS
    print("\n[STEP 2: FEATURE COMPLETENESS CHECK]")
    expected_features = [
        "domino_alignment", "elastic_gap", "kinetic_whip", "bias_regime",
        "energy_burst", "trend_purity", "spectral_divergence", "sc10_axis",
        "volume_profile_tensor", "volume_shape_label", "vol_scalar", "wavelet_level",
        "sgi_gravity", "vam_purity", "svc_confluence", "auction_physics_divergence"
    ]
    
    sample_features = training_data[0]["features"]
    missing = [f for f in expected_features if f not in sample_features]
    extra = [f for f in sample_features.keys() if f not in expected_features]
    
    print(f"  Expected: {len(expected_features)} features")
    print(f"  Generated: {len(sample_features)} features")
    print(f"  Missing: {missing if missing else 'None'}")
    print(f"  Extra: {extra if extra else 'None'}")
    print(f"  Status: {'✅ COMPLETE' if not missing else '❌ INCOMPLETE'}")
    
    # 3. VALIDATE LABEL INTEGRITY
    print("\n[STEP 3: LABEL INTEGRITY CHECK]")
    for i, sample in enumerate(training_data):
        probs = sample["soft_labels"]
        prob_sum = np.sum(probs)
        is_valid = np.isclose(prob_sum, 1.0, atol=1e-6) and all(p >= 0 for p in probs)
        print(f"  Sample {i+1}: sum={prob_sum:.4f} | Valid: {'✅' if is_valid else '❌'}")
    
    # 4. VALIDATE TENSOR SHAPES
    print("\n[STEP 4: TENSOR SHAPE VALIDATION]")
    vol_tensor = sample_features.get("volume_profile_tensor", [])
    expected_vol_shape = 64
    actual_vol_shape = len(vol_tensor)
    print(f"  volume_profile_tensor: expected {expected_vol_shape}, got {actual_vol_shape}")
    print(f"  Shape Status: {'✅ CORRECT' if actual_vol_shape == expected_vol_shape else '❌ MISMATCH'}")
    
    # 5. NUMERICAL STABILITY CHECK
    print("\n[STEP 5: NUMERICAL STABILITY CHECK]")
    has_nan = False
    has_inf = False
    
    for key, val in sample_features.items():
        if isinstance(val, (int, float)):
            if np.isnan(val) or np.isinf(val):
                print(f"  ⚠️ {key}: NaN/Inf detected!")
                has_nan = True
        elif isinstance(val, list):
            arr = np.array(val)
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                print(f"  ⚠️ {key}: NaN/Inf in array!")
                has_inf = True
    
    if not has_nan and not has_inf:
        print("  All values numerically stable: ✅")
    
    # 6. FINAL VERDICT
    print("\n" + "="*50)
    all_pass = not missing and not has_nan and not has_inf
    print(f"TRAINING PIPELINE STATUS: {'✅ READY FOR PRODUCTION' if all_pass else '❌ NEEDS FIXES'}")
    print("="*50)
    
    return all_pass

if __name__ == "__main__":
    verify_training_pipeline()
