import sys
import os
import pandas as pd
import numpy as np

# Python path configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nexus_core.kinetic_engine import kinetic_engine
from app.ml.feat_processor.kinetics import calculate_multifractal_layers

def verify_kinetics():
    print("==================================================")
    print("🧠 [CORTEX] KINETIC TENSOR AUDIT")
    print("==================================================")
    
    # 1. Generate Synthetic Data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1H")
    prices = np.linspace(100, 110, 100) # Simple trend
    df = pd.DataFrame({"close": prices, "high": prices+1, "low": prices-1, "open": prices}, index=dates)
    
    # 2. Run Feat Processor (Vectorized)
    df = calculate_multifractal_layers(df)
    print(f"[FEAT] Columns Added: {[c for c in df.columns if 'kinetic' in c]}")
    
    if "kinetic_is_expansion" in df.columns:
        print("✅ [FEAT] One-Hot Columns Detected in DataFrame")
    else:
        print("❌ [FEAT] One-Hot Columns MISSING")

    # 3. Run Kinetic Engine (Scalar/Live)
    metrics = kinetic_engine.compute_kinetic_state(df)
    pattern_tensor = kinetic_engine.detect_kinetic_patterns(metrics)
    
    print("\n[ENGINE] Live Tensor Output:")
    keys = list(pattern_tensor.keys())
    print(f"   Keys: {keys}")
    
    # Validation
    one_hot_keys = ["kinetic_is_expansion", "kinetic_is_compression", "kinetic_is_reversal"]
    present = all(k in pattern_tensor for k in one_hot_keys)
    
    if present:
        print(f"✅ [ENGINE] One-Hot Tensor Confirmed. Shape: {len(keys)} dimensions.")
        print(f"   Sample: {pattern_tensor}")
    else:
        print("❌ [ENGINE] One-Hot Tensor FAILED. Still using scalar ID?")

    print("==================================================")

if __name__ == "__main__":
    verify_kinetics()
