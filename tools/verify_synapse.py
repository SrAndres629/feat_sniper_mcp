import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ml.feat_processor.engineering import apply_feat_engineering
from nexus_core.structure_engine import structure_engine

def verify_synapse():
    print("==================================================")
    print("🧠 [CORTEX] SYNAPTIC INTEGRATION AUDIT")
    print("==================================================")
    
    # 1. Generate Synthetic Data with Structure
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1H")
    prices = np.linspace(100, 110, 100)
    # Create a fake gap for FVG detection
    prices[50] = prices[49] + 2.0 
    
    df = pd.DataFrame({
        "close": prices, 
        "high": prices+0.5, 
        "low": prices-0.5, 
        "open": prices,
        "volume": np.random.rand(100)*1000
    })
    
    # 2. Run Main Feature Engineer (The Brain)
    try:
        df_feat = apply_feat_engineering(df)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n[OUTPUT] DataFrame Shape: {df_feat.shape}")
    cols = df_feat.columns.tolist()
    
    # 3. Check for Quantum Organs
    
    # A. BREAKER / INVERSION (Float Check)
    struct_cols = ["breaker_bull", "inversion_bull"]
    print("\n[CHECK 1] Structural Geometry (Breakers/Inversion):")
    for c in struct_cols:
        if c in cols:
            dtype = df_feat[c].dtype
            print(f"   ✅ {c} present. Type: {dtype}")
            if not np.issubdtype(dtype, np.number):
                 print(f"      ⚠️ WARNING: {c} should be float/int, found {dtype}")
        else:
            print(f"   ❌ {c} MISSING")

    # B. KINETICS (One-Hot & Physics)
    kin_cols = [
        "kinetic_is_expansion", "kinetic_is_compression", "kinetic_context", 
        "feat_force", "wick_ratio", "feat_force_z", 
        "kinetic_state_impulse", "kinetic_state_confirmed", "kinetic_state_failed",
        "feat_efficiency", "micro_accel", "structure_accel", "macro_accel"
    ]
    print("\n[CHECK 2] Kinetic Tensors (One-Hot & Doctoral Physics):")
    for c in kin_cols:
        if c in cols:
             print(f"   ✅ {c} present.")
        else:
             print(f"   ❌ {c} MISSING")

    # C. SPACE (Intensity)
    space_cols = ["confluence_score", "space_intensity"]
    print("\n[CHECK 3] Space Tensor (Intensity):")
    for c in space_cols:
         if c in cols:
             mean_val = df_feat[c].mean()
             print(f"   ✅ {c} present. Mean: {mean_val:.4f}")
         else:
             print(f"   ❌ {c} MISSING")

    # D. MACRO SENTINEL (One-Hot)
    macro_cols = ["macro_safe", "macro_caution", "macro_danger", "position_multiplier", "minutes_to_event"]
    print("\n[CHECK 4] Macro Sentinel Tensor (One-Hot & Risk):")
    for c in macro_cols:
         if c in cols:
             mean_val = df_feat[c].mean()
             print(f"   ✅ {c} present. Value: {mean_val:.4f}")
         else:
             print(f"   ❌ {c} MISSING")

    print("\n==================================================")
    if all(x in cols for x in (struct_cols + kin_cols + space_cols)):
        print("✅ SYSTEM READY. SYNAPSE CONNECTED.")
        print("   The Neural Network can now see Physics.")
    else:
        print("❌ SYSTEM PARTIALLY CONNECTED. SEE LOGS.")
    print("==================================================")

if __name__ == "__main__":
    verify_synapse()
