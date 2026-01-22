import pandas as pd
import numpy as np
from nexus_core.structure_engine.engine import structure_engine

def generate_institutional_data(n=200):
    prices = np.full(n, 1000.0)
    
    # 1. CREATE MINOR H-FRACTAL at bar 50
    # Prices 48..52: 1000, 1000, 1020, 1000, 1000
    prices[50] = 1020 
    
    # 2. INDUCEMENT SWEEP (Bar 80 - Sweeps 1020 and reserves)
    prices[80] = 1030
    
    # 3. SWING LOW (Bar 100)
    prices[100] = 980
    
    # 4. BEARISH CANDLE BEFORE EXPANSION (at index 100)
    # We'll make bar 99 bearish
    
    df = pd.DataFrame({
        "open":  prices.copy(),
        "high":  prices + 1.0,
        "low":   prices - 1.0,
        "close": prices.copy(),
        "volume":[10000] * n
    })
    
    df.at[99, "open"] = 985
    df.at[99, "close"] = 975
    df.at[99, "high"] = 986
    df.at[99, "low"] = 974
    
    # 5. MASSIVE INSTITUTIONAL EXPANSION (BOS of 1030)
    # Start at 101, jump with explicit GAPS for FVG
    # Index 101: 1000
    # Index 102: 1050 (GAP)
    # Index 103: 1100 (GAP)
    for i in range(101, 120):
        val = 1000 + (i - 101) * 20
        df.at[i, "open"] = val - 15
        df.at[i, "close"] = val + 15
        df.at[i, "high"] = val + 18
        df.at[i, "low"] = val - 17
        df.at[i, "volume"] = 60000
        
    return df

def test_structure_system():
    print("Executing Institutional Topology Unit Test (v6.3)...")
    df = generate_institutional_data(200)
    
    # RUN ENGINE
    df = structure_engine.compute_feat_index(df)
    
    # DIAGNOSTICS
    print("\n--- Structural Diagnostics ---")
    if 'physics_force' in df.columns:
        print(f"Max Physics Force: {df['physics_force'].max():.2f}")
    else:
        print("F-Missing")
        
    print(f"BOS Detected: {(df['bos_bull'] == True).any()}")
    print(f"OB Created: {(df['ob_bull'] == 1.0).any()}")
    print(f"FVG Gravity Max: {df.get('fvg_gravity', pd.Series(0.0)).max():.4f}")
    print(f"Max Feat Index: {df['feat_index'].max()}")
    
    print("\n--- Final Checks ---")
    if (df["bos_bull"] == True).any() and (df["fvg_gravity"].abs().max() > 0.001):
        print("✅ SUCCESS: SMC Vector Engine Functional.")
    else:
        print("❌ FAILED: Neural channels incomplete.")

if __name__ == "__main__":
    test_structure_system()
