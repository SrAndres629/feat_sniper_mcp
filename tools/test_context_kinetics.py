import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nexus_core.acceleration.engine import acceleration_engine

def create_mock_df(jump=5.0, vol=1000.0, zone=0.0):
    df = pd.DataFrame({
        "close": np.linspace(100, 105, 50),
        "open": np.linspace(99, 104, 50),
        "high": np.linspace(101, 106, 50),
        "low": np.linspace(98, 103, 50),
        "volume": np.random.rand(50) * 500
    })
    # Apply State
    df.iloc[-1, df.columns.get_loc("close")] = 105.0 + jump # Jump
    df.iloc[-1, df.columns.get_loc("volume")] = vol
    df["confluence_score"] = 0.0
    df.iloc[-1, df.columns.get_loc("confluence_score")] = zone
    return df

def verify_context():
    print("🧪 CONTEXTUAL KINETICS SIMULATION (DUAL TEST)")
    
    # CASE 1: EXHAUSTION (Strong move into Wall)
    # Jump=3 (Medium), Vol=2000 (High), Zone=3.5 (Wall)
    df_ex = create_mock_df(jump=2.0, vol=1500.0, zone=3.5)
    res_ex = acceleration_engine.compute_acceleration_features(df_ex)
    ctx_ex = res_ex.iloc[-1].get("kinetic_context", 0)
    acc_ex = res_ex.iloc[-1]["accel_score"]
    
    print(f"\n[CASE 1] Exhaustion Test (Accel={acc_ex:.2f}, Zone=3.5)")
    if ctx_ex == -1: print("✅ SUCCESS: Diagnosed EXHAUSTION")
    else: print(f"❌ FAIL: Got {ctx_ex} (Expected -1)")

    # CASE 2: BREAKOUT (Massive move through Wall)
    # Jump=10 (Huge), Vol=5000 (Massive), Zone=3.5 (Wall)
    df_bk = create_mock_df(jump=10.0, vol=5000.0, zone=3.5)
    res_bk = acceleration_engine.compute_acceleration_features(df_bk)
    ctx_bk = res_bk.iloc[-1].get("kinetic_context", 0)
    acc_bk = res_bk.iloc[-1]["accel_score"]
    
    print(f"\n[CASE 2] Breakout Test (Accel={acc_bk:.2f}, Zone=3.5)")
    if ctx_bk == 1: print("✅ SUCCESS: Diagnosed BREAKOUT")
    else: print(f"❌ FAIL: Got {ctx_bk} (Expected 1)")

if __name__ == "__main__":
    verify_context()
