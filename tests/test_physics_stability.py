import pandas as pd
import numpy as np
from nexus_core.physics_engine.engine import physics_engine
from app.core.config import settings

def test_physics_stability():
    print("Testing Physics Engine Stability...")
    
    # CASE 1: Zero Volume / Flat Price (The NaN Trap)
    df_zero = pd.DataFrame({
        "open": [1.0] * 100,
        "high": [1.0] * 100,
        "low": [1.0] * 100,
        "close": [1.0] * 100,
        "volume": [0.0] * 100
    })
    
    physics_res = physics_engine.compute_vectorized_physics(df_zero)
    
    if physics_res.isnull().values.any():
        print("FAILED: NaN detected in Zero Volume case!")
    else:
        print("PASSED: Zero Volume stability confirmed.")
        
    # CASE 2: High Volatility Impulse (Logarithmic Compression Check)
    df_impulse = pd.DataFrame({
        "open": [1.0] * 99 + [1.0],
        "high": [1.1] * 99 + [2.0],
        "low": [0.9] * 99 + [1.0],
        "close": [1.0] * 99 + [2.0],
        "volume": [1000] * 99 + [1000000]
    })
    
    physics_res_imp = physics_engine.compute_vectorized_physics(df_impulse)
    force = physics_res_imp["physics_force"].iloc[-1]
    
    # log1p(large) should be manageable
    if force > 20.0:
         print(f"WARNING: Force seems dangerously high ({force}). Check log compression.")
    else:
         print(f"PASSED: Impulse compression stable. Force: {force:.4f}")

    # CASE 3: Training Targets
    if "target_kinetic_state_confirmed" in physics_res_imp.columns:
        print("PASSED: Training targets found in output.")
    else:
        print("FAILED: Training targets missing!")

if __name__ == "__main__":
    test_physics_stability()
