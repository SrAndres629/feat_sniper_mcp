import numpy as np
import pandas as pd
import time
from app.skills.volume_profile import volume_profile

def simulate_market_scenario(scenario="balanced"):
    """Generates synthetic price/volume data for different regimes."""
    np.random.seed(42)
    n = 200
    
    if scenario == "balanced":
        # D-Shape: Normal distribution centered in range
        prices = np.random.normal(2000, 5, n)
        volumes = np.random.uniform(10, 100, n)
    elif scenario == "p_shape":
        # P-Shape: Peak at top (Short covering / Bullish trend)
        prices = np.concatenate([np.random.normal(1990, 10, 50), np.random.normal(2020, 2, 150)])
        volumes = np.random.uniform(10, 100, n)
    elif scenario == "b_shape":
        # b-Shape: Peak at bottom (Long liquidation / Bearish trend)
        prices = np.concatenate([np.random.normal(1980, 2, 150), np.random.normal(2010, 10, 50)])
        volumes = np.random.uniform(10, 100, n)
    else:
        prices = np.linspace(1980, 2020, n)
        volumes = np.random.uniform(10, 100, n)
        
    df = pd.DataFrame({'close': prices, 'tick_volume': volumes})
    return df

def run_volume_audit():
    print("=== VOLUME CORE AUDIT: OPERATION LIQUID DENSITY ===")
    
    scenarios = ["balanced", "p_shape", "b_shape"]
    
    for scene in scenarios:
        print(f"\n[SCENARIO: {scene.upper()}]")
        df = simulate_market_scenario(scene)
        
        start_time = time.perf_counter()
        profile = volume_profile.get_profile(df)
        end_time = time.perf_counter()
        
        latency = (end_time - start_time) * 1000
        
        if not profile:
            print("❌ Error: Profile calculation failed.")
            continue
            
        print(f"✅ Latency: {latency:.2f}ms (Target: <5ms)")
        print(f"📊 POC: {profile['poc']:.2f} | VAH: {profile['vah']:.2f} | VAL: {profile['val']:.2f}")
        print(f"💎 Shape Verdict: {profile['shape']}")
        print(f"🧮 Kurtosis: {profile['kurtosis']:.2f} | Skew: {profile['skew']:.2f}")
        
        # Check Tensor Consistency
        tensor = profile['profile_tensor']
        print(f"🧠 Tensor Size: {len(tensor)} | Max Normalized: {np.max(tensor):.2f}")
        
        # Validation Logic
        if scene == "balanced" and "D-Shape" in profile['shape']:
            print("🌟 VERDICT: ACCURATE")
        elif scene == "p_shape" and "P-Shape" in profile['shape']:
            print("🌟 VERDICT: ACCURATE")
        elif scene == "b_shape" and "b-Shape" in profile['shape']:
            print("🌟 VERDICT: ACCURATE")
        else:
            print("⚠️ VERDICT: SENSITIVITY CALIBRATION NEEDED")

if __name__ == "__main__":
    run_volume_audit()
