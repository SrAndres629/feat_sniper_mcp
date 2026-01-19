
import os
import sys
import pandas as pd
import numpy as np

# Adjust path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from app.ml.feat_processor.spectral import SpectralFeatureProcessor

def create_pullback_scenario():
    """
    Simulates a perfect pullback:
    - Long term trend is up (SC_10 > SC_9 ...).
    - Short term price drops to SC_4 (Sniper Zone).
    """
    n = 3000 # Need enough for 2048 EMA
    dates = pd.date_range("2025-01-01", periods=n, freq="H")
    
    # Base: Price starts at 2000 and goes up 0.1 per bar
    price = 2000 + np.cumsum(np.random.normal(0.1, 0.5, n))
    
    # Create a pullback at the end
    # Drop price significantly in the last 10 bars
    # But keep it above the 2048 EMA
    price[-20:] = price[-21] - np.linspace(0, 30, 20) 
    
    df = pd.DataFrame({
        "close": price,
        "high": price + 2,
        "low": price - 2,
        "volume": np.random.randint(1000, 5000, n)
    }, index=dates)
    
    return df

def test_deca_core():
    processor = SpectralFeatureProcessor()
    
    # 1. Test Bullish Pullback
    print("--- TESTING PULLBACK PERFECTO ---")
    df = create_pullback_scenario()
    features = processor.process_spectral_features(df)
    
    print(f"Domino Alignment: {features['domino_alignment']:.4f}")
    print(f"Sniper Proximity: {features['sniper_proximity']:.4f}")
    print(f"Kinetic Whip: {features['kinetic_whip']:.4f}")
    print(f"Bias Regime: {features['bias_regime']}")
    
    # Logic Validation
    if features['bias_regime'] > 0 and features['domino_alignment'] > 0.5:
        if abs(features['sniper_proximity']) < 0.5:
            print(">> [VALIDATED] SNIPER ENTRY DETECTED (Price at SC_4 in Bull Trend)")
    
    if features['domino_alignment'] < 0.3:
        print(">> [GOVERNANCE] Reducing position size (Chaos detected)")

if __name__ == "__main__":
    test_deca_core()
