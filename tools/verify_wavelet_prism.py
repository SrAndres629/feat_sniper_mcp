import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ml.feat_processor.spectral import SpectralTensorBuilder

def create_market_scenario():
    """
    Creates a 'Stop Hunt' scenario: 
    Stable uptrend followed by a violent downward wick that quickly recovers.
    """
    np.random.seed(42)
    x = np.linspace(0, 10, 200)
    
    # 1. Base Trend (Uptrend)
    trend = 100 + x * 0.5
    
    # 2. Add White Noise
    noise = np.random.normal(0, 0.05, 200)
    price = trend + noise
    
    # 3. The 'Stop Hunt' (Judas Swing) at index 180-185
    # Violent crash and immediate recovery
    price[180:183] -= 2.5 
    
    return pd.DataFrame({"close": price})

def verify_quantum_prism():
    df = create_market_scenario()
    builder = SpectralTensorBuilder()
    
    print("\n--- [QUANTUM PRISM AUDIT] ---")
    
    # 1. Processing
    tensors = builder.build_tensors(df)
    
    # 2. Validation of Binocular Vision
    # trend_purity should be high generally, but drop during the hunt
    print(f"Trend Purity: {tensors['trend_purity']:.4f}")
    print(f"Energy Burst: {tensors['energy_burst']:.4f}")
    print(f"Spectral Divergence: {tensors['spectral_divergence']:.6f}")
    
    # Check if sc10 (from Wavelet side) is smoother than sc1 (from Raw side)
    # This is a conceptual check.
    
    if tensors['energy_burst'] > 0:
        print("✅ Energy Burst Detected (System is Alive).")
    
    if tensors['trend_purity'] < 1.0:
        print("✅ Signal possesses realistic complexity (Non-unity purity).")
        
    print("\n--- [TENSORES GENERADOS] ---")
    for k, v in tensors.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    verify_quantum_prism()
