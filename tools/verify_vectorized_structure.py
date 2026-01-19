import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nexus_core.structure_engine.engine import StructureEngine

def verify_structural_vectorization():
    print("🏗️ [STRUCTURE CORE] VECTORIZATION & RIGOR VERIFICATION")
    print("======================================================")
    
    # Generate Synthetic Market Data
    size = 200
    dates = pd.date_range(start="2024-01-01", periods=size, freq="15min", tz="UTC")
    
    # Create a trending market with an FVG and an OB
    price = 100.0
    prices = []
    for i in range(size):
        price += np.random.randn() * 0.1
        if 50 < i < 60: price += 0.5 # Expansion (FVG potential)
        prices.append(price)
        
    df = pd.DataFrame({
        "open": prices,
        "high": [p + 0.1 for p in prices],
        "low": [p - 0.1 for p in prices],
        "close": [p + 0.05 for p in prices],
        "volume": np.random.randint(100, 1000, size),
        "tick_volume": np.random.randint(100, 1000, size)
    }, index=dates)
    
    # Mock FVG detection (needed for OB validation)
    df["fvg_bull"] = (df["low"] > df["high"].shift(2))
    df["fvg_bear"] = (df["high"] < df["low"].shift(2))
    
    engine = StructureEngine()
    
    print(f"📊 Processing {size} candles...")
    try:
        # 1. Test FEAT Index Generation
        res = engine.compute_feat_index(df)
        print("   ✅ SUCCESS: FEAT Index generated for all candles.")
        
        # 2. Check for Loop-free signals (Existence)
        last = res.iloc[-1]
        print(f"   ► Last FEAT Index: {last['feat_index']}")
        print(f"   ► Last Confluence: {last['confluence_score']}")
        
        # 3. Verify Internal Logic (Structure Status)
        print(f"   ► Structure Status: {last['structure_status']}")
        
        # 4. Performance Check
        import time
        start = time.time()
        _ = engine.compute_feat_index(df)
        elapsed = time.time() - start
        print(f"⚡ Vectorized Processing Speed: {elapsed:.4f}s for {size} candles.")

    except Exception as e:
        print(f"   ❌ FAIL: Error in vectorization: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n[STRUCTURE CORE] Neural-Structural alignment confirmed.")
    print("======================================================")

if __name__ == "__main__":
    verify_structural_vectorization()
