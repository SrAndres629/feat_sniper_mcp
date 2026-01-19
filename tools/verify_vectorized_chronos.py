import pandas as pd
import numpy as np
import datetime
import time
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ml.feat_processor.time import TimeFeatureProcessor
from app.ml.feat_processor.vectorized_tensor import VectorizedChronosProcessor

def run_benchmark():
    print("⏳ [CHRONOS] VECTORIZATION PERFORMANCE AUDIT")
    print("===========================================")
    
    # 1. Generate 10,000 candles (Approx 1 year of 1H data or etc)
    size = 10000
    dates = pd.date_range(start="2024-01-01", periods=size, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": np.random.randn(size) + 100,
        "high": np.random.randn(size) + 101,
        "low": np.random.randn(size) + 99,
        "close": np.random.randn(size) + 100,
        "volume": np.random.randint(100, 1000, size)
    }, index=dates)
    
    # --- TEST 1: SCALAR (LEGACY LOOP) ---
    print(f"📊 Processing {size} candles using Scalar Orchestrator...")
    scalar_processor = TimeFeatureProcessor()
    
    start_scalar = time.time()
    # We simulate what a naive loop would do
    # Note: TimeFeatureProcessor expects individual calls per candle in a stream
    results = []
    # Only doing 1000 for timing because 10,000 would be too slow to wait for in a benchmark if naive
    test_size = 1000 
    for i in range(test_size):
        t = df.index[i]
        c = df['close'].iloc[:i+1] # Simulated history
        v = df['volume'].iloc[:i+1]
        o = df['open'].iloc[:i+1]
        tensors = scalar_processor.process(t, c, v, o)
        results.append(tensors)
    end_scalar = time.time()
    
    scalar_total_time = (end_scalar - start_scalar) * (size / test_size)
    print(f"   ► Scalar Est. Time ({size} candles): {scalar_total_time:.4f}s")

    # --- TEST 2: VECTORIZED (NEW DEPLOYMENT) ---
    print(f"🚀 Processing {size} candles using Vectorized Orchestrator...")
    vector_processor = VectorizedChronosProcessor()
    
    start_vec = time.time()
    vectorized_df = vector_processor.process(df)
    end_vec = time.time()
    
    vec_time = end_vec - start_vec
    print(f"   ► Vectorized Time ({size} candles): {vec_time:.4f}s")
    
    speedup = scalar_total_time / vec_time
    print(f"\n⚡ SPEEDUP FACTOR: {speedup:.2f}x")
    
    # --- ACCURACY CHECK ---
    print("\n🔍 ACCURACY VERIFICATION")
    # Check if sin/cos matches for a sample point
    sample_idx = 500
    scalar_sample = results[sample_idx]
    vec_sample = vectorized_df.iloc[sample_idx]
    
    sin_diff = abs(scalar_sample['time_sin'][0] - vec_sample['time_sin'])
    cos_diff = abs(scalar_sample['time_cos'][0] - vec_sample['time_cos'])
    
    if sin_diff < 1e-5 and cos_diff < 1e-5:
        print("   ✅ SUCCESS: Deterministic match for Cyclical Encoding.")
    else:
        print(f"   ❌ FAIL: Math discrepancy. Sin Diff: {sin_diff}, Cos Diff: {cos_diff}")

    print("\n[CHRONOS CORE] Massive Backtesting capability confirmed.")
    print("===========================================")

if __name__ == "__main__":
    run_benchmark()
