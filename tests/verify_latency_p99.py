import pandas as pd
import numpy as np
import time
import logging
from app.skills.indicators import calculate_feat_layers
from nexus_brain.inference_api import neural_api

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("latency_bench")

async def benchmark_latency_p99(iterations: int = 100):
    """
    Certificación de Latencia Phase 12: MFAT Pipeline (< 5ms target).
    """
    print(f"🧪 [LATENCY] Starting MFAT Deep-Scan ({iterations} iterations)...")
    
    # 1. Setup Synthetic Data
    periods = 2500 # Enough for Macro + Bias
    df = pd.DataFrame({
        'close': np.random.randn(periods).cumsum() + 2000,
        'high': np.random.randn(periods) + 2005,
        'low': np.random.randn(periods) + 1995,
        'open': np.random.randn(periods) + 2000
    })
    
    latencies = []
    
    # 2. Warm up buffer
    for i in range(len(df)):
        await neural_api.predict_next_candle(df.iloc[i].to_dict())
        
    print(f"✅ [LATENCY] Buffer warmed: {neural_api.history_len} samples.")
    
    # 3. Targeted Benchmark
    for i in range(iterations):
        tick_data = {
            'close': 2000 + np.random.randn(),
            'high': 2005,
            'low': 1995,
            'bid': 2000,
            'ask': 2001
        }
        
        start = time.time()
        await neural_api.predict_next_candle(tick_data)
        end = time.time()
        
        latencies.append((end - start) * 1000)
        
    p99 = np.percentile(latencies, 99)
    avg = np.mean(latencies)
    
    print(f"📊 [LATENCY] RESULTS:")
    print(f"   P99 Latency: {p99:.2f}ms")
    print(f"   AVG Latency: {avg:.2f}ms")
    
    if p99 < 5:
        print("✅ [CERTIFIED] Phase 12 Target Met: MFAT Pipeline is institutional grade.")
    else:
        print(f"🚨 [FAIL] Latency Breach: {p99:.2f}ms > 5ms. Optimization required.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(benchmark_latency_p99())
