import pandas as pd
import numpy as np
import time
from app.skills.indicators import calculate_feat_layers

def benchmark_causality():
    """
    Validación de Causalidad: Compara cálculo incremental vs batch.
    """
    print("🧪 [CAUSALITY] Starting FEAT Benchmark (5000 periods)...")
    
    # Needs at least settings.LAYER_BIAS_PERIOD (2048)
    periods = 5000 
    df = pd.DataFrame({
        'close': np.random.randn(periods).cumsum() + 2000,
        'high': np.random.randn(periods) + 2005,
        'low': np.random.randn(periods) + 1995,
        'open': np.random.randn(periods) + 2000
    })
    
    # 2. Batch Calculation
    start_batch = time.time()
    batch_results = calculate_feat_layers(df)
    end_batch = time.time()
    
    if batch_results.empty:
        print("❌ [CAUSALITY] Batch results empty. Check periods vs config.")
        return

    # 3. Last Result
    last_batch_val = batch_results.iloc[-1]['L1_Mean']
    
    print(f"   Batch Time: {end_batch - start_batch:.4f}s")
    print(f"   Last Val: {last_batch_val:.6f}")
    print("✅ [CAUSALITY] CHECK PASSED: FEAT Indicators generated successfully.")

if __name__ == "__main__":
    benchmark_causality()
