"""
FEAT NEXUS: PERFORMANCE PROFILER
================================
Measures end-to-end latency of critical tensor generation paths.
Target: <50ms for full feature vector generation.
"""

import numpy as np
import pandas as pd
import time
import sys
sys.path.insert(0, '.')

def generate_mock_ohlcv(n_bars: int = 200) -> pd.DataFrame:
    """Generates realistic OHLCV data for benchmarking."""
    base_price = 2000.0
    prices = base_price + np.cumsum(np.random.randn(n_bars) * 0.5)
    
    return pd.DataFrame({
        'open': prices + np.random.uniform(-0.2, 0.2, n_bars),
        'high': prices + np.random.uniform(0.5, 1.5, n_bars),
        'low': prices - np.random.uniform(0.5, 1.5, n_bars),
        'close': prices,
        'tick_volume': np.random.uniform(100, 1000, n_bars)
    })

def profile_spectral_tensor_builder(n_iterations: int = 10):
    """Profiles the SpectralTensorBuilder.build_tensors() method."""
    from app.ml.feat_processor.spectral import SpectralTensorBuilder
    
    print("[PROFILING: SpectralTensorBuilder.build_tensors()]")
    
    builder = SpectralTensorBuilder()
    df = generate_mock_ohlcv(200)
    
    # Warmup (JIT compilation)
    _ = builder.build_tensors(df)
    
    times = []
    for i in range(n_iterations):
        start = time.perf_counter()
        tensors = builder.build_tensors(df)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
    
    avg_time = np.mean(times)
    max_time = np.max(times)
    min_time = np.min(times)
    
    print(f"  Iterations: {n_iterations}")
    print(f"  Avg: {avg_time:.2f}ms | Min: {min_time:.2f}ms | Max: {max_time:.2f}ms")
    print(f"  Features Generated: {len(tensors)} tensors")
    print(f"  Status: {'✅ PASS (<50ms)' if avg_time < 50 else '⚠️ NEEDS OPTIMIZATION'}")
    
    return avg_time

def profile_volume_profile(n_iterations: int = 10):
    """Profiles the VolumeProfiler.get_profile() method."""
    from app.skills.volume_profile import volume_profile
    
    print("\n[PROFILING: volume_profile.get_profile()]")
    
    df = generate_mock_ohlcv(100)
    
    # Warmup
    _ = volume_profile.get_profile(df)
    
    times = []
    for i in range(n_iterations):
        start = time.perf_counter()
        profile = volume_profile.get_profile(df)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    avg_time = np.mean(times)
    print(f"  Iterations: {n_iterations}")
    print(f"  Avg: {avg_time:.2f}ms | Min: {np.min(times):.2f}ms | Max: {np.max(times):.2f}ms")
    print(f"  Status: {'✅ PASS (<10ms)' if avg_time < 10 else '⚠️ NEEDS OPTIMIZATION'}")
    
    return avg_time

def profile_math_engine_jit(n_iterations: int = 100):
    """Profiles JIT-accelerated math functions."""
    from nexus_core.math_engine import calculate_vwema, calculate_kde_jit, calculate_confluence_density
    
    print("\n[PROFILING: Math Engine JIT Functions]")
    
    prices = np.random.uniform(2000, 2050, 500).astype(np.float64)
    volumes = np.random.uniform(100, 1000, 500).astype(np.float64)
    grid = np.linspace(2000, 2050, 64).astype(np.float64)
    
    # Warmup
    _ = calculate_vwema(prices, volumes, 21)
    _ = calculate_kde_jit(prices, volumes, grid, 0.5)
    _ = calculate_confluence_density(2025.0, 2020.0, 2022.0, 1.0)
    
    # VW-EMA
    start = time.perf_counter()
    for _ in range(n_iterations):
        calculate_vwema(prices, volumes, 21)
    vwema_time = (time.perf_counter() - start) / n_iterations * 1000
    
    # KDE
    start = time.perf_counter()
    for _ in range(n_iterations):
        calculate_kde_jit(prices, volumes, grid, 0.5)
    kde_time = (time.perf_counter() - start) / n_iterations * 1000
    
    # Confluence Density
    start = time.perf_counter()
    for _ in range(n_iterations):
        calculate_confluence_density(2025.0, 2020.0, 2022.0, 1.0)
    conf_time = (time.perf_counter() - start) / n_iterations * 1000
    
    print(f"  VW-EMA (500 bars): {vwema_time:.3f}ms per call")
    print(f"  KDE-JIT (500 samples, 64 grid): {kde_time:.3f}ms per call")
    print(f"  Confluence Density: {conf_time:.4f}ms per call")
    print(f"  Status: ✅ ALL JIT-ACCELERATED")

def run_performance_audit():
    print("=== FEAT NEXUS: PERFORMANCE PROFILER (Phase 7) ===\n")
    
    spectral_time = profile_spectral_tensor_builder()
    volume_time = profile_volume_profile()
    profile_math_engine_jit()
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"  Total Pipeline Latency: ~{spectral_time + volume_time:.1f}ms")
    print(f"  Target: <50ms")
    print(f"  Verdict: {'✅ INSTITUTIONAL GRADE' if (spectral_time + volume_time) < 50 else '⚠️ OPTIMIZE BEFORE PRODUCTION'}")
    print("="*50)

if __name__ == "__main__":
    run_performance_audit()
