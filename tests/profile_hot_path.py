"""
HOT PATH PROFILER — Market Physics Latency Analysis
====================================================
Identifies the exact bottleneck causing 54ms latency.

Tests individual functions to find the friction point:
1. NumPy array creation from deque
2. ATR calculation
3. Acceleration calculation
4. Regime detection
"""
import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List
import statistics

@dataclass
class ProfileResult:
    function_name: str
    avg_time_us: float  # microseconds
    min_time_us: float
    max_time_us: float
    std_dev_us: float
    calls: int


def profile_function(func, *args, iterations: int = 1000, **kwargs) -> ProfileResult:
    """Profile a function over multiple iterations."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append((end - start) * 1_000_000)  # Convert to microseconds
    
    return ProfileResult(
        function_name=func.__name__ if hasattr(func, '__name__') else str(func),
        avg_time_us=statistics.mean(times),
        min_time_us=min(times),
        max_time_us=max(times),
        std_dev_us=statistics.stdev(times) if len(times) > 1 else 0,
        calls=iterations
    )


def run_hot_path_profiler():
    """Profile each component of the market physics hot path."""
    print("\n" + "=" * 60)
    print("🔬 HOT PATH PROFILER — Market Physics")
    print("=" * 60 + "\n")
    
    # Simulate realistic buffer state (100 elements as in production)
    BUFFER_SIZE = 100
    price_deque = deque([2000.0 + i * 0.1 for i in range(BUFFER_SIZE)], maxlen=BUFFER_SIZE)
    volume_deque = deque([100.0 + i for i in range(BUFFER_SIZE)], maxlen=BUFFER_SIZE)
    timestamp_deque = deque([time.time() + i * 0.1 for i in range(BUFFER_SIZE)], maxlen=BUFFER_SIZE)
    
    results: List[ProfileResult] = []
    
    # ===============================================================
    # TEST 1: Deque to NumPy Array Conversion
    # ===============================================================
    def deque_to_array():
        return np.array(price_deque)
    
    results.append(profile_function(deque_to_array, iterations=1000))
    print(f"1. Deque → NumPy Array: {results[-1].avg_time_us:.2f}μs (max: {results[-1].max_time_us:.2f}μs)")
    
    # ===============================================================
    # TEST 2: Mean/Std Calculation
    # ===============================================================
    prices = np.array(price_deque)
    
    def calc_mean_std():
        mean = np.mean(prices)
        std = np.std(prices)
        return mean, std
    
    results.append(profile_function(calc_mean_std, iterations=1000))
    print(f"2. Mean + Std Calculation: {results[-1].avg_time_us:.2f}μs (max: {results[-1].max_time_us:.2f}μs)")
    
    # ===============================================================
    # TEST 3: ATR Calculation (np.diff + np.abs + np.mean)
    # ===============================================================
    def calc_atr():
        return np.mean(np.abs(np.diff(prices)))
    
    results.append(profile_function(calc_atr, iterations=1000))
    print(f"3. ATR Calculation: {results[-1].avg_time_us:.2f}μs (max: {results[-1].max_time_us:.2f}μs)")
    
    # ===============================================================
    # TEST 4: Velocity Calculation
    # ===============================================================
    times_arr = np.array(timestamp_deque)
    
    def calc_velocity():
        delta_p = prices[-1] - prices[-2]
        delta_t = max(times_arr[-1] - times_arr[-2], 0.001)
        return delta_p / delta_t
    
    results.append(profile_function(calc_velocity, iterations=1000))
    print(f"4. Velocity Calculation: {results[-1].avg_time_us:.2f}μs (max: {results[-1].max_time_us:.2f}μs)")
    
    # ===============================================================
    # TEST 5: Full Acceleration (Vol Intensity × Norm Velocity)
    # ===============================================================
    vols = np.array(volume_deque)
    
    def calc_full_acceleration():
        # Mean/std of volumes
        mean_vol = np.mean(vols)
        vol_intensity = vols[-1] / mean_vol
        
        # ATR
        atr = np.mean(np.abs(np.diff(prices)))
        
        # Velocity
        delta_p = prices[-1] - prices[-2]
        delta_t = 0.1  # Simulated
        raw_velocity = delta_p / delta_t
        
        # Normalized velocity
        norm_velocity = raw_velocity / atr if atr > 0 else 0
        
        # Acceleration
        return vol_intensity * abs(norm_velocity)
    
    results.append(profile_function(calc_full_acceleration, iterations=1000))
    print(f"5. Full Acceleration: {results[-1].avg_time_us:.2f}μs (max: {results[-1].max_time_us:.2f}μs)")
    
    # ===============================================================
    # TEST 6: Threshold Calculation (Mean + 2*Std of acceleration history)
    # ===============================================================
    accel_history = deque([0.5 + i * 0.01 for i in range(100)], maxlen=100)
    
    def calc_threshold():
        acc_array = np.array(accel_history)
        acc_mean = np.mean(acc_array)
        acc_std = np.std(acc_array)
        return acc_mean + (2.0 * acc_std)
    
    results.append(profile_function(calc_threshold, iterations=1000))
    print(f"6. Threshold (μ+2σ): {results[-1].avg_time_us:.2f}μs (max: {results[-1].max_time_us:.2f}μs)")
    
    # ===============================================================
    # TEST 7: FULL PIPELINE (as in production)
    # ===============================================================
    def full_pipeline():
        # Convert all deques
        p = np.array(price_deque)
        v = np.array(volume_deque)
        t = np.array(timestamp_deque)
        
        # Stats
        mean_vol = np.mean(v)
        vol_intensity = v[-1] / mean_vol
        vol_z = (v[-1] - mean_vol) / max(np.std(v), 0.0001)
        
        # ATR
        atr = np.mean(np.abs(np.diff(p)))
        
        # Velocity
        delta_p = p[-1] - p[-2]
        delta_t = max(t[-1] - t[-2], 0.001)
        velocity = np.clip(delta_p / delta_t, -1e6, 1e6)
        
        # Normalized acceleration
        norm_vel = velocity / atr if atr > 0 else 0
        accel = vol_intensity * abs(norm_vel)
        
        # Threshold
        acc_arr = np.array(accel_history)
        threshold = np.mean(acc_arr) + 2 * np.std(acc_arr)
        
        return accel > threshold
    
    results.append(profile_function(full_pipeline, iterations=1000))
    print(f"7. FULL PIPELINE: {results[-1].avg_time_us:.2f}μs (max: {results[-1].max_time_us:.2f}μs)")
    
    # ===============================================================
    # ANALYSIS
    # ===============================================================
    print("\n" + "=" * 60)
    print("📊 ANÁLISIS DEL HOT PATH")
    print("=" * 60)
    
    # Sort by avg time
    sorted_results = sorted(results, key=lambda x: x.avg_time_us, reverse=True)
    
    total_time = sum(r.avg_time_us for r in results[:6])  # Exclude full pipeline
    print(f"\n⏱️ Tiempo Total (sin pipeline): {total_time:.2f}μs = {total_time/1000:.3f}ms")
    print(f"⏱️ Full Pipeline: {results[-1].avg_time_us:.2f}μs = {results[-1].avg_time_us/1000:.3f}ms")
    
    print("\n🔴 TOP BOTTLENECKS:")
    for i, r in enumerate(sorted_results[:3], 1):
        percentage = (r.avg_time_us / results[-1].avg_time_us) * 100
        print(f"   {i}. {r.function_name}: {r.avg_time_us:.2f}μs ({percentage:.1f}% del pipeline)")
    
    # Identify the culprit
    print("\n" + "=" * 60)
    if sorted_results[0].function_name == "deque_to_array":
        print("🎯 CULPABLE: Conversión Deque → NumPy Array")
        print("   FIX: Pre-allocate NumPy arrays instead of deques")
    elif "threshold" in sorted_results[0].function_name.lower():
        print("🎯 CULPABLE: Cálculo de Threshold (μ+2σ)")
        print("   FIX: Use running statistics (Welford algorithm)")
    else:
        print(f"🎯 CULPABLE: {sorted_results[0].function_name}")
        print("   FIX: Consider Numba JIT compilation")
    
    print("=" * 60 + "\n")
    
    return results


if __name__ == "__main__":
    run_hot_path_profiler()
