# 🗡️ Skill: Math Optimization (.ai/skills/math_optimization.md)

## 🎯 Objective
Ensure maximum numerical throughput and zero-latency execution for physics-based calculations.

## 📜 Numba (@njit) Style Guide
1. **Cache Enforcement**: Always use `cache=True` to prevent overhead during system restarts.
2. **Pure Functions**: JIT functions must be pure. No I/O, no global mutable state, no Python objects (`dict`, `list`).
3. **NumPy Vectorization**: Prioritize NumPy slicing over manual loops where possible, but use Numba loops if conditional complexity is high.
4. **Type Casting**: Explicitly cast indices to `np.int64` and values to `np.float64`.

## 🛠️ Comparison Example (Lead Architect Standard)
```python
# [VETOED] Non-optimized, slow loop
def slow_avg(arr):
    return sum(arr) / len(arr)

# [APPROVED] Senior JIT implementation
@njit(cache=True)
def fast_avg(arr: np.ndarray) -> float:
    return np.mean(arr)
```

## ✅ Success Criteria
- Physics layer centroids calculated in < 1ms.
- ZERO GIL-poisoning during high-frequency ticks.
