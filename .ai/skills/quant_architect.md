# 🗡️ Skill: Quant Architect

## 🎯 Objective
High-performance modification of the `NEXUS_CORE` following the Lead Architect's physics-first mandate.

## 📜 High-Performance Math Standards
1. **Numba Enforcement**: All iterative numerical logic in `math_engine.py` or `structure_engine.py` MUST be decorated with `@njit(cache=True)`.
2. **Type Strictness**: Use `np.float64` for all internal states to prevent precision drift during multifractal calculations.
3. **Avoid Python Objects**: Never use `dict`, `list`, or `class` instances inside a JIT loop. Use NumPy arrays exclusively.

## 🛠️ Optimized Example
```python
@njit(cache=True)
def fast_entropy_integral(probabilities: np.ndarray) -> float:
    entropy = 0.0
    for i in range(probabilities.shape[0]):
        p = probabilities[i]
        if p > 1e-9:
            entropy -= p * np.log2(p)
    return entropy
```

## ✅ Success Criteria
- Benchmarked throughput > 1M iterations/sec.
- Zero JIT compilation warnings during runtime.
