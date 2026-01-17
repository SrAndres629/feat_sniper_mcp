# 🗡️ Skill: Structural Physics (03)

## 🎯 Objective
Govern the mathematical fidelity of the Nexus Core, focusing on fractal geometry and volume profile density.

## 📂 Domain
- `nexus_core/structure_engine.py`
- `nexus_core/math_engine.py`
- `nexus_core/acceleration.py`

## 📜 Specialized Instructions
1. **Fractal Precision**: All CHOCH/BOS detections MUST align with fractal peaks/troughs. No ad-hoc price overrides.
2. **Numba Enforcement**: Any heavy iterative logic (Entropy, Density Integrals) MUST be annotated with `@njit` and avoid Python objects in the loop.
3. **Zone Decay**: Supply/Demand zones must implement time-based decay. Older zones have linearly decreasing weight in the final `struct_score`.

## 🚫 Prohibited Actions
- **No Floating Zones**: Zones must be anchored to historical high/low candles.
- **No Non-Stochastic Slope**: Use linear regression for slope stability, never simple (p2-p1)/dt.

## ✅ Success Criteria
- Market Structure Score reliably identifies Expansion vs Accumulation phases.
- PVP Integrals correctly pinpoint institutional "Reality Walls".
