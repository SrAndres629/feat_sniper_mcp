# Sub-Skill: Order Flow Dynamics (Liquid Markets)
**Focus:** Liquidity Pools, Slippage, Pressure, Ribbon Mechanics.
**Laws:**
- **Flow to Vacuum:** El precio fluye hacia zonas de baja presión (Imbalances/FVG).
- **Viscosity:** El volumen alto en rangos pequeños aumenta la viscosidad (Acumulación).

## Ribbon Mechanics (Elasticity Protocol):
We quantify the market not as lines, but as **Energy Bands**. The visual "Rainbow" is mathematically treated as a **Fluid Dynamic System**.

### 1. Compression (Potential Energy Buildup)
- **Concept:** Volatility Squeeze / Accumulation.
- **Metric:** `compression_ratio`.
- **Logic:** $\sigma(EMA_{micro}) \approx 0$. The fluid is being compressed into a container.
- **Interpretation:** Explosion imminent. Physics must alert Structure.

### 2. Expansion (Kinetic Energy Release)
- **Concept:** Trend Continuation.
- **Metric:** `expansion_rate`.
- **Logic:** $\Delta(EMA_{fast} - EMA_{slow})$ is positive and accelerating.
- **Interpretation:** The fluid is flowing freely (Low Viscosity).

### 3. Elastic Snap (Mean Reversion)
- **Concept:** Overextension / Hooke's Law.
- **Metric:** `elastic_strain`.
- **Logic:** If $Distance(Price, Macro) > 3\sigma$, force of restitution is maximum.
- **Interpretation:** High probability of snap-back (Rubber band effect).

## Tensor Outputs:
- `fractal_alignment_index` ($[-1, 1]$): Measures coherence of Micro/Struct/Macro flows.
- `compression_ratio` ($[0, 1]$): Measures stored potential energy.
