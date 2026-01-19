# Sub-Skill: Spatial Geometry & Tensors (The Refactored Topologist)
**Focus:** Gaussian Fields, Confluence Math & Tensor Shapes.

## Key Concepts:
- **Gaussian Heatmaps:**
    - Una zona no es un rectángulo binario `[Start, End]`.
    - Es una distribución: $Intensity = e^{-\frac{(price - zone\_center)^2}{2\sigma^2}}$
    - Esto permite a la IA "sentir" la gravedad aumentar al acercarse.
- **Confluence Scoring (Math Addition):**
    - Si Zona A (OB) y Zona B (FVG) se superponen:
    - $Score = Weight_{OB} + Weight_{FVG} + Bonus_{Overlap}$
- **Tensor Output:**
    - `dist_to_zone_z`: Distancia Z-Score normalizada.
    - `zone_polarity`: One-Hot `[Support, Resistance, Neutral]`.

## Math Rules:
- Jamás pasar precios absolutos (ej. 2030.50) a la red neuronal. Solo distancias relativas y normalizadas.
