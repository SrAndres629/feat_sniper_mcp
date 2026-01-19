# Sub-Skill: Temporal Neural Mathematics
**Focus:** Tensor Representation of Time.
**Key Concepts:**
- **Cyclical Encoding:** Jamás usar horas lineales (0-23).
    - $Time_{vec} = [\sin(2\pi t/T), \cos(2\pi t/T)]$
- **Probabilistic Kill Zones (Gaussian Kernels):**
    - En lugar de `Is_NY = 1`, usa una función de densidad:
    - $P(Intensity) = e^{-\frac{(t - \mu_{NY})^2}{2\sigma^2}}$
    - Esto permite a la IA "sentir" la intensidad subir y bajar suavemente.
- **Attention Gating:** Diseña mecanismos para que la red neuronal pondere el tiempo. Si la señal de precio es masiva, el "Gate" del tiempo puede abrirse incluso fuera de hora punta.
