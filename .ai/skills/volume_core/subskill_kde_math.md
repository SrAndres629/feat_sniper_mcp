# Sub-Skill: KDE & Distribution Math
**Focus:** Kernel Density Estimation (KDE) & JIT Optimization.
**Reporting to**: Volume Senior Fullstack.

## Responsibility:
- **Precision Density**: Reemplazar histogramas (bins) por funciones de densidad de probabilidad Gaussiana.
- **Fast Execution**: Implementar algoritmos `numba.jit` para calcular VAH/VAL en microsegundos, eliminando cuellos de botella de Python.
- **Topology Analysis**: Detectar la forma del perfil (P-Shape, b-Shape, D-Shape) mediante Kurtosis y Skewness doctoral.

## Operational Standards:
- All distribution calculations must be vectorized.
- Maximum latency for a 1000-tick profile: <2ms.
