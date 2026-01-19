# Sub-Skill: Computational Mathematics
**Focus:** Numerical Stability & Optimization.
**Rules:**
- **Vectorization:** Prohibido usar bucles `for` en Python para datos de mercado. Usa `torch.tensor` o `numpy.array`.
- **Log-Sum-Exp Trick:** Para evitar underflow numérico en cálculos de probabilidad.
- **Epsilon Injection:** Siempre sumar `+ 1e-8` en divisiones para evitar `ZeroDivisionError`.
