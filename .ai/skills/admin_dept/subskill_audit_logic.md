# Sub-Skill: Logic Auditing (The Triple Filter)
**Focus:** Mathematical, Financial, and Computational Integrity.

## 1. Auditoría Matemática (Probabilística)
*   **Normalización:** Verifica que ningún departamento envíe precios absolutos (ej. 2050.50). Todo debe ser relativo (Distancias ATR, Z-Scores).
*   **Invarianza:** Asegura que las señales sean idénticas en M1, M5 y H1 si la fractaridad se alinea.
*   **Incertidumbre:** Si un modelo neuronal dice "80% Win" pero la "Incertidumbre" (Varianza Monte Carlo) es alta, se degrada la señal.

## 2. Auditoría Financiera (Auction Theory)
*   **Ley de Oferta/Demanda:** ¿Estamos comprando en zonas caras (Premium) o baratas (Discount)?
*   **Liquidez:** ¿La operación provee liquidez o consume liquidez? (Maker vs Taker mindset).
*   **Riesgo de Ruina:** ¿El Stop Loss técnico viola las reglas de Gestión de Capital del banco?

## 3. Auditoría Computacional (Performance)
*   **Latencia:** Ningún cálculo departamental puede bloquear el bucle principal por más de 50ms.
*   **Vectorización:** ¿Se usaron bucles `for` de Python en lugar de operaciones vectoriales `numpy`? (Falta Grave).
*   **Estado:** ¿Los departamentos limpiaron su caché o estamos operando con datos de ayer?
