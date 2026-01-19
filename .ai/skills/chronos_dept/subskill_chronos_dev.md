# Sub-Skill: Chronos Development & Optimization
**Focus:** High-Performance Python & Timezone Safety.
**Rules:**
- **Timezone Strictness:** Todo objeto `datetime` debe tener `tzinfo` (UTC o America/New_York). Prohibido usar `datetime.now()` sin zona.
- **State Machine Efficiency:** Implementar la detección de fases (`get_micro_phase`) usando tablas hash o diccionarios optimizados, no cadenas infinitas de `if-elif-else`.
- **Vectorization:** Los cálculos de ciclos deben ser vectorizados para procesar miles de velas de backtesting en milisegundos.
