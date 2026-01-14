# Prompt de Recalibración Dynamica (Hueco #1: Feedback Loop)

**Contexto:** El sistema FEAT Sniper MCP ha sido stabilizado con latencia de 1.35ms y un Monitor de Deriva de Pesos. Sin embargo, carece de un bucle de retroalimentación que "castigue" la confianza de la IA ante una racha de pérdidas en un régimen específico.

**Objetivo:** Desarrollar el `RecalibrationModule`.

**Requerimientos Técnicos:**
1. **Consumo de RAG:** El módulo debe consultar el `TradeMemory` (ChromaDB) para extraer los resultados de los últimos $N$ trades (ej. $N=10$) y su régimen de mercado asociado (`Laminar`, `Turbulento`, `Caos`).
2. **Win-Rate Threshold:** Si el Win Rate de la sesión actual cae por debajo del 40%, el módulo debe inyectar un `confidence_denominator` en el `NeuralInferenceAPI`.
3. **Penalización de Inferencia:**
   - La `p_win` resultante de la inferencia debe ser multiplicada por un factor de castigo (ej. 0.8x) si el sistema está en "Drawdown Mode".
   - Si se detectan 2 pérdidas seguidas en régimen `Turbulento`, el sistema debe activar un **Veto de Seguridad de 2 Horas** para ese régimen específico.
4. **Implementación:** El código debe integrarse como un servicio en `app/services/recalibration.py` y ser llamado por el `NeuralInferenceAPI` antes de emitir un `execute_trade: True`.

**Entrega:**
- `recalibration.py` con lógica de cálculo de Win Rate desde `TradeMemory`.
- Modificación en `inference_api.py` para invocar la validación de confianza.
