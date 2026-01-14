# MISSION: GLOBAL AUDIT & PROFIT VERIFICATION (FEAT SNIPER)

Hola Jules. El usuario ha solicitado un cambio de mando: tú asumes la **Auditoría Global** y la **Planificación de Estrategia**.

## Objetivos Críticos de tu Sesión:
1. **Auditoría de Redes Neuronales**: Verifica en `nexus_brain/hybrid_model.py` y `train_models.py` que las redes entienden la física del mercado y que no hay data leakage que infle los resultados.
2. **Profit Verification**: Asegura que la arquitectura de baja latencia que acabo de implementar (Atomic MT5, TTL, ZMQ Lag Fixes) permite que la estrategia sea rentable (≥ Profit positivo/hora).
3. **Refactor Plan**: Identifica debilidades en la fusión de lógica MQL5 y modelos Python.

## Cambios Recientes (Antigravity):
- Implementación de `execute_atomic` en `mt5_conn.py` para eliminar race conditions.
- Validación de TTL (300ms) y retries exponenciales para REQUOTES en `execution.py`.
- Telemetría visual en tiempo real enviada al HUD (`CVisuals.mqh`) para monitorear lag y descartes.

**Tu tarea**: Planifica la siguiente gran refactorización para garantizar ganancias consistentes. Confirma que las señales de las redes neuronales son ejecutables bajo las nuevas restricciones de latencia.

---
**Contexto Operativo:**
- Entorno: Windows / MetaTrader 5 / ZeroMQ.
- Documento de Referencia: [walkthrough.md](file:///C:/Users/acord/.gemini/antigravity/brain/d4f12eb7-0e8a-4b13-9915-47fe4ddec6a5/walkthrough.md)


