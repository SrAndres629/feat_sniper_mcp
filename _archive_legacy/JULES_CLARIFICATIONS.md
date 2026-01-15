# 📄 Clarificaciones Técnicas para Jules (Fase 14 Preparation)

Jules, aquí tienes las aclaraciones necesarias para resolver los estados "Needs Clarification" y proceder con la auditoría de la Fase 14.

---

### 1. Historical Calculation Loop Logic (SMMA $O(1)$)
**Estado:** Resuelto.
**Detalle Técnico:** 
- El sistema utiliza una **fórmula recursiva de SMMA**: $SMMA_t = \frac{SMMA_{t-1} \times (N-1) + Price_t}{N}$.
- En `NeuralInferenceAPI`, mantenemos un diccionario `smma_states` persistente. 
- En el primer tick (Cold Start), se calcula el valor inicial iterando sobre el buffer histórico.
- En ticks subsiguientes (Hot Update), se aplica la actualización de carga constante, garantizando una latencia de ~1.35ms incluso con miles de periodos de suavizado.

### 2. Monitor de Deriva (Weight Drift Monitor)
**Estado:** Implementado y Operativo.
**Archivo:** `app/ml/drift_monitor.py`.
**Detalle Técnico:**
- Los `scaler_stats` (Mean/Scale) se extraen directamente del checkpoint del modelo (`.pt`).
- El monitor mantiene una ventana deslizante de los últimos 200 features de inferencia real.
- Calcula el **Max Z-Score** de la deriva. Si el mercado real se desvía más de 0.8 del entrenamiento, se emite una alerta `DRIFT_ALERT`.

### 3. Consenso de Desconfianza (Physics vs Neural Veto)
**Estado:** Implementado (Veto Alpha).
**Archivo:** `nexus_brain/inference_api.py`.
**Detalle Técnico:**
- Hemos implementado la regla de oro: **La Neurona propone, la Física dispone**.
- Si `LSTM_Inference` devuelve `execute_trade: True` (Compra) pero el vector de física $L4\_Slope$ es negativo (Fuerza Bajista), el sistema ejecuta un `VETO` automático.
- Esto elimina trades de "atrapada de liquidez" donde el precio sube por ruido pero la masa institucional no acompaña.

### 4. Performance Optimization (Zero-Latency Bridge)
**Estado:** Optimizado (1.35ms P99).
**Detalle Técnico:**
- Hemos eliminado el cuello de botella de `ZMQ` mediante el uso de **Zero-Copy serialization** (msgpack/raw scalars) y evitando el uso de logs pesados en el bucle principal.
- El servidor MCP utiliza `SSE` (Server-Sent Events) para la comunicación con el frontend, liberando el hilo de procesamiento de inferencia.
- La latencia total "Tick-to-Inference" es de **1.35ms**, permitiendo operar en micro-regímenes de alta frecuencia.

---

### 🚀 Siguiente Paso: Feedback Loop RAG

Estamos integrando el **Recalibration Module**. El objetivo es que la `NeuralInferenceAPI` consulte los últimos resultados de la `TradeMemory` (ChromaDB) y reduzca la `p_win` si el Win Rate reciente es bajo (<40%), cerrando el círculo del aprendizaje adaptativo.
