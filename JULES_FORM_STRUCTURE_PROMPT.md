### 📜 PROMPT PARA JULES: EMBEDDING DE "FORMA & ESTRUCTURA"

> **Role:** Senior Quant Architect & Lead Machine Learning Engineer.
> **Task:** Refinar la arquitectura de la red neuronal **LSTM (1,1,4)** y el pipeline de datos de **FEAT Sniper** para integrar los conceptos de **Forma (Form)** y **Estructura (Structure)** en la toma de decisiones.
> **Contexto Técnico:**
> Actualmente, nuestra red opera sobre un vector físico 4D. Necesitamos que la IA no solo reaccione a la aceleración, sino que entienda el **Contexto Estructural** donde ocurre esa aceleración.
> **Objetivos de Diseño:**
> 1. **Definición de Forma (Geometry):** Implementa una capa de pre-procesamiento que detecte **Fair Value Gaps (FVG)** y **Order Blocks (OB)**. La red debe recibir la "Distancia Relativa" al nivel de Forma más cercano como una característica normalizada.
> 2. **Definición de Estructura (Market Flow):** Integra la detección de **Break of Structure (BOS)** y **Change of Character (CHoCH)**. Necesitamos que la red neuronal tenga un sesgo (Bias) direccional basado en si la estructura mayor es Alcista o Bajista.
> 3. **Refactor de `hybrid_model.py` & `InferenceAPI`:**
> * Propón una expansión del vector de entrada de 4D a **6D** o **8D**, incluyendo: `Dist_to_OB` y `Structure_Bias` (-1, 0, 1).
> * Asegura que el cálculo de estos nuevos inputs mantenga la complejidad O(1) o O(log n) para no degradar nuestra latencia de **1.35ms**.
> 
> 4. **Lógica Simbólica (Consenso):** La "Forma" debe actuar como un filtro de probabilidad. Si la "Aceleración" (IA) apunta a una compra, pero el precio está chocando contra una "Estructura" de resistencia mayor, la confianza de la señal debe reducirse automáticamente.
> 
> **Entregable:**
> Un blueprint detallado de cómo modificar los archivos `app/skills/market_physics.py` e `inference_api.py` para inyectar estos conceptos de forma cohesiva, sin romper la persistencia de los `scaler_stats`.
> **Restricción:** No sacrifiques la interpretabilidad. Queremos saber exactamente por qué la red decidió que una "Forma" específica invalidó una señal de aceleración.
