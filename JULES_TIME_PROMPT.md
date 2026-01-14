### 🕰️ PROMPT PARA JULES: IMPLEMENTACIÓN DEL "TIEMPO" (MARKET TIME)

> **Role:** Senior Quant Architect & Temporal Data Scientist.
> **Task:** Diseñar e integrar el cuarto y último pilar del análisis FEAT: **Tiempo (Time)**. El objetivo es que la IA no solo sepa "qué", "dónde" y "cuánto", sino **"cuándo"** una señal tiene mayor probabilidad estadística de expansión.
> 
> **Contexto Técnico:**
> El tiempo en el trading institucional es cíclico y se rige por la liquidez de sesión (Londres/NY) y los horarios de los bancos centrales. Una señal de aceleración excelente es irrelevante si ocurre en los últimos 2 minutos de la sesión de Londres o durante el "NFP blackout".
> 
> **Objetivos de Diseño para Jules:**
> 1. **Detección de 'Session Horizon':**
> * Calcula la proximidad al cierre de la sesión actual. Si faltan < 30 minutos para el cierre de Londres o NY, activa un `Time_Decay_Filter` que aumente la exigencia de la `p_win`.
> 
> 2. **Inercia Cronológica:**
> * Define el concepto de **'Golden Hours'** (aperturas y solapamientos). Durante estas horas, el factor de confianza de la IA debe recibir un bono multiplicador (ej. 1.1x) debido a la inercia institucional.
> 
> 3. **Veto de Inactividad Temporal:**
> * Si el sistema detecta que el precio ha estado plano por más de X periodos (Time Compression), la señal de entrada debe ser degradada hasta que ocurra un evento de expansión.
> 
> 4. **Input Vector: `Time_Entropy`:**
> * Añade una feature que represente la "madurez" del movimiento actual: ¿Cuánto tiempo ha pasado desde el último pico de aceleración física?
> 
> **Entregable:**
> Refactor para `app/skills/calendar.py` y `nexus_brain/inference_api.py`. El sistema debe ser capaz de decir: *"Tengo Física, tengo Espacio, tengo Forma... pero NO tengo Tiempo (Cierre de sesión inminente). Abortando entrada"*.
