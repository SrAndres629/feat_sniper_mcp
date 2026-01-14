### 📜 PROMPT PARA JULES: ARQUITECTURA TEMPORAL Y CRONOMETRÍA DE ALFA (FEAT)

> **Role:** Senior Quant Strategist & Market Microstructure Expert.
> **Task:** Implementar el pilar del **Tiempo (Time)** en el motor de decisión de **FEAT Sniper**. Debes dotar a la red neuronal **LSTM (1,1,4)** y a la **FEAT Chain** de la capacidad de entender la dinámica temporal del flujo de órdenes y la estacionalidad intradía.
> **Contexto Técnico:**
> El Tiempo es la dimensión que valida o invalida las otras tres (F, E, A). Una señal de Aceleración excelente en un Espacio libre no tiene el mismo valor al cierre de Nueva York que en la apertura de Londres.
> **Objetivos de Diseño para Jules:**
> 1. **Signal Time-to-Live (TTL) & Decay:**
> * Implementa una función de **Decaimiento de Confianza**: La validez de una señal de la IA debe reducirse linealmente o exponencialmente según el tiempo transcurrido (medido en milisegundos y ticks) desde su generación.
> * Si el precio no alcanza el primer objetivo en el tiempo  esperado basado en la volatilidad actual, el sistema debe ejecutar un **'Time-Based Exit'**.
> 
> 2. **Mapeo de Killzones y Ciclos de Sesión:**
> * Define las ventanas de alta probabilidad (**Killzones**): Londres, Nueva York y el "Overlap".
> * El vector de entrada debe incluir un `Session_Intensity_Score` que normalice la actividad esperada. La IA debe ser más escéptica ante movimientos rápidos en horas de baja liquidez (Asia).
> 
> 3. **Análisis de Velocidad Relativa (Time-Relative Velocity):**
> * Crea una métrica que compare la velocidad actual del precio con la velocidad promedio de la misma hora en los últimos 20 días.
> * Esto ayudará a detectar **anomalías temporales** que suelen preceder a los movimientos institucionales.
> 
> 4. **Filtro de Impacto de Noticias (Temporal Proximity):**
> * Diseña un hook para que el sistema reduzca el riesgo o entre en **'Safety Mode'** en la proximidad de eventos macroeconómicos (±5 minutos de noticias de alto impacto).
> * El tiempo de "congelación" debe ser dinámico basado en cuánto tarda el mercado en recuperar el régimen **Laminar**.
> 
> 5. **Optimización de Latencia:**
> * Todos los cálculos temporales deben basarse en el `Decision_TS` del `mcp_server` para garantizar que no haya desfases entre la inferencia y la ejecución en MT5.
> 
> **Entregable:**
> Un esquema de actualización para `app/skills/history.py` y `trade_mgmt.py` que incorpore estas reglas de tiempo. Queremos que el **TradeManager** sea consciente de que el tiempo es un recurso finito y que el Alpha tiene fecha de caducidad.
> **Restricción:** Mantener la coherencia con el **Protocolo POM**. La lógica temporal debe ser lo suficientemente ligera para no añadir más de 0.1ms a nuestra latencia actual.
