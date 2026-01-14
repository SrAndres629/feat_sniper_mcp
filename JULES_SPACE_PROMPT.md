### 📜 PROMPT PARA JULES: IMPLEMENTACIÓN DEL "ESPACIO" (MARKET SPACE)

> **Role:** Senior Quant Architect & Volatility Specialist.
> **Task:** Integrar el concepto de **Espacio (Space)** en el motor de inferencia y la lógica de gestión de riesgo de **FEAT Sniper**. El objetivo es que la red neuronal **LSTM (1,1,4)** evalúe la viabilidad de una señal basándose en el "oxígeno" disponible en el gráfico.
> **Contexto Técnico:**
> El Espacio define la relación entre el **Rango (ATR)**, el **Spread** y la **Distancia a la Liquidez Opuesta**. Una señal sin espacio es un trade de baja probabilidad, incluso si la aceleración física es correcta.
> **Objetivos de Diseño para Jules:**
> 1. **Cálculo del 'Trading Runway' (Pista de Aterrizaje):**
> * Implementa una función que calcule la distancia entre el precio actual y el siguiente nivel de **Forma** (OB/FVG) o **Estructura** (High/Low de sesión).
> * Define el **Espacio Neto**: (Distancia al Objetivo - Spread). Si el Espacio es < (ATR * 0.5), la señal debe ser degradada.
> 
> 2. **Normalización de Volatilidad (ATR-Relative):**
> * El vector de física debe entender el espacio en términos de ATR. No es lo mismo 10 pips en una hora muerta que 10 pips en la apertura de Nueva York.
> * Crea una métrica de **Eficiencia de Espacio**: ¿Cuánto se ha desplazado el precio respecto al volumen inyectado?
> 
> 3. **Filtro de Fricción (Spread/Liquidity):**
> * El "Espacio" no es gratis. Integra el costo del Spread en tiempo real dentro de la probabilidad de éxito. Si el Spread consume más del 20% del Espacio esperado hacia el primer Take Profit, la IA debe emitir un 'Low Space Warning'.
> 
> 4. **Expansión del Input Vector:**
> * Añade la feature `Space_Ratio`: (Espacio Neto / ATR).
> * Asegura que el cálculo sea O(1) usando los datos ya cacheados en el `mcp_server`.
> 
> 
> **Entregable:**
> Un esquema de refactorización para `app/skills/execution.py` y `app/ml/normalization.py`. Queremos que el **TradeManager** bloquee órdenes si el "Espacio" es insuficiente, protegiendo el capital de entradas en rangos comprimidos o "choppy markets".
> **Restricción:** El cálculo del Espacio debe ser dinámico. Si el ATR se expande (Régimen Turbulento), el Espacio requerido para operar debe expandirse proporcionalmente.
