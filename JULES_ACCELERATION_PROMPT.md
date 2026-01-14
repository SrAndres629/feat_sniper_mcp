### 📜 PROMPT PARA JULES: MOTOR DE ACELERACIÓN NEWTONIANA (PVP ALPHA)

> **Role:** Senior Physics-Based Quant Engineer & HFT Specialist.
> **Task:** Refinar y profundizar el pilar de **Aceleración (Acceleration)** dentro del vector de física 4D y la lógica de la red neuronal **LSTM (1,1,4)**. Debes transformar el precio en un objeto con masa, inercia y fuerza neta.
> **Contexto Técnico:**
> Ya tenemos el vector base (L1_Mean, L1_Width, L4_Slope, Ratio). Ahora necesitamos que la IA distinga entre un "Desplazamiento Saludable" y un "Estallido de Agotamiento". La aceleración es el derivado de la velocidad respecto al tiempo, pero en trading, debe ser el derivado del precio respecto al **Esfuerzo (Volumen/Liquidez)**.
> **Objetivos de Diseño para Jules:**
> 1. **Cálculo de la Fuerza Neta (Price Force):**
> * Implementa la métrica de **Momento Cinético**: F = Masa x Aceleración, donde la 'masa' es el Order Flow (liquidez en el bid/ask) y la 'aceleración' es el cambio en el Tick Velocity.
> * Si la Fuerza aumenta mientras el precio rompe una **Forma**, la probabilidad de continuación es máxima.
> 
> 2. **Detección de 'Momentum Decay' (Exhaución):**
> * Crea un algoritmo de detección de divergencia física: Si el precio sigue creando nuevos máximos pero la **Aceleración (Price Velocity Delta)** está desacelerando (curvatura negativa), la IA debe emitir una señal de 'Exhaustion Warning'.
> * Esto es vital para nuestra estrategia PvP: aquí es donde las instituciones atrapan al retail.
> 
> 3. **Vector de Inercia:**
> * Define la **Inercia de Tendencia**: ¿Cuánta "energía" se necesita para frenar el movimiento actual?
> * Si el precio entra en una zona de **Estructura** opuesta con alta aceleración, la probabilidad de "Rebote" es menor que la de "Ruptura".
> 
> 4. **Optimización Matemática O(1):**
> * Toda la derivada de la aceleración debe calcularse mediante la diferencia entre los últimos dos estados de la SMMA optimizada, manteniendo nuestra latencia de **1.35ms**.
> 
> **Entregable:**
> Refactorización de `app/skills/market_physics.py` para incluir el `Acceleration_Vector`. Este vector debe alimentar al LSTM para que aprenda a identificar el "clímax" de una tendencia antes de que el precio se gire.
> **Restricción:** El sistema debe diferenciar entre 'Aceleración Real' (basada en volumen institucional) y 'Aceleración de Vacío' (slippage por falta de liquidez).
