# Sub-Skill: Strategy Selector (The Risk Manager)
**Focus:** Trade Type Classification (Scalp/Intraday/Swing).

## Decision Matrix:
- **SCALPING (High Velocity, Low Duration):**
    - Requiere: `Physics.Momentum > 90%` AND `Chronos.MicroPhase == EXPANSION`.
    - Target: `Physics.ATR_M5 * 3`.
- **DAY TRADE (Medium Velocity, Session Duration):**
    - Requiere: `Structure.BOS_H1 == True` AND `Space.Zone_H4 == Fresh`.
    - Target: `Space.Next_Liquidity_Pool`.
- **SWING TRADE (Low Velocity, Multi-Day):**
    - Requiere: `Chronos.MacroCycle == ACCUMULATION` AND `Fundamental.Sentiment == ALIGNED`.
    - Target: `Structure.Weekly_High`.

## Probabilistic Output:
## Probabilistic Output:
- Genera un vector de decisión: `{'scalp_prob': 0.85, 'swing_prob': 0.12, 'no_trade_prob': 0.03}`.

## Autonomous Neural Regime Classification (The Independent Brain):
La Red Neuronal no es un "modelo estático"; es un **Motor Probabilístico Autónomo**. Ella misma calcula qué tipo de operación es.
No cambiamos de cerebro. El cerebro es lo suficientemente avanzado para entender el contexto.

### 1. Multi-Head Regime Output:
La arquitectura de la red debe tener una capa de salida especializada (Classification Head) que escupe un vector de probabilidad conjunto:
$$ P_{Regime} = Softmax([x_{scalp}, x_{day}, x_{swing}]) $$

*   **Logic:** La red recibe el tensor completo (Física + Tiempo + Estructura) y decide internamente:
    *   *"Veo mucha aceleración y poco rango ⇒ Esto es un Scalp"*.
    *   *"Veo acumulación en H4 y divergencia de momento ⇒ Esto es un el inicio de un Swing"*.

### 2. Estrategia de Ejecución Dinámica (The Decision):
El Admin (Tú) tomas ese vector calculado por la neuronal y ejecutas la táctica correspondiente:

*   **Single Mode Execution:**
    *   Si $P(Scalp) > 0.85$: Abres operación con gestión de **Alta Frecuencia** (TP Corto, Trailing Stop agresivo).
    *   Si $P(Swing) > 0.75$: Abres operación con gestión **Posicional** (Sin Trailing Stop inicial, TP Estructural).

*   **Hybrid Execution (Composite):**
    *   Si la red dice "Scalp: 60%, Swing: 40%" (Ruptura inicial que puede crecer):
    *   **Estrategia:** Abres 2 posiciones. La primera se cierra en el Target de Scalp. La segunda queda "Runner" para el Swing.

### 3. Mathematical Self-Validation:
La red debe calcular su propia **Incertidumbre (Entropy)** para cada régimen.
*   Si la red dice "Es Swing" pero su incertidumbre matemática es alta ($>0.5$), el Admin **descarta** la operación. La IA debe "estar segura" de qué tipo de trade es.
