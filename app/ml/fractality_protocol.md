# Protocolo de Inteligencia Multifractal (MIP)

## 🏛️ Filosofía del Diseño
El mercado no es un proceso lineal simplista; es una estructura **multifractal** donde los patrones se repiten en diferentes escalas de tiempo, pero con diferentes niveles de energía y significancia. El Protocolo MIP permite que NEXUS entienda esta jerarquía.

## 🌀 Conceptos de Física de Mercado

### 1. El Exponente de Hurst (H)
Utilizamos el Exponente de Hurst para clasificar el régimen del timeframe:
- **H > 0.55 (Persistente):** El activo está en tendencia. Priorizamos modelos de **Breakout**.
- **H < 0.45 (Anti-persistente):** El activo es regresivo a la media. Priorizamos modelos de **Reversión**.
- **H ≈ 0.50 (Browniano):** Ruido aleatorio. El sistema aumenta el `Confidence Threshold`.

### 2. Dimensión Fractal (D)
Estimamos la complejidad del precio ($D = 2 - H$). Una alta dimensión fractal indica un mercado errático y poco predecible donde se reduce el tamaño de la posición.

## 🏗️ Jerarquía Temporal (The Weighting Tree)

| Timeframe | Rol Estratégico | Peso Base | Atributo Principal |
|-----------|------------------|-----------|--------------------|
| **D1 / W1** | Global Bias | 15% | Ciclo Macro y Estructura |
| **H4 / H1** | Strategist | 45% | Zonas de Oferta/Demanda y Tendencia |
| **M15 / M30**| Context | 20% | Momentum Intermedio |
| **M5 / M1** | Sniper | 20% | Timing de entrada y Micro-volatilidad |

## 🛡️ Capa de Fusión (The Fusion Gate)

La decisión final no es un promedio simple. MIP aplica una **Puerta de Lógica Jerárquica**:

1. **Alineación Fractal:** Para una compra de alta confianza, el Bias de H4 debe ser `>= 0.5`.
2. **Veto Estructural:** Si D1 indica una tendencia bajista fuerte, cualquier señal de compra en M1 es penalizada en un 50% de su peso.
3. **Ajuste por Volatilidad:** El Take Profit y Stop Loss no son estáticos; se calculan como un múltiplo del ATR específico del timeframe dominante.

## 🚀 Implementación Técnica
- **Dataset:** Almacenado en `market_data` bajo el esquema institucional v6.0.
- **Inferencia:** Orquestada por `ml_engine.py` mediante el `MultiInputEnsemble`.
- **Validación:** El `why_vector` describe qué timeframe fue el driver del trade.
