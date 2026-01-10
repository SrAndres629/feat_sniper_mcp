# FEAT Sniper MCP - Análisis Técnico Institucional

## 1. Arquitectura del Sistema

### 1.1 Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FEAT SNIPER NEXUS                                 │
├─────────────────────┬───────────────────────┬───────────────────────────────┤
│    MT5 (WINDOWS)    │    DOCKER (PYTHON)    │        OUTPUTS                │
├─────────────────────┼───────────────────────┼───────────────────────────────┤
│                     │                       │                               │
│  ┌───────────────┐  │  ┌─────────────────┐  │  ┌─────────────────────────┐  │
│  │  UnifiedModel │  │  │   MCP Server    │  │  │  Señales BUY/SELL/WAIT  │  │
│  │    .mq5       │──┼──│   (FastMCP)     │──┼──│  Confianza 0-100%       │  │
│  └───────────────┘  │  └─────────────────┘  │  └─────────────────────────┘  │
│         │           │           │           │                               │
│  ┌──────┴──────┐    │  ┌────────┴────────┐  │  ┌─────────────────────────┐  │
│  │   CFEAT     │    │  │   RAG Memory    │  │  │  FSM State:             │  │
│  │   CFSM      │────┼──│   (ChromaDB)    │──┼──│  ACCUMULATION/EXPANSION │  │
│  │   CLiquidity│    │  │                 │  │  │  DISTRIBUTION/MANIP     │  │
│  └─────────────┘    │  └─────────────────┘  │  └─────────────────────────┘  │
│         │           │           │           │                               │
│  ┌──────┴──────┐    │  ┌────────┴────────┐  │  ┌─────────────────────────┐  │
│  │   CEMAs     │    │  │   ML Skills     │  │  │  Zonas Institucionales: │  │
│  │ (Multifrac) │    │  │   (Python)      │  │  │  FVG, OB, BOS, CHoCH    │  │
│  └─────────────┘    │  └─────────────────┘  │  └─────────────────────────┘  │
│                     │                       │                               │
└─────────────────────┴───────────────────────┴───────────────────────────────┘
```

### 1.2 Módulos Core (MQL5)

| Módulo | Líneas | Función |
|--------|--------|---------|
| `CFEAT.mqh` | 483 | Framework FEAT: Engineer → Tactician → Sniper pipeline |
| `CFSM.mqh` | 171 | Wyckoff FSM: 6 estados de mercado con lógica multifractal |
| `CLiquidity.mqh` | 453 | Detección de zonas institucionales (FVG, OB, BOS, CHoCH) |
| `CEMAs.mqh` | 600+ | Sistema de EMAs multifractal (Micro/Oper/Macro/Bias) |
| `CInterop.mqh` | 200+ | Bridge ZMQ para comunicación con Python |

### 1.3 Skills Python

| Skill | Función | ML Actual |
|-------|---------|-----------|
| `ml_sniper.py` | Clasificador de setups | ❌ Heurístico (reglas) |
| `advanced_analytics.py` | Shadow testing, sentiment | ❌ Simulado |
| `market.py` | Datos de mercado, volatilidad | N/A |
| `liquidity.py` | Wrapper de CLiquidity | N/A |
| `execution.py` | Ejecución de órdenes | N/A |

---

## 2. Análisis de Machine Learning Actual

### 2.1 Estado Actual: Rule-Based System

```python
# ml_sniper.py - Líneas 33-37
score = 0
if features["vola_status"]: score += 0.3
if features["rsi_oversold"] or features["rsi_overbought"]: score += 0.4
if features["is_liquid"]: score += 0.3
```

**Diagnóstico**: El sistema actual **NO usa ML real**. Es un sistema de reglas heurísticas que simula la lógica de un RandomForest pero con pesos fijos.

### 2.2 Librerías ML Disponibles (pero no usadas)

| Librería | Instalada | Usada |
|----------|-----------|-------|
| `numpy` | ✅ | ✅ (cálculos básicos) |
| `pandas` | ✅ | ✅ (estructuras de datos) |
| `scikit-learn` | ✅ | ❌ No usada |
| `pytorch` | ✅ | ❌ No usada |
| `sentence-transformers` | ✅ | ✅ (RAG embeddings) |

### 2.3 Gaps Identificados

| Aspecto | Estado | Nivel |
|---------|--------|-------|
| Modelos entrenados | ❌ No existen | CRÍTICO |
| Feature engineering | ⚠️ Manual básico | MEDIO |
| Cross-validation | ❌ No implementado | CRÍTICO |
| Backtesting ML | ⚠️ Shadow test simulado | MEDIO |
| Ensemble methods | ❌ No existen | ALTO |
| RNN/LSTM temporales | ❌ No existen | ALTO |

---

## 3. Propuestas de Mejora ML

### 3.1 Fase 1: ML Supervisado Básico

**Objetivo**: Reemplazar reglas heurísticas por modelos entrenados.

```python
# Propuesta: ml_models.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

class FEATClassifier:
    """Clasificador entrenado para estados FEAT."""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100)
        
    def fit(self, X_features, y_labels):
        tscv = TimeSeriesSplit(n_splits=5)
        # Implementar walk-forward validation
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
```

**Impacto**: +15-25% precisión vs reglas actuales.

---

### 3.2 Fase 2: Deep Learning Temporal

**Objetivo**: Capturar patrones secuenciales.

```python
# Propuesta: lstm_predictor.py
import torch.nn as nn

class FEATSequenceModel(nn.Module):
    """LSTM para predicción de estados de mercado."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 6)  # 6 estados FSM
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
```

**Impacto**: Captura dependencias temporales que reglas no pueden.

---

### 3.3 Fase 3: Reinforcement Learning

**Objetivo**: Optimización autónoma de decisiones de trading.

| Componente | Implementación |
|------------|----------------|
| Environment | Gym wrapper sobre MT5 data |
| Agent | PPO/SAC con action space discreto |
| Reward | Sharpe ratio + drawdown penalty |
| State | FEAT metrics + FSM state + Liquidity context |

---

## 4. Nuevas Salidas de Análisis Propuestas

### 4.1 Volatilidad Dinámica

```python
@mcp.tool()
async def dynamic_volatility_map(symbol: str):
    """
    Mapa de volatilidad con regímenes adaptativos.
    
    Algoritmo: GARCH(1,1) + Cambio de régimen Markov
    Justificación: Captura clusters de volatilidad
    Impacto: Ajuste automático de SL/TP según régimen
    """
```

### 4.2 Heatmap de Liquidez

```python
@mcp.tool()
async def liquidity_heatmap(symbol: str, timeframes: list):
    """
    Mapa de calor multi-timeframe de zonas institucionales.
    
    Algoritmo: Agregación de FVG/OB/ZS con decay temporal
    Justificación: Confluencia = mayor probabilidad
    Impacto: Visualización clara de zonas de alta probabilidad
    """
```

### 4.3 Predicción de Breakouts

```python
@mcp.tool()
async def breakout_probability(symbol: str):
    """
    Probabilidad de ruptura usando clustering.
    
    Algoritmo: K-Means sobre (compression, volume, time_in_range)
    Justificación: Patrones pre-breakout son clusterizables
    Impacto: Alertas tempranas de movimientos explosivos
    """
```

### 4.4 Correlación Intermercado

```python
@mcp.tool()
async def cross_market_correlation(base: str, comparisons: list):
    """
    Detección de correlaciones dinámicas.
    
    Algoritmo: Rolling Pearson + DTW para lead/lag
    Justificación: DXY lidera EUR/USD, etc.
    Impacto: Confirmar/filtrar señales con activos correlacionados
    """
```

---

## 5. Roadmap de Implementación

| Fase | Duración | Entregables |
|------|----------|-------------|
| **1** | 2 semanas | Dataset histórico + Feature pipeline |
| **2** | 3 semanas | RandomForest + GBM entrenados |
| **3** | 4 semanas | LSTM temporal + backtesting walk-forward |
| **4** | 6 semanas | RL agent con paper trading |

---

## 6. Conclusión

**Estado actual**: Sistema rule-based sofisticado con arquitectura FEAT/FSM/Liquidity bien diseñada, pero **sin ML real**.

**Oportunidad**: Las estructuras de datos (SEngineerReport, STacticianReport, SSniperReport, SFSMMetrics) son **excelentes features** para entrenar modelos.

**Recomendación inmediata**: Comenzar con Fase 1 (GradientBoosting sobre features existentes) para ganar +15% precisión con ~2 semanas de trabajo.
