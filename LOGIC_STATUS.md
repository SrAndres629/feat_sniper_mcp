# LOGIC STATUS - FEAT Sniper NEXUS

## 📅 Fecha: 2026-01-10

## ✅ Fase 1: Mapeo de Datos MT5 → Python

### Estructura `SBarDataExport` en CInterop.mqh

| Categoría | Campo | Estado |
|-----------|-------|--------|
| **FEAT Form** | `hasBOS`, `hasCHoCH`, `hasHCH`, `isIntentCandle` | ✅ En struct |
| **FEAT Space** | `atZone`, `proximityScore`, `activeZoneType` | ✅ En struct |
| **FEAT Accel** | `velocity`, `momentum`, `deltaFlow`, `rsi`, `macdHist` | ✅ En struct |
| **FEAT Time** | `isKillzone`, `isLondonKZ`, `isNYKZ`, `activeSession` | ✅ En struct |
| **EMA Layers** | `microComp`, `microSlope`, `operSlope`, `macroSlope`, `biasSlope` | ✅ En struct |
| **FSM** | `marketState`, `compositeScore` | ✅ En struct |
| **PVP** | `poc_level`, `vah`, `val` | ⚠️ No en struct (future add) |

### JSON Exportado por BuildJson()

```json
{
  "type": "MKT_SNAPSHOT",
  "symbol": "XAUUSD",
  "time": "2026.01.10 09:00",
  "price": {"o": 2650.5, "h": 2651.0, "l": 2649.0, "c": 2650.8, "v": 1234},
  "feat": {
    "velocity": 0.0045,
    "momentum": 0.0012,
    "deltaFlow": 0.0008,
    "rsi": 65.4,
    "macdHist": 0.00015,
    "isInstitutional": true,
    "isExhausted": false
  },
  "state": "EXPANSION",
  "score": 78.5
}
```

---

## ✅ Fase 2: Disponibilidad de Tools MCP

### Tools Listos para N8N

| Tool | Función | Estado |
|------|---------|--------|
| `get_full_market_context(symbol)` | Super endpoint con todo | ✅ Listo |
| `get_trade_decision(symbol)` | Señal + confianza | ✅ Listo |
| `get_market_snapshot(symbol)` | Datos básicos | ✅ Listo |
| `execute_trade(symbol, action, volume)` | Ejecución MT5 | ✅ Listo |
| `ml_predict(features)` | Predicción ML | ✅ Listo |
| `remember(text, category)` | Guardar en RAG | ✅ Listo |
| `recall(query)` | Buscar en RAG | ✅ Listo |

---

## ✅ Fase 3: JSON para N8N

### Estructura de `get_full_market_context()`

```json
{
  "symbol": "XAUUSD",
  "timeframe": "M5",
  "timestamp": "2026-01-10T13:00:00.000Z",
  
  "raw_data": {
    "open": 2650.50,
    "high": 2651.20,
    "low": 2649.80,
    "close": 2650.90,
    "volume": 1234,
    "bid": 2650.85,
    "ask": 2651.05,
    "spread_points": 20
  },
  
  "indicators": {
    "feat": {
      "score": 78.5,
      "form": {"bos": true, "choch": false, "intent_candle": true},
      "space": {"at_zone": true, "proximity": 0.85, "zone_type": "FVG"},
      "acceleration": {"velocity": 0.004, "momentum": 0.001, "rsi": 65.4},
      "time": {"is_killzone": true, "session": "NY_KILLZONE"}
    },
    "ema_layers": {
      "micro": {"compression": 0.3, "slope": 0.002},
      "operational": {"slope": 0.001},
      "macro": {"slope": 0.0005},
      "bias": {"slope": 0.0002, "direction": "BULLISH"}
    },
    "pvp": {"poc": 2648.50, "vah": 2655.00, "val": 2642.00},
    "fsm_state": "EXPANSION"
  },
  
  "ml_insight": {
    "model": "GBM",
    "prediction": "BUY",
    "win_probability": 0.78,
    "anomaly_score": 0.05,
    "is_anomaly": false,
    "top_drivers": ["FEAT_Strong_Signal", "EMA_Bullish_Cross"],
    "explanation": "GBM model suggests LONG with high confidence (78%). Price bounced at PVP support with strong FEAT score."
  },
  
  "memory_context": [
    {"text": "XAUUSD typically rallies during NY session...", "relevance": 0.85}
  ],
  
  "strategy_guidance": {
    "system_prompt_suggestion": "You are analyzing XAUUSD...",
    "decision_checklist": ["Is it a killzone?", "Is FSM in EXPANSION?", "..."]
  }
}
```

---

## ✅ Fase 4: Conexión SSH

### Acceso Remoto para N8N

```bash
# Desde servidor N8N
ssh -L 8000:localhost:8000 user@windows-host

# Endpoint MCP SSE disponible en:
http://localhost:8000/sse

# Tools disponibles:
- get_full_market_context
- get_trade_decision
- execute_trade
```

---

## 📊 Matriz de Cobertura FEAT

| Concepto FEAT | Dato en MT5 | Llega a Python | En ML Features |
|---------------|-------------|----------------|----------------|
| **FORMA** | BOS, CHoCH, Intent | ✅ SBarDataExport | ⏳ Pendiente |
| **ESPACIO** | Zone proximity | ✅ SBarDataExport | ⏳ Pendiente |
| **ACELERACIÓN** | Velocity, Momentum | ✅ BuildJson | ⏳ Pendiente |
| **TIEMPO** | Killzone flags | ✅ SBarDataExport | ⏳ Pendiente |

---

## 🔄 Próximos Pasos

1. **Enriquecer BuildJson()** para incluir todos los campos de SBarDataExport
2. **Añadir PVP levels** a la estructura de datos
3. **Integrar datos ZMQ** en `get_full_market_context()`
4. **Entrenar ML** con features FEAT completas

---

## ✅ Confirmación Final

> **Sistema listo.** El Agente N8N puede conectarse por túnel SSH al puerto 8000, solicitar el contexto completo con `get_full_market_context()`, y decidir si ejecuta la orden usando la lógica FEAT.

---

*Generado automáticamente por FEAT Sniper NEXUS*
