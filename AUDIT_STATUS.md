# AUDIT STATUS - FEAT Sniper NEXUS

## 📅 Fecha: 2026-01-10

## ✅ Fase 1: Limpieza Completada

### Archivos Movidos a `_deprecated/`

| Archivo | Razón |
|---------|-------|
| `run.py` | Script de inicio obsoleto |
| `brute_force.py` | Desarrollo experimental |
| `start_mcp.bat` | Reemplazado por start_nexus.bat |
| `start_server.bat` | Reemplazado por start_nexus.bat |
| `start_bridge.bat` | Reemplazado por start_nexus.bat |
| `test.mq5` / `test.ex5` | Archivos de prueba |
| `build_error.txt` | Logs de compilación |
| `build_fixed.txt` | Logs de compilación |
| `build_success_attempt.txt` | Logs de compilación |
| `compile_log.txt` | Logs de compilación |
| `error.txt` | Logs de error antiguos |
| `requirements_utf8.txt` | Duplicado |

### Estructura Final Vital

```
feat_sniper_mcp/
├── mcp_server.py          # MCP Server principal
├── docker-compose.yml     # Orquestación Docker
├── start_nexus.bat        # Script de inicio
├── requirements.txt       # Dependencias
├── Dockerfile             # Build de imagen
├── app/                   # Código Python
│   ├── ml/                # ML Engine (GBM, LSTM)
│   ├── skills/            # Skills MCP
│   ├── services/          # RAG Memory
│   └── core/              # Conexión MT5
├── data/                  # SQLite WAL
├── models/                # Modelos entrenados
└── FEAT_Sniper_Master_Core/  # Código MQL5
```

---

## ✅ Fase 2: Endpoint N8N

### Tool: `get_trade_decision()`

**Descripción**: Decisión de trading unificada para integración con N8N.

**Request**:
```json
{
  "symbol": "XAUUSD",
  "timeframe": "M5"
}
```

**Response (Estructura JSON para N8N)**:
```json
{
  "symbol": "XAUUSD",
  "timeframe": "M5",
  "signal": "BUY" | "SELL" | "WAIT",
  "confidence": 0.85,
  "market_state": "ACCUMULATION" | "EXPANSION" | "DISTRIBUTION" | "MANIPULATION",
  "data_context": {
    "price": {"bid": 2650.50, "ask": 2650.80},
    "volatility": {"atr": 15.5, "status": "NORMAL", "spread_points": 30},
    "current_candle": {"open": 2648.0, "high": 2651.0, "low": 2647.5, "close": 2650.5},
    "ml_source": "LSTM" | "GBM" | "NONE",
    "is_anomaly": false
  },
  "timestamp": "2026-01-10T13:00:00.000Z",
  "execution_enabled": false
}
```

### Conexión SSH para N8N

```bash
# Desde máquina remota con N8N
ssh -L 8000:localhost:8000 user@windows-host

# N8N puede conectar a:
# http://localhost:8000/sse
```

---

## ✅ Fase 3: Estado de Puertos y Servicios

| Puerto | Servicio | Estado |
|--------|----------|--------|
| 8000 | MCP SSE API | ⏳ Rebuild pendiente |
| 5555 | ZMQ Bridge | ⏳ Rebuild pendiente |
| 3000 | Web Dashboard | ⏳ Rebuild pendiente |

---

## 📋 Fase 4: Data Harvest

| Componente | Estado |
|------------|--------|
| `data_collector.py` | ✅ Listo para recolección |
| SQLite WAL | ✅ Configurado |
| Oracle Labeling | ✅ Implementado |

---

## 🔄 Próximos Pasos

1. `docker-compose down --rmi all --volumes`
2. `docker-compose up --build -d`
3. Validar logs
4. Conectar N8N vía SSH tunnel

---

*Generado automáticamente por FEAT Sniper NEXUS*
