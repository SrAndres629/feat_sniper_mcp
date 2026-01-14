# PROYECTO FEAT SNIPER - MAPA DE CONTEXTO (CEREBRO EXTERNO)

## 🏛️ Arquitectura: High Council (10 Master Tools)
El sistema opera bajo `mcp_server.py`, el cual orquesta 10 herramientas maestras que delegan a módulos especializados.

## 🧠 Módulos Cognitivos (Core Logic)
### 1. The Rule Engine (FEAT Strategy)
- **Archivo:** `app/skills/feat_chain.py`
- **Clase:** `FEATChain`
- **Método Principal:** `analyze(candle_data, current_price) -> bool`
- **Responsabilidad:** Validar condiciones Forma, Espacio, Aceleración, Tiempo.
- **Dependencias:** `market_physics`, `liquidity_map` (futuro).

### 2. The Sensory Cortex (Market Physics)
- **Archivo:** `app/skills/market_physics.py`
- **Clase:** `MarketPhysics`
- **Métodos:** 
  - `ingest_tick(data) -> Regimen`
  - `calculate_acceleration(vol, price)`
- **Lógica:** Detecta aceleración si Volumen > 2x Media(20).

### 3. The Neural Link (AI Inference)
- **Archivo:** `nexus_brain/hybrid_model.py` (Dockerized)
- **Responsabilidad:** Inferencia ML (Torch) sobre datos normalizados.
- **Acceso:** Vía HTTP o función directa si está en local.

## 🛡️ Gestión de Riesgo y Ejecución
### 4. The Vault (Risk Engine)
- **Archivo:** `app/services/risk_engine.py`
- **Clase:** `RiskEngine`
- **Responsabilidad:** Aprobar trades basándose en DD diario y exposición.

### 5. Trade Execution
- **Archivo:** `app/skills/trade_mgmt.py`
- **Clase:** `TradeManager`
- **Responsabilidad:** Enviar órdenes vía ZMQ y gestionar reintentos.

## 🔌 Infraestructura (System Nervous System)
- **ZMQ Bridge:** `app/core/zmq_bridge.py` (Publisher/Subscriber).
- **MT5 Conn:** `app/core/mt5_conn.py`.
- **Server:** `mcp_server.py` (FastMCP Lifespan inyecta dependencias).

## 📝 Estado Actual (Fase 1 Completada)
- `mcp_server.py`: Inyectado con FEAT, Risk, Trade.
- `market_physics.py`: Implementado.
- `feat_chain.py`: Implementado (Killzones + Aceleración).

## 🎯 Próximos Pasos (Fase 2)
1. Conectar `brain_run_inference` en `mcp_server.py` hacia `FEATChain.analyze`.
2. Refinar `_check_espacio` y `_check_forma` en `feat_chain.py`.
