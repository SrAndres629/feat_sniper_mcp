# FEAT Sniper NEXUS Architecture

## Overview

FEAT Sniper NEXUS es una arquitectura institucional que separa el **motor de trading** (MT5/Windows) del **cerebro de IA** (Docker/Python), permitiendo memoria infinita y procesamiento ML sin afectar latencia de ejecución.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ARQUITECTURA NEXUS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐         ZMQ (5555)        ┌────────────────────────┐ │
│  │    MT5 WINDOWS   │ ◄──────────────────────►  │    DOCKER BRAIN       │ │
│  │                  │                           │                        │ │
│  │  ┌────────────┐  │   Señales/Precios/FSM     │  ┌──────────────────┐  │ │
│  │  │ UnifiedModel│──┼──────────────────────────┼──┤  MCP Server (SSE)│  │ │
│  │  │   .mq5     │  │                           │  │   Port 8000      │  │ │
│  │  └────────────┘  │                           │  └────────┬─────────┘  │ │
│  │                  │                           │           │            │ │
│  │  ┌────────────┐  │                           │  ┌────────▼─────────┐  │ │
│  │  │  CFEAT     │  │                           │  │    RAG Memory    │  │ │
│  │  │  CFSM      │  │                           │  │   (ChromaDB)     │  │ │
│  │  │  CLiquidity│  │                           │  │   Persistent Vol │  │ │
│  │  └────────────┘  │                           │  └──────────────────┘  │ │
│  └──────────────────┘                           └────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **MT5 → Docker (ZMQ PUB/SUB)**
   - `UnifiedModel_Main.mq5` calcula FEAT score, FSM state, liquidez
   - Publica datos via ZMQ en puerto 5555
   
2. **Docker Processing**
   - `ZMQ Bridge` recibe datos
   - `RAG Memory` almacena patrones para aprendizaje
   - `MCP Server` expone tools para agentes IA

3. **Docker → Agents (SSE)**
   - Agentes (Claude, GPT, etc.) consultan via SSE en puerto 8000
   - Tools disponibles: `remember`, `recall`, `forget`, `memory_stats`

## SSH Tunnel Access (Remote Memory)

El puerto **8000** expone la API SSE del Agente.  
Para acceso remoto seguro con "memoria ilimitada", usa un túnel SSH:

```bash
# Desde tu máquina local, crea túnel al servidor con Docker
ssh -L 8000:localhost:8000 -L 5555:localhost:5555 user@remote-server

# Ahora puedes conectar tu agente local a:
# - SSE API: http://localhost:8000
# - ZMQ: tcp://localhost:5555
```

Esto permite que cualquier agente IA (Claude, Qwen, etc.) acceda a la memoria RAG persistente como si estuviera local.

## Quick Start

### One-Click Setup (Windows)
```batch
start_nexus.bat
```

### Manual Setup
```bash
# 1. Build and start
docker compose up --build -d

# 2. Check logs
docker logs feat-sniper-brain -f

# 3. Verify endpoints
# SSE: http://localhost:8000
# ZMQ: tcp://localhost:5555
```

## Project Structure

```
feat_sniper_mcp/
├── app/                          # Python backend
│   ├── core/
│   │   ├── mt5_conn.py          # MT5 connection (passive in Docker)
│   │   ├── zmq_bridge.py        # ZMQ pub/sub handling
│   │   └── config.py            # Settings
│   ├── services/
│   │   ├── rag_memory.py        # ChromaDB vector store
│   │   └── supabase_sync.py     # Cloud sync
│   └── skills/                   # Trading skills (market, execution, etc.)
│
├── FEAT_Sniper_Master_Core/     # MQL5 indicator code
│   ├── UnifiedModel_Main.mq5    # Main indicator
│   └── Include/UnifiedModel/    # CFEAT, CFSM, CLiquidity modules
│
├── mcp_server.py                # FastMCP entry point (SSE transport)
├── docker-compose.yml           # Docker orchestration
├── Dockerfile                   # Python image
├── requirements.txt             # Python dependencies
├── start_nexus.bat              # One-click startup (Windows)
├── ARCHITECTURE.md              # This file
└── _deprecated/                 # Archived obsolete files
```

## MCP Tools Available

| Tool | Description |
|------|-------------|
| `remember(text, category)` | Store information in persistent RAG memory |
| `recall(query, limit)` | Semantic search in memory |
| `forget(category)` | Delete memories by category |
| `memory_stats()` | Get memory statistics |
| `get_market_snapshot(symbol)` | Current market state |
| `execute_trade(...)` | Execute trading orders |
| `get_account_status()` | Account metrics |

## Volumes & Persistence

| Volume | Path | Purpose |
|--------|------|---------|
| `chroma-data` | `/app/data/chroma` | RAG memory persistence (survives restarts) |

## Environment Variables

```env
# .env file
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server
CHROMA_PERSIST_DIR=/app/data/chroma
DOCKER_MODE=true
```

## Ports

| Port | Protocol | Service |
|------|----------|---------|
| 8000 | HTTP/SSE | MCP Server (AI Agents) |
| 5555 | TCP/ZMQ | MT5 Bridge (Market Data) |
| 9090 | HTTP | Prometheus Metrics |

---
*Last updated: 2026-01-10*
