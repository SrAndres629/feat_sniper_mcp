# FEAT SNIPER NEXUS - ARCHITECTURE MAP
## Complete System Blueprint for AI Analysis

> **Purpose**: This document provides a complete map of the system architecture.
> AI agents should read this FIRST before analyzing individual files.

---

## 📁 REPOSITORY STRUCTURE (Essential Files Only)

### 🔴 CONTEXT KILLERS (EXCLUDE FROM ANALYSIS)

These folders/files consume massive context without providing value:

| Path                      | Size     | Reason to Exclude                         |
| :------------------------ | :------- | :---------------------------------------- |
| `.venv/`                  | ~1.6GB   | Python virtual environment (dependencies) |
| `dashboard/node_modules/` | ~434MB   | Node.js dependencies                      |
| `__pycache__/`            | Variable | Python bytecode (auto-generated)          |
| `.git/`                   | Variable | Version control history                   |
| `.ruff_cache/`            | Variable | Linter cache                              |
| `.pytest_cache/`          | Variable | Test cache                                |
| `project_atlas.json`      | 83KB     | Auto-generated project map (JSON noise)   |
| `tools/compile_log.txt`   | 10KB     | Build output log                          |
| `*.pyc`, `*.pyo`          | Variable | Compiled Python files                     |
| `models/*.pt`             | Variable | Neural weight files (binary)              |
| `data/*.db`               | Variable | SQLite databases (binary)                 |
| `logs/*.log`              | Variable | Runtime logs                              |

**Add to `.gitignore` or AI exclusion config:**
```
.venv/
node_modules/
__pycache__/
.git/
.ruff_cache/
.pytest_cache/
*.pyc
*.pt
*.db
*.log
project_atlas.json
```

---

## 🧠 CORE ARCHITECTURE

```
FEAT SNIPER NEXUS
├── ENTRY POINT: LAUNCH_FEAT_DAEMON.bat
│   └── Starts nexus_daemon.py
│
├── DAEMON LAYER (nexus_daemon.py)
│   ├── Launches: FastAPI Server (port 8000)
│   ├── Launches: MCP Server (mcp_server.py)
│   └── Launches: Streamlit Dashboard (port 8501)
│
├── API LAYER (app/api/)
│   ├── server.py      → REST endpoints, WebSocket
│   ├── workers.py     → Background task management
│   └── models.py      → Pydantic schemas
│
├── BRAIN LAYER (nexus_core/)
│   ├── nexus_engine.py    → Main trading loop
│   ├── strategy_engine.py → Trade decisions
│   ├── kinetic_engine.py  → Price physics
│   └── [11 sub-engines]
│
├── NEURAL LAYER (app/ml/)
│   ├── strategic_cortex/  → PPO Policy Network
│   ├── feat_processor/    → Feature extraction
│   └── ml_engine/         → ML orchestration
│
└── UI LAYER (dashboard/)
    └── war_room.py → Streamlit Dashboard
```

---

## 📂 DETAILED FILE MAP

### `/nexus_core/` - The Trading Brain

| File                    | Purpose                                | Inputs                 | Outputs                 | Connects To                     |
| :---------------------- | :------------------------------------- | :--------------------- | :---------------------- | :------------------------------ |
| `nexus_engine.py`       | Main orchestration loop                | MT5 data, Fractals     | Trade signals           | strategy_engine, kinetic_engine |
| `strategy_engine.py`    | Trade decision logic                   | Neural probs, Physics  | TradeLeg objects        | money_management, features      |
| `kinetic_engine.py`     | Price physics (momentum, acceleration) | OHLC DataFrame         | kinetic_metrics dict    | adaptation_engine               |
| `money_management.py`   | Position sizing, Risk Officer          | Account balance, Phase | Volume, Lot size        | strategy_engine                 |
| `features.py`           | Feature extraction for ML              | OHLC, OFI              | Feature vector (16-dim) | ml_engine                       |
| `math_engine.py`        | Low-level math (ATR, EMA, RSI)         | Price arrays           | Indicator values        | All engines                     |
| `adaptation_engine.py`  | Dynamic parameter adjustment           | ATR, Regime            | Thresholds              | kinetic_engine                  |
| `convergence_engine.py` | Multi-signal confluence                | Multiple signals       | Unified score           | strategy_engine                 |
| `memory.py`             | Short-term state storage               | Any                    | Cached values           | nexus_engine                    |

### `/nexus_core/microstructure/` - Zero-Lag Tick Analysis

| File                 | Purpose                    | Inputs       | Outputs             |
| :------------------- | :------------------------- | :----------- | :------------------ |
| `scanner.py`         | Real-time microstructure   | Tick stream  | MicrostructureState |
| `ticker.py`          | Tick buffer management     | Raw ticks    | OrderedArrays       |
| `hurst.py`           | Hurst exponent calculation | Prices       | H value [0,1]       |
| `ofi.py`             | Order Flow Imbalance       | Tick volumes | OFI z-score         |
| `entropy_scanner.py` | Shannon entropy            | Prices       | Entropy [0,1]       |

### `/nexus_core/fundamental_engine/` - News/Macro Analysis

| File                       | Purpose                   | Inputs           | Outputs                    |
| :------------------------- | :------------------------ | :--------------- | :------------------------- |
| `engine.py`                | DEFCON calculator         | Calendar events  | Kill switch, Position mult |
| `calendar_client.py`       | Event data interface      | Provider config  | EconomicEvent list         |
| `forexfactory_provider.py` | Real ForexFactory scraper | URL              | Parsed events              |
| `risk_modulator.py`        | Event proximity risk      | Minutes to event | DEFCON level               |

### `/nexus_core/structure_engine/` - Price Structure Analysis

| File            | Purpose                      | Inputs         | Outputs            |
| :-------------- | :--------------------------- | :------------- | :----------------- |
| `engine.py`     | FEAT Index calculation       | Features       | FEAT score [0-100] |
| `levels.py`     | Support/Resistance detection | OHLC           | Level objects      |
| `pvp_engine.py` | Point of Control, VAL/VAH    | Volume profile | PVP metrics        |

### `/app/ml/` - Machine Learning Layer

| File                   | Purpose                  | Inputs       | Outputs             |
| :--------------------- | :----------------------- | :----------- | :------------------ |
| `ml_normalization.py`  | ATR-based normalization  | Raw features | Normalized features |
| `market_regime.py`     | Regime detection         | Features     | Regime label        |
| `temporal_features.py` | Time-based features      | Timestamps   | Session flags       |
| `fractal_analysis.py`  | Multi-timeframe fractals | OHLC per TF  | Coherence score     |
| `rlaif_critic.py`      | RLAIF value estimation   | State vector | Value score         |

### `/app/ml/strategic_cortex/` - Neural Network Core

| File                | Purpose                 | Inputs          | Outputs                     |
| :------------------ | :---------------------- | :-------------- | :-------------------------- |
| `policy_network.py` | PPO Actor-Critic        | State (16-dim)  | Action probs, Value         |
| `state_encoder.py`  | Raw data → Neural input | Market snapshot | StateVector tensor          |
| `__init__.py`       | Module exports          | -               | policy_agent, state_encoder |

### `/app/ml/feat_processor/` - FEAT Chain

| File            | Purpose              | Inputs           | Outputs            |
| :-------------- | :------------------- | :--------------- | :----------------- |
| `force.py`      | Force score          | Volume, Momentum | Force [0-100]      |
| `exhaustion.py` | Exhaustion detection | Price extremes   | Exhaustion [0-100] |
| `absorption.py` | Absorption zones     | Volume clusters  | Absorption [0-100] |
| `trend.py`      | Trend strength       | EMA slopes       | Trend [0-100]      |

### `/app/core/` - Infrastructure

| File              | Purpose             | Inputs      | Outputs            |
| :---------------- | :------------------ | :---------- | :----------------- |
| `config.py`       | Settings loader     | .env file   | Settings object    |
| `mt5_conn/`       | MT5 connection pool | Credentials | Async MT5 executor |
| `nexus_engine.py` | Engine wrapper      | Config      | Initialized engine |

### `/app/api/` - REST API Layer

| File         | Purpose                 | Inputs        | Outputs            |
| :----------- | :---------------------- | :------------ | :----------------- |
| `server.py`  | FastAPI application     | HTTP requests | JSON responses     |
| `workers.py` | Background task manager | Commands      | Subprocess control |
| `models.py`  | Pydantic schemas        | -             | Type definitions   |

### `/dashboard/` - Web UI

| File          | Purpose             | Inputs        | Outputs   |
| :------------ | :------------------ | :------------ | :-------- |
| `war_room.py` | Streamlit dashboard | API responses | Visual UI |

### `/nexus_training/` - Training Environment

| File                  | Purpose                | Inputs         | Outputs         |
| :-------------------- | :--------------------- | :------------- | :-------------- |
| `simulate_warfare.py` | Adversarial simulation | Episodes count | Trained weights |

### `/.ai/` - AI Governance Documentation

| File                            | Purpose                          |
| :------------------------------ | :------------------------------- |
| `skills/00_CTO_ORCHESTRATOR.md` | Master project overview          |
| `CONSTITUTION.md`               | Core principles and rules        |
| `skills/*/`                     | Department-specific instructions |

---

## 🔄 DATA FLOW

```
MT5 Market Data
    │
    ▼
┌─────────────────┐
│  Tick Listener  │ (app/core/mt5_conn/tick_listener.py)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Microstructure │ (nexus_core/microstructure/scanner.py)
│  Scanner        │ → OFI, Entropy, Hurst
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Kinetic Engine │ (nexus_core/kinetic_engine.py)
│                 │ → Momentum, Acceleration, Absorption
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature        │ (nexus_core/features.py)
│  Extraction     │ → 16-dim StateVector
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Neural Network │ (app/ml/strategic_cortex/policy_network.py)
│  (PPO Policy)   │ → Action probabilities
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Strategy       │ (nexus_core/strategy_engine.py)
│  Engine         │ → TradeLeg objects
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Money Manager  │ (nexus_core/money_management.py)
│                 │ → Position size
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MT5 Executor   │ (app/core/mt5_conn/)
│                 │ → REAL ORDERS
└─────────────────┘
```

---

## 🎯 KEY INTEGRATION POINTS

### Neural Network Input (16 dimensions)
```python
StateVector:
  - balance_normalized     # Account health
  - phase_survival         # Capital phase (one-hot)
  - phase_consolidation
  - phase_institutional
  - ofi_z_score           # Order flow
  - entropy_score         # Market noise
  - hurst_exponent        # Trend persistence
  - spread_normalized     # Liquidity
  - feat_composite        # FEAT chain score
  - scalp_prob            # ML probability
  - day_prob
  - swing_prob
  - titanium_support      # Physics validation
  - titanium_resistance
  - acceleration          # Kinetic state
  - hurst_gate_valid      # Signal gate
```

### API Endpoints
```
GET  /api/status              → System health
POST /api/simulation/start    → Start training
POST /api/simulation/stop     → Stop training
GET  /api/simulation/status   → Progress
POST /api/emergency/close-all → Panic button
WS   /ws/logs                 → Real-time logs
```

---

## 📋 FILES AI SHOULD PRIORITIZE

**Tier 1 (Must Read)**:
- `nexus_core/nexus_engine.py`
- `nexus_core/strategy_engine.py`
- `app/ml/strategic_cortex/policy_network.py`
- `.ai/CONSTITUTION.md`

**Tier 2 (Important)**:
- `nexus_core/kinetic_engine.py`
- `nexus_core/features.py`
- `app/api/server.py`
- `dashboard/war_room.py`

**Tier 3 (Reference)**:
- `nexus_core/microstructure/*.py`
- `app/ml/feat_processor/*.py`
- `nexus_training/simulate_warfare.py`

---

> **Last Updated**: 2026-01-20
> **Version**: 2.0 (MISSION CONTROL Architecture)
