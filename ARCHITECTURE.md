# FEAT SNIPER NEXUS - SYSTEM MEMORY
## Master Context Document for AI Agents

> **Purpose**: This is the SINGLE SOURCE OF TRUTH for understanding the project.
> Any AI (Claude, Gemini, etc.) should read this FIRST before any analysis.
> This document tells you what exists, where it is, and whether to analyze it.

---

## 🎯 HOW TO USE THIS DOCUMENT

When asked to "audit" or "analyze" files:
1. Check if the file is in **CORE** (analyze deeply)
2. Check if it's in **INFRASTRUCTURE** (understand purpose, skip internals)
3. Check if it's in **BLOAT** (acknowledge existence, never analyze)

---

## 📊 FILE CLASSIFICATION SYSTEM

### 🟢 CORE FILES (~50 files, ~300KB)
**These ARE the project. Analyze deeply.**

### 🟡 INFRASTRUCTURE FILES (~30 files, ~100KB)
**These support the project. Know what they do, don't analyze internals.**

### 🔴 BLOAT FILES (~2GB)
**These are dependencies/cache. Know they exist, NEVER analyze.**

---

# COMPLETE FILE INVENTORY

## 🟢 CORE: TRADING ENGINE (`nexus_core/`)

| File                    | Purpose                                            |  Analyze?   |
| :---------------------- | :------------------------------------------------- | :---------: |
| `nexus_engine.py`       | Main trading loop, orchestrates all components     |   ✅ DEEP    |
| `strategy_engine.py`    | Trade decision logic, TradeLeg creation            |   ✅ DEEP    |
| `kinetic_engine.py`     | Price physics (momentum, acceleration, absorption) |   ✅ DEEP    |
| `features.py`           | Feature extraction for ML (16-dim vector)          |   ✅ DEEP    |
| `math_engine.py`        | Low-level indicators (ATR, EMA, RSI)               | ✅ Reference |
| `money_management.py`   | Position sizing, RiskOfficer                       |   ✅ DEEP    |
| `adaptation_engine.py`  | Dynamic parameter adjustment                       | ✅ Reference |
| `convergence_engine.py` | Multi-signal confluence                            | ✅ Reference |
| `memory.py`             | Short-term state cache                             |   ⚡ Quick   |

### `nexus_core/microstructure/` - Zero-Lag Tick Analysis
| File                 | Purpose                          |  Analyze?   |
| :------------------- | :------------------------------- | :---------: |
| `scanner.py`         | Real-time microstructure scanner |   ✅ DEEP    |
| `ticker.py`          | Tick buffer (TickBuffer class)   | ✅ Reference |
| `hurst.py`           | Hurst exponent calculation       |   ⚡ Quick   |
| `ofi.py`             | Order Flow Imbalance             |   ⚡ Quick   |
| `entropy_scanner.py` | Shannon entropy                  |   ⚡ Quick   |

### `nexus_core/fundamental_engine/` - News/Macro Analysis
| File                       | Purpose                    |  Analyze?   |
| :------------------------- | :------------------------- | :---------: |
| `engine.py`                | DEFCON levels, Kill Switch |   ✅ DEEP    |
| `calendar_client.py`       | Event data interface       | ✅ Reference |
| `forexfactory_provider.py` | Real ForexFactory scraper  | ✅ Reference |
| `risk_modulator.py`        | Event proximity → risk     |   ⚡ Quick   |

### `nexus_core/herd_radar.py` - Retail Sentiment (NEW)
| File            | Purpose                                | Analyze? |
| :-------------- | :------------------------------------- | :------: |
| `herd_radar.py` | MyFxBook scraper, contrarian liquidity |  ✅ DEEP  |

**Provides**: `contrarian_score`, `liquidity_above`, `liquidity_below` for neural network.

### `nexus_core/structure_engine/` - Price Structure

| File            | Purpose                        |  Analyze?   |
| :-------------- | :----------------------------- | :---------: |
| `engine.py`     | FEAT Index calculation         |   ✅ DEEP    |
| `levels.py`     | Support/Resistance detection   | ✅ Reference |
| `pvp_engine.py` | Volume profile (POC, VAL, VAH) | ✅ Reference |

### `nexus_core/physics_engine/` - Price Physics
| File               | Purpose                    | Analyze? |
| :----------------- | :------------------------- | :------: |
| `gravity_model.py` | Price attraction to levels | ⚡ Quick  |

### `nexus_core/zone_projector/` - Zone Analysis
| File                | Purpose                    | Analyze? |
| :------------------ | :------------------------- | :------: |
| `spatial_engine.py` | Zone projection algorithms | ⚡ Quick  |

---

## 🟢 CORE: NEURAL NETWORK (`app/ml/`)

| File                     | Purpose                        |  Analyze?   |
| :----------------------- | :----------------------------- | :---------: |
| `ml_normalization.py`    | ATR-based normalization        |   ✅ DEEP    |
| `market_regime.py`       | Regime detection (trend/range) | ✅ Reference |
| `temporal_features.py`   | Time-based features            |   ⚡ Quick   |
| `fractal_analysis.py`    | Multi-TF fractals              | ✅ Reference |
| `rlaif_critic.py`        | RLAIF value estimation         | ✅ Reference |
| `multi_time_learning.py` | MTF learning                   |   ⚡ Quick   |

### `app/ml/strategic_cortex/` - Neural Core
| File                | Purpose                       |  Analyze?   |
| :------------------ | :---------------------------- | :---------: |
| `policy_network.py` | PPO Actor-Critic, StateVector | ✅✅ CRITICAL |
| `state_encoder.py`  | Raw data → Tensor             |   ✅ DEEP    |

### `app/ml/feat_processor/` - FEAT Chain
| File            | Purpose              | Analyze? |
| :-------------- | :------------------- | :------: |
| `force.py`      | Force score [0-100]  | ⚡ Quick  |
| `exhaustion.py` | Exhaustion detection | ⚡ Quick  |
| `absorption.py` | Absorption zones     | ⚡ Quick  |
| `trend.py`      | Trend strength       | ⚡ Quick  |

### `app/ml/data_collector/` - Data Pipeline
| File           | Purpose                   |  Analyze?   |
| :------------- | :------------------------ | :---------: |
| `labeler.py`   | Training label generation | ✅ Reference |
| `collector.py` | Data collection           |   ⚡ Quick   |

---

## 🟢 CORE: API & DASHBOARD

### `app/api/` - REST API Layer
| File         | Purpose                    |  Analyze?   |
| :----------- | :------------------------- | :---------: |
| `server.py`  | FastAPI endpoints          |   ✅ DEEP    |
| `workers.py` | Background task management | ✅ Reference |
| `models.py`  | Pydantic schemas           |   ⚡ Quick   |

### `dashboard/` - Web UI
| File          | Purpose             | Analyze? |
| :------------ | :------------------ | :------: |
| `war_room.py` | Streamlit dashboard |  ✅ DEEP  |

---

## 🟢 CORE: TRAINING

### `nexus_training/` - Simulation Environment
| File                  | Purpose                | Analyze? |
| :-------------------- | :--------------------- | :------: |
| `simulate_warfare.py` | Adversarial simulation |  ✅ DEEP  |

---

## 🟢 CORE: INFRASTRUCTURE

### `app/core/` - System Infrastructure
| File                        | Purpose               |  Analyze?   |
| :-------------------------- | :-------------------- | :---------: |
| `config.py`                 | Settings loader       | ✅ Reference |
| `mt5_conn/connection.py`    | MT5 connection pool   | ✅ Reference |
| `mt5_conn/tick_listener.py` | Real-time tick stream | ✅ Reference |

### Root Files
| File                     | Purpose            |  Analyze?   |
| :----------------------- | :----------------- | :---------: |
| `nexus_daemon.py`        | Process supervisor |   ✅ DEEP    |
| `mcp_server.py`          | MCP AI interface   | ✅ Reference |
| `LAUNCH_FEAT_DAEMON.bat` | Entry point        |   ⚡ Quick   |

---

## 🟢 CORE: AI GOVERNANCE

### `.ai/` - AI Instructions
| File                            | Purpose                    |  Analyze?   |
| :------------------------------ | :------------------------- | :---------: |
| `CONSTITUTION.md`               | Core principles            | ✅✅ CRITICAL |
| `skills/00_CTO_ORCHESTRATOR.md` | Master project overview    |   ✅ DEEP    |
| `skills/*.md`                   | Department-specific guides | ✅ Reference |

---

## 🟡 INFRASTRUCTURE (Know Purpose, Skip Internals)

### `tools/` - Utility Scripts (~73 files)
**Purpose**: Diagnostic, verification, and maintenance scripts.
**When to analyze**: Only if specifically asked about a particular tool.

Key tools to know exist:
- `verify_*.py` - Various verification scripts
- `test_*.py` - Test scripts
- `fractal_diagnosis.py` - Market fractal analysis
- `force_clean.py` - File cleanup utility
- `download_history.py` - Historical data download

### `tests/` - Unit Tests (~30 files)
**Purpose**: Pytest test suites.
**When to analyze**: Only when debugging test failures.

### `docs/` - Documentation (~13 files)
**Purpose**: Markdown documentation.
**When to analyze**: Reference only when asked.

### `n8n_workflows/` - Automation
**Purpose**: n8n workflow JSON files.
**When to analyze**: Only for integration questions.

### SQL Files (Root)
| File                       | Purpose                    |
| :------------------------- | :------------------------- |
| `knowledge_schema.sql`     | ChromaDB/Knowledge schema  |
| `supabase_schema.sql`      | Supabase table definitions |
| `institutional_schema.sql` | Trading data schema        |

### Docker Files
| File                 | Purpose               |
| :------------------- | :-------------------- |
| `Dockerfile`         | Container build       |
| `docker-compose.yml` | Service orchestration |

### Requirements Files
| File                     | Purpose                 |
| :----------------------- | :---------------------- |
| `requirements.txt`       | All Python dependencies |
| `requirements_base.txt`  | Minimal dependencies    |
| `requirements_heavy.txt` | ML dependencies         |

---

## 🔴 BLOAT (Acknowledge, NEVER Analyze)

### Virtual Environments
| Path                       | Size    | Contents                                     |
| :------------------------- | :------ | :------------------------------------------- |
| `.venv/`                   | ~1.6 GB | Python packages (numpy, torch, pandas, etc.) |
| `.venv/Lib/site-packages/` | ~1.6 GB | Actual package code                          |

### Node.js Dependencies
| Path                      | Size     | Contents                       |
| :------------------------ | :------- | :----------------------------- |
| `dashboard/node_modules/` | ~434 MB  | Next.js, React, Tailwind, etc. |
| `dashboard/.next/`        | Variable | Next.js build output           |

### Cache & Build Artifacts
| Path             | Purpose                      |
| :--------------- | :--------------------------- |
| `__pycache__/`   | Python bytecode (everywhere) |
| `.git/`          | Git version history          |
| `.ruff_cache/`   | Ruff linter cache            |
| `.pytest_cache/` | Pytest cache                 |
| `.mypy_cache/`   | MyPy type checker cache      |
| `.numba_cache/`  | Numba JIT cache              |

### Binary/Generated Files
| Pattern              | Purpose                                      |
| :------------------- | :------------------------------------------- |
| `*.pt`, `*.pth`      | PyTorch model weights                        |
| `*.db`, `*.sqlite3`  | SQLite databases                             |
| `*.log`              | Runtime logs                                 |
| `*.pyc`, `*.pyo`     | Compiled Python                              |
| `project_atlas.json` | Auto-generated project map (83KB JSON noise) |

### Runtime Data
| Path      | Purpose                  |
| :-------- | :----------------------- |
| `data/`   | Runtime data (JSON, DBs) |
| `models/` | Saved neural weights     |
| `logs/`   | Application logs         |

---

## 🔄 DATA FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│                        MARKET DATA                              │
│                     (MT5 Real-time Feed)                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  TIER 1: RAW DATA PROCESSING                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Tick Listener  │  │  Microstructure │  │  Kinetic        │ │
│  │  (tick_listener)│→ │  Scanner        │→ │  Engine         │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  TIER 2: FEATURE EXTRACTION                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  features.py    │  │  FEAT Processor │  │  Adaptation     │ │
│  │  (16-dim vector)│← │  (F,E,A,T)      │← │  Engine         │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  TIER 3: NEURAL DECISION                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  State Encoder  │→ │  Policy Network │→ │  Action Probs   │ │
│  │  (Tensor build) │  │  (PPO Actor)    │  │  (BUY/SELL/HOLD)│ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  TIER 4: STRATEGY & RISK                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Strategy       │→ │  Money Manager  │→ │  Fundamental    │ │
│  │  Engine         │  │  (RiskOfficer)  │  │  Engine (DEFCON)│ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  TIER 5: EXECUTION                                             │
│  ┌─────────────────┐  ┌─────────────────┐                      │
│  │  NexusEngine    │→ │  MT5 Executor   │→ REAL ORDERS        │
│  │  (Orchestrator) │  │  (connection.py)│                      │
│  └─────────────────┘  └─────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 QUICK REFERENCE: AUDIT PRIORITIES

When told to "audit the system", analyze in this order:

### Priority 1: Neural Core (MOST IMPORTANT)
1. `app/ml/strategic_cortex/policy_network.py`
2. `app/ml/strategic_cortex/state_encoder.py`
3. `nexus_core/features.py`

### Priority 2: Decision Logic
4. `nexus_core/strategy_engine.py`
5. `nexus_core/nexus_engine.py`
6. `nexus_core/money_management.py`

### Priority 3: Physics & Microstructure
7. `nexus_core/kinetic_engine.py`
8. `nexus_core/microstructure/scanner.py`
9. `nexus_core/structure_engine/engine.py`

### Priority 4: API & Dashboard
10. `app/api/server.py`
11. `dashboard/war_room.py`
12. `nexus_daemon.py`

### Priority 5: Training
13. `nexus_training/simulate_warfare.py`

### Priority 6: Governance
14. `.ai/CONSTITUTION.md`
15. `.ai/skills/00_CTO_ORCHESTRATOR.md`

---

## 🧠 NEURAL NETWORK INPUT (StateVector)

The PPO Policy Network receives a 16-dimensional input:

```python
StateVector = [
    balance_normalized,      # Account health [0,1]
    phase_survival,          # One-hot: Survival phase
    phase_consolidation,     # One-hot: Consolidation phase
    phase_institutional,     # One-hot: Institutional phase
    ofi_z_score,            # Order Flow Imbalance [-3,3]
    entropy_score,          # Market noise [0,1]
    hurst_exponent,         # Trend persistence [0,1]
    spread_normalized,      # Liquidity [0,1]
    feat_composite,         # FEAT chain score [0,100]
    scalp_prob,             # ML probability [0,1]
    day_prob,               # ML probability [0,1]
    swing_prob,             # ML probability [0,1]
    titanium_support,       # Physics validation [0,1]
    titanium_resistance,    # Physics validation [0,1]
    acceleration,           # Price acceleration [-1,1]
    hurst_gate_valid,       # Signal gate [0,1]
]
```

---

## 📡 API ENDPOINTS REFERENCE

```
System Control:
  GET  /api/status                → System health
  POST /api/emergency/close-all   → Panic button
  POST /api/risk/update           → Risk factor

Simulation:
  POST /api/simulation/start      → Start training
  POST /api/simulation/stop       → Stop training
  GET  /api/simulation/status     → Progress

Analytics:
  GET  /api/analytics/performance → Stats

Models:
  POST /api/models/reload         → Hot-reload weights

Streaming:
  WS   /ws/logs                   → Real-time logs
```

---

> **Last Updated**: 2026-01-20
> **Version**: 3.0 (Complete System Memory)
> **For**: Claude, Gemini, and any AI agent analyzing this project
