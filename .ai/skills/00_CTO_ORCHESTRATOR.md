# CTO ORCHESTRATOR: Supreme Project Charter (v7.0)

## Corporate Vision
Transforming market chaos into mathematical certainty. The FEAT Software Factory operates as an elite quant production line where physics, machine learning, and auction theory converge into a single sovereign execution engine.

> **Last Updated**: 2026-01-20 | Operations MACRO SHIELD + HERD RADAR

## The Engineering Sectors (Org Chart)

| Sector      | Lead Engineer                                                                                                                  | Primary Focus                                        |
| :---------- | :----------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------- |
| **Neural**  | [ML Architect](file:///c:/Users/acord/OneDrive/Desktop/Bot/feat_sniper_mcp/.ai/skills/neural_dept/ml_architect.md)             | PPO Policy Network, State Encoder, Strategic Cortex  |
| **Physics** | [Spectral Engineer](file:///c:/Users/acord/OneDrive/Desktop/Bot/feat_sniper_mcp/.ai/skills/physics_dept/spectral_engineer.md)  | Kinetic Engine, Wavelet Denoising, Absorption States |
| **Volume**  | [Volume Master](file:///c:/Users/acord/OneDrive/Desktop/Bot/feat_sniper_mcp/.ai/skills/volume_core/volume_senior_fullstack.md) | PVP/POC, KDE Profile, Microstructure Scanner         |
| **Macro**   | *NEW* Fundamental Analyst                                                                                                      | News Events (DEFCON), Retail Sentiment (HERD RADAR)  |
| **Ops**     | [Lead DevOps](file:///c:/Users/acord/OneDrive/Desktop/Bot/feat_sniper_mcp/.ai/skills/ops_dept/lead_devops.md)                  | API Server, Daemon, Docker, MT5 Connection           |
| **QA**      | [QA Engineer](file:///c:/Users/acord/OneDrive/Desktop/Bot/feat_sniper_mcp/.ai/skills/qa_dept/qa_engineer.md)                   | Simulation, Self-Audit, Signal Purity                |
| **Meta**    | [Meta Architect](file:///c:/Users/acord/OneDrive/Desktop/Bot/feat_sniper_mcp/.ai/skills/meta_dept/meta_architect.md)           | .ai Governance, Context Sync, Documentation          |

---

## FILE OWNERSHIP MAP

### 🧠 NEURAL SECTOR (ML Architect)
| File                                        | Purpose                     |
| :------------------------------------------ | :-------------------------- |
| `app/ml/strategic_cortex/policy_network.py` | PPO Actor-Critic Network    |
| `app/ml/strategic_cortex/state_encoder.py`  | State → Tensor conversion   |
| `app/ml/rlaif_critic.py`                    | RLAIF Value Estimation      |
| `nexus_core/features.py`                    | Feature extraction (16-dim) |

### ⚛️ PHYSICS SECTOR (Spectral Engineer)
| File                                    | Purpose                               |
| :-------------------------------------- | :------------------------------------ |
| `nexus_core/kinetic_engine.py`          | Price physics, momentum, acceleration |
| `nexus_core/adaptation_engine.py`       | Dynamic parameter adjustment          |
| `nexus_core/math_engine.py`             | Low-level math (ATR, EMA, RSI)        |
| `nexus_core/structure_engine/engine.py` | FEAT Index, T-Score                   |

### 📊 VOLUME SECTOR (Volume Master)
| File                                        | Purpose                       |
| :------------------------------------------ | :---------------------------- |
| `nexus_core/microstructure/scanner.py`      | Real-time microstructure      |
| `nexus_core/microstructure/ticker.py`       | TickBuffer management         |
| `nexus_core/microstructure/ofi.py`          | Order Flow Imbalance          |
| `nexus_core/structure_engine/pvp_engine.py` | Volume Profile (POC, VAL/VAH) |

### 🌍 MACRO SECTOR (Fundamental Analyst) - NEW
| File                                                     | Purpose                             |
| :------------------------------------------------------- | :---------------------------------- |
| `nexus_core/fundamental_engine/engine.py`                | DEFCON levels, Kill Switch          |
| `nexus_core/fundamental_engine/forexfactory_provider.py` | News scraper                        |
| `nexus_core/fundamental_engine/risk_modulator.py`        | Event proximity risk                |
| `nexus_core/herd_radar.py`                               | **NEW** Retail sentiment (MyFxBook) |

### ⚙️ OPS SECTOR (Lead DevOps)
| File                 | Purpose                    |
| :------------------- | :------------------------- |
| `nexus_daemon.py`    | Process supervisor         |
| `app/api/server.py`  | FastAPI REST endpoints     |
| `app/api/workers.py` | Background task management |
| `app/core/mt5_conn/` | MT5 connection pool        |
| `mcp_server.py`      | MCP AI interface           |

### 🎯 STRATEGY SECTOR (CTO Direct)
| File                                 | Purpose                      |
| :----------------------------------- | :--------------------------- |
| `nexus_core/nexus_engine.py`         | Main orchestration loop      |
| `nexus_core/strategy_engine.py`      | Trade decision logic         |
| `nexus_core/money_management.py`     | Position sizing, RiskOfficer |
| `nexus_training/simulate_warfare.py` | Adversarial simulation       |

### 🖥️ UI SECTOR (Dashboard)
| File                    | Purpose             |
| :---------------------- | :------------------ |
| `dashboard/war_room.py` | Streamlit dashboard |

---

## RECENT OPERATIONS (2026-01-20)

### Operation MACRO SHIELD
**Status**: ✅ COMPLETE
**Commit**: `9ae1d112`

Integrated FundamentalEngine into NexusEngine:
- Gate 0: DEFCON kill switch (highest priority)
- Mock news windows in simulation (-5.0 penalty)
- MACRO SENTINEL display in dashboard

### Operation HERD RADAR
**Status**: ✅ COMPLETE
**Commit**: `5caf3dea`

Integrated retail sentiment from MyFxBook:
- `herd_radar.py` scrapes Long/Short percentages
- `contrarian_score` feature for neural network
- 🐑 HERD RADAR display in dashboard

---

## ACTIVE PROTOCOLS (Phase 5)

1. **DEFCON Kill Switch**: No trading during DEFCON 1 (news imminent < 30 min)
2. **Contrarian Liquidity**: If retail 63% short → Smart Money targets stops above
3. **Adaptive Meta-Controller**: Dynamic scaling via `AdaptationEngine`
4. **Soft Labeling**: Probabilistic targets via `labeler.py`

---

## ROADMAP (Phase 6)

- [ ] Integration Tests for MACRO SHIELD
- [ ] Supabase Edge Function for sentiment proxy
- [ ] High-Frequency Hardening (Cython/C++)
- [ ] Label Purity Audit

---

## Development Workflow

1. **Request**: User provides feature/fix
2. **Delegation**: CTO identifies sector owner
3. **Execution**: Sector Lead implements code
4. **Audit**: Self-audit via `ops_dept/subskill_self_audit.md`
5. **Verification**: QA validates
6. **Delivery**: CTO notifies user

---
*Command & Control: Antigravity AI - CTO Office v7.0*
