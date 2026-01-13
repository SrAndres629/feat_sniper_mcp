# FEAT NEXUS: Data Dictionary

## Overview
Diccionario técnico de todas las constantes, variables y tensores del sistema RLAIF.

---

## 1. FSM Control (nexus_control.py)

### TradingState Enum
| State | Value | WinRate | Behavior |
|-------|-------|---------|----------|
| `RECALIBRATION` | "recalibration" | <35% | Trading stopped |
| `SUPERVISED` | "supervised" | 35-70% | LLM approval required |
| `AUTONOMOUS` | "autonomous" | >70% | Auto-execute |

### PerformanceTracker Constants
| Constant | Value | Description |
|----------|-------|-------------|
| `WINRATE_AUTONOMOUS_THRESHOLD` | 0.70 | WinRate to enter AUTONOMOUS |
| `WINRATE_RECALIBRATION_THRESHOLD` | 0.35 | WinRate to enter RECALIBRATION |
| `MIN_TRADES_FOR_EVALUATION` | 10 | Minimum trades before state change |
| `EVALUATION_WINDOW` | 50 | Rolling window for WinRate |

### Persistence
- File: `data/fsm_state.json`
- Schema: `{state, last_updated, winrate, recent_trades[]}`

---

## 2. The Vault (risk_engine.py)

### TheVault Constants
| Constant | Value | Description |
|----------|-------|-------------|
| `VAULT_STATE_FILE` | "data/vault_state.json" | Persistence file |
| `COMPOUNDING_MULTIPLIER` | 2.0 | Trigger at 2x capital |
| `VAULT_PERCENTAGE` | 0.50 | 50% to vault on trigger |

### Variables
| Variable | Type | Description |
|----------|------|-------------|
| `initial_capital` | float | Starting equity (e.g., $30) |
| `vault_balance` | float | Protected profits (virtual) |
| `trading_capital` | float | Active capital for margin |
| `last_trigger_equity` | float | Baseline for next trigger |

### Persistence
- File: `data/vault_state.json`
- Schema: `{initial_capital, vault_balance, trading_capital, total_transfers, last_trigger_equity}`

---

## 3. Neural Network Inputs (nexus_brain/hybrid_model.py)

### Energy Map Tensor
| Dimension | Shape | Description |
|-----------|-------|-------------|
| Channels | 1 | Grayscale energy map |
| Height | 50 | Price bins |
| Width | 50 | Time bins |
| **Full Shape** | `[batch, 1, 50, 50]` | CNN input |

### Dense Features Vector (12 dims)
| Index | Feature | Source |
|-------|---------|--------|
| 0 | feat_score | nexus_core/features.py |
| 1 | fsm_state | Normalized state code |
| 2 | liquidity_ratio | Order book analysis |
| 3 | volatility_zscore | ATR z-score |
| 4 | momentum_kinetic_micro | M1 momentum |
| 5 | entropy_coefficient | Price disorder |
| 6 | rsi_normalized | RSI / 100 |
| 7 | atr | Raw ATR value |
| 8 | candle_body_ratio | (close-open)/close |
| 9 | candle_range_ratio | (high-low)/close |
| 10 | ema_diff | EMA_fast - EMA_slow |
| 11 | volume_normalized | Volume / 1000 |

---

## 4. n8n Bridge (n8n_bridge.py)

### Configuration
| Variable | Source | Description |
|----------|--------|-------------|
| `N8N_WEBHOOK_URL` | .env | Webhook endpoint |
| `N8N_API_KEY` | .env | Authorization token |
| `timeout` | Config | Request timeout (30s) |

### Persistence
- Config: `data/n8n_config.json`
- Feedback Log: `data/llm_feedback_log.jsonl`

---

## 5. Risk Engine (risk_engine.py)

### Settings Reference
| Setting | Default | Description |
|---------|---------|-------------|
| `RISK_PER_TRADE_PERCENT` | 2.0 | % of equity per trade |
| `MAX_DAILY_DRAWDOWN_PERCENT` | 5.0 | Daily DD limit |
| `ATR_TRAILING_MULTIPLIER` | 1.5 | Trailing stop factor |
| `INITIAL_CAPITAL` | 30.0 | Starting capital for Vault |
| `EQUITY_UNLOCK_THRESHOLD` | 50.0 | Unlock 3rd position |

---

## 6. RLAIF Training (train_hybrid.py)

### Paths
| Path | Description |
|------|-------------|
| `data/market_data.db` | Training data source |
| `data/llm_corrections.jsonl` | LLM correction samples |
| `models/feat_hybrid_v1.pth` | Trained model output |

### Training Hyperparameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 5 | Training iterations |
| `batch_size` | 32 | Samples per batch |
| `learning_rate` | 0.001 | Adam optimizer LR |
| `correction_weight` | 2.0 | Weight multiplier for LLM corrections |
