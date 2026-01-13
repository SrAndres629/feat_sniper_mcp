# FEAT NEXUS: Input/Output Map

## System Architecture Flow

```
MT5 Terminal → ZMQ Bridge → Data Collector → Neural Network → FSM → n8n LLM → Execution
     ↓              ↓              ↓               ↓           ↓        ↓          ↓
  Ticks/DOM    Port 5555    SQLite/FEAT      Prediction  State Check  Audit   MT5 Orders
```

---

## Module I/O Specifications

### 1. nexus_control.py (FSM Controller)

| Type | Name | Format | Description |
|------|------|--------|-------------|
| **INPUT** | Trade result | `{profit: float, closed: bool}` | From MT5 |
| **INPUT** | Performance metrics | `{winrate: float, trades: int}` | From PerformanceTracker |
| **OUTPUT** | Current state | `TradingState` enum | RECAL/SUPERVISED/AUTO |
| **OUTPUT** | Execution decision | `bool` | Allow trade or not |

### State Transitions
```
evaluate_performance() → {winrate, closed_count}
transition_state(winrate) → TradingState
get_status() → {current_state, winrate, wins, losses, thresholds}
```

---

### 2. risk_engine.py (TheVault)

| Type | Name | Format | Description |
|------|------|--------|-------------|
| **INPUT** | current_equity | `float` | From MT5 account_info |
| **OUTPUT** | vault_trigger | `{type, vault_amount, reinvest_amount}` or `None` |
| **OUTPUT** | effective_margin | `float` | Margin excluding vault |

### Key Functions
```python
check_vault_trigger(current_equity) → Optional[Dict]
# Returns None if not triggered, or transfer alert dict

get_effective_margin(account_free_margin) → float
# Returns capped margin excluding virtual vault

get_status() → Dict
# Returns full vault state for telemetry
```

---

### 3. brain_core/mcp_server.py (MCP Tools)

#### get_inference_full_context(trade_id)
| Type | Format |
|------|--------|
| **INPUT** | `trade_id: str` (optional) |
| **OUTPUT** | `{trade_id, symbol, timestamp, input_tensors, feat_scores, prediction, justification}` |

#### get_neural_state()
| Type | Format |
|------|--------|
| **INPUT** | None |
| **OUTPUT** | `{fsm: {state, winrate, trades}, vault: {balance, capital}, timestamp}` |

#### store_trade_reflection(trade_id, nn_proposal, llm_feedback, market_result)
| Type | Format |
|------|--------|
| **INPUT** | `trade_id: str, nn_proposal: dict, llm_feedback: dict, market_result: dict` |
| **OUTPUT** | `{status: "stored", doc_id: str}` |

---

### 4. app/services/n8n_bridge.py

#### request_audit(AuditRequest)
| Field | Type | Description |
|-------|------|-------------|
| trade_id | str | Unique trade identifier |
| symbol | str | Trading instrument |
| direction | str | BUY/SELL |
| entry_price | float | Proposed entry |
| proposed_sl/tp | float | Risk management |
| nn_confidence | float | Network confidence 0-1 |
| feat_scores | dict | FEAT metrics |
| justification | str | Human-readable reasoning |

**Returns:**
```json
{
  "decision": "APPROVE|REJECT",
  "feedback": "LLM explanation",
  "suggested_sl": 2850.5,
  "suggested_tp": 2870.0,
  "confidence_adjustment": 0.0
}
```

---

### 5. nexus_training/train_hybrid.py

#### train_hybrid_brain(epochs, batch_size, save_path, learning_rate)
| Type | Name | Format |
|------|------|--------|
| **INPUT** | epochs | int (default 5) |
| **INPUT** | batch_size | int (default 32) |
| **INPUT** | learning_rate | float (default 0.001) |
| **OUTPUT** | Result dict | `{status, epochs, samples, final_loss, model_path}` |

#### trigger_retraining(dataset_id, epochs, learning_rate)
| Type | Name | Format |
|------|------|--------|
| **INPUT** | dataset_id | str (optional) |
| **OUTPUT** | Result dict | Same as train_hybrid_brain + timestamp |

---

### 6. Data Flows

#### Trade Proposal Flow (SUPERVISED Mode)
```
1. NN generates prediction → feat_scores + confidence
2. FSM checks state → SUPERVISED
3. n8n_bridge.request_audit(proposal) → webhook call
4. Wait for LLM response → APPROVE/REJECT
5. If APPROVE → execute_trade()
6. store_trade_reflection() → ChromaDB
```

#### Vault Trigger Flow
```
1. Position closes → profit calculated
2. risk_engine.check_vault_trigger(equity)
3. If equity >= 2x baseline:
   - vault_balance += profit * 0.5
   - trading_capital = baseline + (profit * 0.5)
   - Send alert via webhook
4. Update data/vault_state.json
```

#### RLAIF Learning Flow
```
1. Trade closes → profit/loss recorded
2. Compare NN prediction vs LLM feedback vs result
3. If discrepancy → create_correction_dataset()
4. Corrections saved to data/llm_corrections.jsonl
5. On trigger_retraining() → corrections loaded with 2x weight
6. Model updated → models/feat_hybrid_v1.pth
```
