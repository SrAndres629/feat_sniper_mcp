# RLAIF Bicameral Architecture - AI Agent Guide

## Overview
Sistema Maestro-Aprendiz donde LLM (n8n) supervisa Redes Neuronales con ciclo de mejora continua.

## FSM States

| State | WinRate | Trading | LLM Role |
|-------|---------|---------|----------|
| 🔴 RECALIBRATION | <35% | STOPPED | Diagnoses & fixes |
| 🟡 SUPERVISED | 35-70% | Requires approval | Approves trades |
| 🟢 AUTONOMOUS | >70% | Auto-execute | Post-mortem audit |

## MCP Tools Available

### Telemetry
```python
# Get NN's "thoughts" for trade
get_inference_full_context(trade_id="xyz")

# Get FSM state + Vault status
get_neural_state()
```

### Capital Management
```python
# Check Vault (capital protection)
get_vault_status()
# Returns: {vault_balance, trading_capital, next_trigger_at}
```

### RLAIF Training
```python
# Trigger retraining after corrections
trigger_retraining(dataset_id="corrections_v1", epochs=5)

# Store trade reflection
store_trade_reflection(trade_id, nn_proposal, llm_feedback, market_result)
```

## n8n Webhook Payloads

### Trade Audit Request
```json
{
  "type": "TRADE_AUDIT",
  "trade_id": "t_123",
  "symbol": "XAUUSD",
  "direction": "BUY",
  "nn_confidence": 0.75,
  "feat_scores": {...},
  "justification": "..."
}
```

### Expected Response
```json
{
  "decision": "APPROVE|REJECT",
  "feedback": "Reason for decision",
  "suggested_sl": 2850.5,
  "suggested_tp": 2870.0
}
```

## State Persistence
- FSM: `data/fsm_state.json`
- Vault: `data/vault_state.json`
- Corrections: `data/llm_corrections.jsonl`
