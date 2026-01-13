# n8n Workflow Import Guide

## Quick Start

### 1. Start n8n
```bash
docker run -it --rm -p 5678:5678 n8nio/n8n
```

### 2. Import Workflows

1. Open http://localhost:5678
2. Go to **Settings → Import from File**
3. Import these files:
   - `n8n_workflows/feat_trade_auditor.json`
   - `n8n_workflows/antigravity_code_improver.json`

### 3. Configure Anthropic Credentials

1. **Settings → Credentials → Add Credential**
2. Select **Anthropic**
3. Add your API Key

### 4. Activate Workflows

1. Open each workflow
2. Click **Activate** (toggle on)

---

## Workflow Endpoints

| Workflow | Trigger | Endpoint |
|----------|---------|----------|
| Trade Auditor | Webhook | `POST /webhook/feat-audit` |
| Code Improver | Schedule | Every hour |

---

## Test Trade Auditor

```bash
curl -X POST http://localhost:5678/webhook/feat-audit \
  -H "Content-Type: application/json" \
  -d '{
    "trade_id": "TEST-001",
    "symbol": "XAUUSD",
    "direction": "BUY",
    "nn_confidence": 0.85,
    "feat_scores": {"density": 0.7, "kinetic": 0.8},
    "justification": "Strong bullish momentum"
  }'
```

Expected response:
```json
{
  "decision": "APPROVE",
  "feedback": "High confidence with consistent FEAT scores",
  "suggested_sl": null,
  "suggested_tp": null
}
```
