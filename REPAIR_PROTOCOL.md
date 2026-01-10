# 🩺 NEXUS Self-Healing & Repair Protocol

## Overview
NEXUS PRIME is designed with an **Autonomous Resilience Layer**. The system not only detects errors but generates machine-readable instructions to fix itself via the Antigravity AI Agent.

## 🔍 The Detection Loop
The `nexus_auditor.py` script runs continuously during the lifecycle and performs:
1.  **Infrastructure Check**: Docker container presence and connectivity.
2.  **Logic Audit**: Config loading and MQL5 library integrity.
3.  **Data Flow Audit**: ZMQ latency monitoring and Supabase sync verification.

## 🛠️ The REPAIR_REQUEST Format
When the auditor finds an anomaly, it emits a standardized JSON block:

```json
REPAIR_REQUEST_START
{
  "anomalies": ["ZMQ_TIMEOUT", "DB_SYNC_ERROR"],
  "context": {
    "port": 5555,
    "last_sync": "2026-01-10T14:30:00Z"
  },
  "timestamp": "2026-01-10T14:57:16+00:00"
}
REPAIR_REQUEST_END
```

## 🤖 The Antigravity Response
When the AI agent (Antigravity) detects this block in the logs, it follows the **Saneamiento Protocol**:
1.  **Diagnosis**: Analyzes the specific anomaly code (e.g., `MT5_NOT_FOUND`).
2.  **Action**: Executes surgical commands (e.g., `.\nexus.bat`) or patches code (e.g., matching schemas).
3.  **Verification**: Re-runs the auditor until the "SISTEMA READY PARA OPERAR" signal is emitted.

## 🚦 Common Anomaly Codes

| Code | Meaning | Fix Action |
| :--- | :--- | :--- |
| `MT5_DISCONNECT` | Market connection lost. | Restart MT5 via `nexus_control.py`. |
| `DB_SCHEMA_MISMATCH` | Supabase column names changed. | Patch `supabase_sync.py` to match SQL. |
| `ZMQ_LATENCY_HIGH` | Data flow bottleneck. | Flush bridge buffers or restart Docker. |
| `RAG_STORAGE_CORRUPT` | ChromaDB disk error. | Re-index narrative memory. |

---
*Self-Healing Protocol | Mission Critical | NEXUS PRIME*
