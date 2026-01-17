# 🛠️ Workflow: Debugging (.ai/skills/workflow_debugging.md)

## 🎯 Purpose
Procedures for identifying synaptic misfires, physical latency, and neural drift.

## 🔬 Diagnostic Toolkit

### 1. The Chaos Test
Verify system resilience under environmental stress:
```powershell
python tools/chaos_test.py
```
*Purpose: Simulates broker disconnects, high-spread peaks, and extreme volatility to ensure Circuit Breakers trip correctly.*

### 2. Immune System Probe
Verify that the protective gating is active and correctly blocking toxic signals:
```powershell
python verify_immune_system.py
```

### 3. Log-Based Forensic Audit
Check for institutional alignment in execution:
```powershell
Get-Content logs/trade_manager.log -Tail 100 | Select-String "ORDER_SEND"
```

## 🚑 Recovery Protocol
1. **Reset Circuit Breaker**: If tripped by false positive, clear the lock in `app/services/circuit_breaker.py` (Manual intervention required if DD > 6%).
2. **Flush Neural Cache**: Clear `data/live_state.json` to force a fresh inference synchronization.
3. **Re-verify Synaptic Flow**: `python verify_integral_flow.py`.

## 📜 Principles
- **Never Patch in Production**: Any code fix must be verified in `Demo/Shadow` mode before master-merge.
- **Fail-Safe over Profit**: If a bug is detected, the primary objective is to HALT the system, not "try to fix it while it's running".
