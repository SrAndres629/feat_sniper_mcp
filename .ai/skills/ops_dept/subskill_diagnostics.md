# 🚑 Workflow: Emergency Diagnostics

## 🎯 Purpose
Rapid resolution of system-wide halts, MT5 disconnections, or malicious data anomalies.

## 🛠️ Diagnostics Protocol

### 1. MT5 Synapse Verification
// turbo
Check terminal connectivity and ZeroMQ bridge:
```powershell
python -m app.core.mt5_conn --test
```
*If fails: Verify MT5 is open and Expert Advisor 'UnifiedModel' is running.*

### 2. Sentinel Veto Audit
Search logs for the specific kill-switch trigger:
```powershell
Get-Content logs/mcp_server.log -Tail 50 | Select-String "KILL"
```
*If Vetoed: Inspect `data/live_state.json` for toxic anomaly scores.*

### 3. Neural Integrity Probe
Run the automated synaptic wiring test:
// turbo
```powershell
python verify_integral_flow.py
```
*If fails: Re-verify `requirements.txt` and Torch CUDA state.*

### 4. Memory Flush
If DB or Memory lag is detected:
```powershell
Remove-Item data/*.json.tmp -Force
Remove-Item data/*.db -ErrorAction SilentlyContinue
```

## 🏁 Resolution
Once diagnostics pass, restart the nexus:
```powershell
./nexus.bat
```
