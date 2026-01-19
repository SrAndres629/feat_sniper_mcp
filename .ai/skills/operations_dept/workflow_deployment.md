# 🚀 Workflow: Deployment (.ai/skills/workflow_deployment.md)

## 🎯 Purpose
Strategic deployment protocol to transition from training/debug states to Institutional Production.

## 🛠️ Deployment Checklist

### 1. Global Safety Check (Silent Mode)
Ensure the system starts in **SHADOW** mode to prevent accidental order execution during initial bridge sync:
```powershell
# Set TRADING_MODE=SHADOW in .env
Get-Content .env | Select-String "TRADING_MODE"
```

### 2. Operational Infrastructure
Verify all external synapses are active:
- **ZMQ Bridge**: Expert Advisor `UnifiedModel` must be green on an active MT5 chart.
- **Supabase**: `python -m app.core.supabase_sync --test`
- **CPU/GPU**: Ensure `torch.cuda.is_available()` is True for local neural nodes.

### 3. Fail-Stop Validation
Verify that the sentinel will halt startup if integrity is compromised:
```powershell
python -m app.services.system_sentinel --audit
```

### 4. Synaptic Ignition
Start the nexus using the military-grade script:
```powershell
./nexus.bat
```

## 📜 Post-Launch Protocol
1. Observe **HUD** for real-time physics validation.
2. Monitor **Live Dashboard** at `localhost:3000`.
3. Verify **Circuit Breaker** status is `Safe` (Level 0).
