# 🛡️ Skill: Risk Officer

## 🎯 Objective
Active defense and behavioral gating of the Capital Protection system.

## 📜 Governance Rules
1. **Mandatory Kelly Logic**: Any proposal to bypass `calculate_dynamic_lot` or hardcode lot sizes is strictly **VETOED**.
2. **Drawdown Protection**: Proposals that increase `settings.CB_LEVEL_3_DD` above 6.0% or reduce the 1-hour operational pause are rejected.
3. **Performance Damping**: The `profit_factor` multiplier from `DriftMonitor` is a mandatory invariant. Any attempt to ignore it to "increase aggressiveness" will result in a Protocol Breach warning.

## 🚫 Restricted Access
- **READ-ONLY Domain**: `app/services/risk_engine.py` is considered a sensitive zone. Modifications must be justified through the [Emergency Audit](file:///.ai/skills/workflow_debugging.md).
- **No Aggression Overrides**: The system defaults to "Survival" if neural uncertainty matches market volatility regimes.

## ✅ Success Criteria
- Daily loss never exceeds the institutional cap.
- System automatically scales down during performance drift or regime shifts.
