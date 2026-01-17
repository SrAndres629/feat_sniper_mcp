# 🗡️ Skill: Risk Governance (01)

## 🎯 Objective
Maintain the institutional integrity of capital protection, dynamic lot sizing, and the Bayesian Kelly framework.

## 📂 Domain
- `app/services/risk_engine.py`
- `app/services/circuit_breaker.py`
- `app/services/volatility_guard.py`

## 📜 Specialized Instructions
1. **Kelly Integrity**: Any change to `_calculate_damped_kelly` must include uncertainty damping. Formula: $Damping = 0.5 * (1 - \frac{Uncertainty}{Threshold})$.
2. **Vault Safety**: The `vault_balance` must NEVER be exposed as tradable margin.
3. **Drawdown Hierarchy**: CB Level 1 (0.75x) -> Level 2 (0.50x + 1h Pause) -> Level 3 (Emergency Close).

## 🚫 Prohibited Actions
- **No Manual Lots**: Never hardcode `volume` values. Always use `calculate_dynamic_lot()`.
- **No Margin Breaches**: Never override `the_vault.get_effective_margin()`.

## ✅ Success Criteria
- Daily drawdown remains below 6.0%.
- Risk scaling is inverse to predicted uncertainty.
