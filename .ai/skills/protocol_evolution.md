# 🌑 Workflow: Protocol Evolution

## 🎯 Purpose
Strategic weekly evolution of the system's synaptic weights to maintain dominance over shifting market regimes.

## 🌑 Phase 1: Deep Synaptic Synchrony (Saturday-Sunday)
Exhaustive data collection and model recalibration:
1. **The Harvest**: `python -m app.ml.data_collector --days 30 --silent` (Add last month’s fresh reality).
2. **The Forge**: `python -m nexus_training.train_hybrid` (Full weights update).
3. **The Gating**: Loss must be `< 0.15` and `p_win` accuracy `> 65%`.

## 🌑 Phase 2: Shadow Deployment (Monday-Tuesday)
Run the new weights in Shadow mode to verify real-market convergence:
1. `export TRADING_MODE=SHADOW`
2. `./nexus.bat`
3. Check `DriftMonitor` logs for synaptic drift.

## 🌑 Phase 3: Institutional Handover
Verify the [Iron Dome Manifest](file:///.ai/memory/manifest.json) is updated and push to Master:
```powershell
git add .
git commit -m "EVOLUTION: Synaptic Weights Update - [Regime Name]"
git push
```

## ✅ System Principles
- **Never Evolve in Turbulent Regimes**: If ATR > 1.5x average, postpone retraining until volatility stabilizes.
- **Continuous Logic Preservation**: Logic (Physics) is never evolved; only weights (Experience) are.
