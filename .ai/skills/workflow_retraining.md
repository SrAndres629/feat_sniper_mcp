# 🧠 Workflow: Retraining (.ai/skills/workflow_retraining.md)

## 🎯 Purpose
Standardized procedure for updating the neural weights based on recent market distributions and multifractal shifts.

## 🛠️ Retraining Protocol

### 1. Data Collection (6 months M5)
Collect deep historical data to capture sufficient regime transitions:
```powershell
python -m app.ml.data_collector --symbol XAUUSD --days 180 --timeframe M5
```

### 2. Synthetic Convergence Generation
Generate a diverse training set using Geometric Brownian Motion and physics-aware labelers:
```powershell
python -m nexus_training.generate_synthetic_convergence
```

### 3. Neural Training
Train the `HybridProbabilistic` model using the unified feature dictionary:
```powershell
python -m nexus_training.train_hybrid
```

### 4. Validation & Quality Gates
Before deployment, verify the following metrics:
- **Loss**: Must be `< 0.2` (Cumulative Cross-Entropy + Bayesian Uncertainty).
- **Val Accuracy**: Must be `> 65%` for directional heads.
- **Inference Stability**: Verify MC Dropout standard deviation is normalized.

## 🏁 Resolution
Once validated, update the production model alias in `app/core/config.py` and run:
`python verify_integral_flow.py`
