# 🗡️ Skill: Neural Cognition (02)

## 🎯 Objective
Orchestrate the stochastic neural inference path and latent feature adaptation.

## 📂 Domain
- `app/ml/ml_engine.py`
- `app/ml/feat_processor.py`
- `app/ml/models/`

## 📜 Specialized Instructions
1. **Multi-Head Consistency**: All inference calls MUST return the following mapping: `{"buy", "sell", "hold", "p_win", "alpha_multiplier", "uncertainty"}`.
2. **Latent Adaptation**: `compute_latent_vector` is the singular gateway. Never bypass it for raw feature access.
3. **MC Dropout**: Ensure 30 iterations for uncertainty estimation in production.

## 🚫 Prohibited Actions
- **No deterministic p_win**: Never skip the Monte Carlo sampling loop.
- **No raw feature scaling**: All features MUST pass through the standardized normalization layers in `ml_normalization.py`.

## ✅ Success Criteria
- Inference latency remains < 150ms.
- Uncertainty estimation captures market volatility regimes reliably.
