# 🧠 .ai/context/neural_anatomy.md

## 🔬 Model: HybridProbabilistic (v2.1-SINGULARITY)

### 1. Structural Diagram (Logical)
```text
[INPUTS]
   │
   ├─ Temporal (Price/Ind) ──> [TCN Blocks (3 Layers)] ──> [Bi-LSTM (2 Layers)] ──> [Attention Head] ──┐
   │                                                                                                  │
   ├─ Kinetic (FEAT/PVP) ────> [Latent Encoder (32-dim)] ─────────────────────────────────────────────┤─> [FUSION LAYER]
   │                                                                                                  │       │
   └─ Spatial (Volume Map) ──> [Cortex CNN (3 Layers)] ──> [32-dim Vector] ───────────────────────────┘       │
                                                                                                          │
                                                                       ┌──────────────────────────────────┘
                                                                       │
                                                   [MULTI-HEAD PRODUCTION OUTPUTS]
                                                                       │
                                                   ├─ Logits: [Buy/Sell/Hold]
                                                   ├─ P_Win: Probability de Éxito
                                                   ├─ Volatility: Predicted Regime
                                                   └─ Alpha: Risk Multiplier
```

### 2. Input Specifications
- **Price Sequence**: 40 steps (lookback window).
- **Kinetic Tensor**: 18-dimension physics vector (Micro->Bias).
- **Energy Map**: 50x50 Spatial Matrix (Volume Density).

### 3. Training Paradigm
- **Loss**: ConvergentSingularityLoss (Physics-Aware).
- **Inference**: Monte Carlo Dropout for Epistemic Hubris control.
