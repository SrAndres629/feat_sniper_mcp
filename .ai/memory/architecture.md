# 🏗️ .ai/context/architecture.md

## 🌌 The Institutional Synapse
The project operates as an asynchronous pipeline where Physics validates Neural Intent.

### 1. Data Acquisition & Normalization
- **Source**: MT5 Raw OHLCV (M1/M5).
- **Process**: ZMQ Bridge -> `mcp_server.py`.
- **Normalization**: ATR-based scale invariance in `KineticEngine`.

### 2. Feature Engineering (The Body)
- **Engine**: `FeatProcessor` (Numba accelerated).
- **Output**: 
    - 50x50 Spatial Energy Maps (Vision).
    - 18-dimension Latent Vectors (Structural Bias).
    - Z-Score normalized Kinetic Tensors.

### 3. Neural Inference (The Brain)
- **Model**: `HybridProbabilistic` (TCN + BiLSTM + Attention + CNN).
- **Mechanism**: Monte Carlo Dropout (30 iterations) for Uncertainty estimation.
- **Heads**: 
    - `logits`: [Sell, Hold, Buy]
    - `p_win`: Binary confidence.
    - `volatility`: Regime prediction.
    - `alpha`: Risk multiplier.

### 4. Bayesian Fusion & Risk (The Shield)
- **Convergence**: `ConvergenceEngine` fuses Neural Alpha with Kinetic Coherence.
- **Epistemic Veto**: If `Uncertainty > Threshold`, logic is blocked.
- **Risk**: `RiskEngine` applies Damped Kelly and Performance Damping from `DriftMonitor`.
