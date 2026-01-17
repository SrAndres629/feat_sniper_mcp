# 🧠 PROJECT_CONTEXT.md: The Technical Landscape

## 🚀 Current Architectural State (Level 62)
The system has surpassed deterministic logic, operating now on a **Probabilistic Convergence Framework**.

### 1. The Cognition Loop
The "Synaptic Path" follows this strict hierarchy:
1. **Context Acquisition**: M1/M5 MT5 OHLCV Data.
2. **Spatio-Temporal Engineering**: `FEATProcessor` creates 50x50 Spatial Maps + 18-dim Latent Vectors.
3. **Neural Adaptation**: `MLEngine` (Hybrid TCN-BiLSTM) generates Multi-Head predictions (Alpha, Prob, Vol, Class).
4. **Bayesian Fusion**: `ConvergenceEngine` evaluates Neural Alpha vs Kinetic Coherence.
5. **Epistemic Gate**: Signal is blocked if Uncertainty > `settings.CONVERGENCE_MAX_UNCERTAINTY`.

### 2. Physical Foundations
- **Momentum & Kinetic Centroids**: Market is analyzed as 4 EMA Physics Layers (Micro, Structure, Macro, Bias).
- **PVP (Price Volume Profile)**: Resistance zones are calculated via High-Frequency Volume Density Integrals.
- **Newtonian Gating**: Signals must align with price acceleration vectors to pass.

### 3. Institutional Risk (The Body)
- **Bayesian Kelly**: Lot sizing is proportional to $\frac{Probability}{Uncertainty}$.
- **Performance Damping**: Real-time Profit Factor (from `DriftMonitor`) automatically scales risk up or down.
- **The Vault**: 50% of realized profits are electronically "locked" from the trading margin.

---

## 🛡️ Critical Constraints for AI
- **Precision**: No hardcoded magic numbers. Use `app/core/config.py`.
- **Numba Optimization**: High-frequency math MUST use `@njit` where possible.
- **Fail-Stop**: If a critical system (Risk/ML) fails to load, `mcp_server.py` MUST halt startup.
