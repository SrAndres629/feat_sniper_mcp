# 🧠 .ai/context/neural_brain.md

## 🔬 Model: HybridProbabilistic (v2.1-SINGULARITY)

### 1. Input Tensors
- **Temporal (X)**: (Batch, Seq, Features). OHLCV + Basic Indicators.
- **Structural (feat_input)**: (Batch, 32). Encoded vectors from Form, Space, Accel, Time.
- **Spatial (spatial_map)**: (Batch, 1, 50, 50). CNN-based energy density maps.

### 2. Architecture Blocks
- **Block 1: TCN (Temporal Convolutional Network)**: 3 layers with dilated convolutions for long-term receptive fields.
- **Block 2: Bi-LSTM**: Bidirectional context capture for temporal dependencies.
- **Block 3: Attention**: Dot-product attention to weigh critical time steps.
- **Block 4: Spatial Cortex (CNN)**: Feature extraction from volume energy maps.
- **Block 5: Latent Fusion**: Concatenation of Temporal, Structural, and Spatial latent spaces.

### 3. Multi-Head Production Outputs
- **Logits**: Directional class distribution.
- **P_Win**: Sigmoid probability of reaching Target Profit before Stop Loss.
- **Volatility**: Predicted market regime for adaptive guard rails.
- **Alpha**: Neural risk multiplier for Kelly sizing.

### 4. Epistemic Uncertainty
- **Monte Carlo Dropout**: Enabled during inference (`force_dropout=True`).
- **Standard Deviation**: Calculated over N=30 samples to measure model confidence.
