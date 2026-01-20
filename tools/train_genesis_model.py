"""
FEAT NEXUS: OPERATION GENESIS (Model 1.0 Trainer)
=================================================
Bootstraps the first generation of the Neural Engine (Gen 1.0).
Defines the institutional PyTorch architecture and trains on synthetic data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, '.')

from app.ml.feat_processor.spectral import SpectralTensorBuilder
from app.ml.training.labeling import generate_multi_head_label

# ==========================================
# 1. NEURAL ARCHITECTURE (The Hybrid Brain)
# ==========================================
class FeatSniperBrain(nn.Module):
    def __init__(self, input_dim=82, hidden_dim=128, num_heads=3):
        super(FeatSniperBrain, self).__init__()
        
        # A. Tensor Processing (Input Layer)
        self.input_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # B. Temporal Context (LSTM)
        # We assume sequence length of 1 for simplicity in this MVP (Stateless for now, or State-managed externally)
        # For this Gen 1.0, we prioritize the Dense features over sequence memory to speed up inference.
        
        # C. Attention-Like Weighting (Self-Attention simplified)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # D. The Three Heads (Multi-Task Learning)
        # Head 1: Scalp Probability
        self.head_scalp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Head 2: Day Probability
        self.head_day = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Head 3: Swing Probability
        self.head_swing = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        
        embedding = self.input_net(x)
        
        # Logic: In a full sequence model, attention would apply over time steps.
        # Here we apply a direct transformation.
        
        p_scalp = self.head_scalp(embedding)
        p_day = self.head_day(embedding)
        p_swing = self.head_swing(embedding)
        
        # Combine into Prob Distribution (Softmax over headers)
        concat = torch.cat([p_scalp, p_day, p_swing], dim=1)
        return torch.softmax(concat, dim=1)

# ==========================================
# 2. DATA GENERATION (Synthetic Bootstrap)
# ==========================================
def generate_genesis_data(n_samples=5000):
    print(f"Generating {n_samples} synthetic training samples...")
    builder = SpectralTensorBuilder()
    X = []
    y = []
    
    for _ in range(n_samples):
        # Mock OHLCV
        prices = 2000.0 + np.cumsum(np.random.randn(100))
        df = pd.DataFrame({
            'close': prices,
            'high': prices + 1,
            'low': prices - 1,
            'tick_volume': np.random.uniform(100, 1000, 100)
        })
        
        # Features
        features = builder.build_tensors(df)
        
        # Flatten dictionary to vector
        # Order must match feature_vector_schema.md
        # 14 scalars + 64 volume tensor = 78 floats. 
        # (Schema said 82, let's auto-detect)
        
        vec = []
        vec.append(features['domino_alignment'])
        vec.append(features['elastic_gap'])
        vec.append(features['kinetic_whip'])
        vec.append(features['bias_regime'])
        vec.append(features['energy_burst'])
        vec.append(features['trend_purity'])
        vec.append(features['spectral_divergence'])
        vec.append(features['sc10_axis'])
        vec.append(features['vol_scalar'])
        vec.append(float(features['wavelet_level'])) # int to float
        vec.append(features['sgi_gravity'])
        vec.append(features['vam_purity'])
        vec.append(features['svc_confluence'])
        vec.append(features['auction_physics_divergence'])
        
        # Volume Tensor (64)
        vol_tensor = features.get('volume_profile_tensor', [])
        
        if isinstance(vol_tensor, list): vec.extend(vol_tensor)
        elif isinstance(vol_tensor, np.ndarray): vec.extend(vol_tensor.tolist())
        else: vec.extend([0.0] * 64) # Fallback if missing
        
        # Categorical (Volume Shape) - Simple Label Encoding
        shape_map = {"Neutral": 0, "P-Shape": 1, "b-Shape": 2, "D-Shape": 3}
        vec.append(float(shape_map.get(features.get('volume_shape_label', 'Neutral'), 0)))
        
        # Pad or Truncate to exact 82 dimensions
        TARGET_DIM = 82
        if len(vec) < TARGET_DIM:
            vec.extend([0.0] * (TARGET_DIM - len(vec)))
        elif len(vec) > TARGET_DIM:
            vec = vec[:TARGET_DIM]
            
        X.append(vec)
        
        # Label generation
        sgi_val = features.get('sgi_gravity', 0.0)
        profit = np.random.uniform(-50, 150) + (sgi_val * 20)
        duration = np.random.uniform(10, 300)
        
        soft_labels, _ = generate_multi_head_label(abs(profit), duration)
        y.append(soft_labels)
        
    # Convert using explicit dtype to avoid object array creation
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train_genesis():
    print("=== OPERATION GENESIS: TRAINING MODEL 1.0 ===")
    
    # 1. Prepare Data
    X_train, y_train = generate_genesis_data(2000)
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 2. Initialize Brain
    model = FeatSniperBrain(input_dim=82, hidden_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss() # Can use KLDivLoss for soft labels
    
    # 3. Train
    print("\nStarting Training Epochs...")
    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Simple loss: CrossEntropy expects class indices, but we have soft labels.
            # We'll use MSE for soft probability matching or CrossEntropy on argmax
            # For pure soft label regression, MSE is a simple start
            loss = nn.MSELoss()(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/5 | Loss: {total_loss/len(train_loader):.6f}")

    # 4. Save Artifact
    os.makedirs("app/ml/models/active", exist_ok=True)
    save_path = "app/ml/models/active/model_gen_1.0.pth"
    torch.save(model.state_dict(), save_path)
    
    print(f"\n✅ GENESIS COMPLETE. Model saved to: {save_path}")
    print("System is ready for Inference.")

if __name__ == "__main__":
    train_genesis()
