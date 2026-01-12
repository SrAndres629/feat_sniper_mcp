"""
FEAT NEXUS: HYBRID NEURAL NETWORK (PyTorch)
===========================================
Dual-stream architecture:
1. CNN Stream: Processes FEAT Energy Map (Spatial patterns of liquidity).
2. Dense Stream: Processes FourJarvis Vector (Structural scores).
3. Fusion Layer: Synthesizes Context + Pattern for probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFEATNetwork(nn.Module):
    def __init__(self, dense_input_dim=12):
        super(HybridFEATNetwork, self).__init__()
        
        # 1. CNN Stream (For 50x50 Energy Map)
        # Input: (1, 50, 50) -> Heatmap
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Output after 2 pools (50->25->12): 32 * 12 * 12
        self.cnn_fc = nn.Linear(32 * 12 * 12, 64)
        
        # 2. Dense Stream (For FourJarvis Scores & Physics)
        # Input: [feat_form, feat_space, feat_accel, feat_time, pvp_z, cvd_sl, etc.]
        self.dense1 = nn.Linear(dense_input_dim, 32)
        self.dense2 = nn.Linear(32, 16)
        
        # 3. Fusion Stream
        # Input: 64 (CNN) + 16 (Dense) = 80
        self.fusion1 = nn.Linear(80, 40)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(40, 1) # Probability (Sigmoid)

    def forward(self, energy_map, dense_features):
        """
        energy_map: [Batch, 1, 50, 50]
        dense_features: [Batch, dense_input_dim]
        """
        # CNN Flow
        x1 = self.pool(F.relu(self.conv1(energy_map)))
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = x1.view(-1, 32 * 12 * 12)
        x1 = F.relu(self.cnn_fc(x1))
        
        # Dense Flow
        x2 = F.relu(self.dense1(dense_features))
        x2 = F.relu(self.dense2(x2))
        
        # Fusion
        combined = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fusion1(combined))
        x = self.dropout(x)
        x = torch.sigmoid(self.output(x))
        
        return x

def save_hybrid_model(model, path="models/feat_hybrid_v1.pth"):
    torch.save(model.state_dict(), path)
    
def load_hybrid_model(path="models/feat_hybrid_v1.pth", dense_dim=12):
    model = HybridFEATNetwork(dense_input_dim=dense_dim)
    try:
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
    except FileNotFoundError:
        return None
