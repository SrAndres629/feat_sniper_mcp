"""
[ARCH-HYBRID-PROB-V4-DOCTORAL]
Residual TCN-BiLSTM-Attention with Physics Gating.
==================================================
Scientific Implementation of Hybrid Neural Architecture.

Key Components:
1.  **Residual TCN**: Dilated Causal Convolutions with Residual Connections.
    - Captures long-range temporal dependencies.
    - Uses WeightNorm and Dropout for regularization.
2.  **Bi-LSTM**: Captures sequential context (past and future in training).
3.  **Physics Gating Unit (PGU)**:
    - Fuses Microstructure Physics (OFI, Illiquidity, Kinetic Energy)
    - MECHANISM: Gating Signal = Sigmoid(Linear(Physics_Tensor))
    - Logic: Neural_Signal = LSTM_Out * Gating_Signal
    - Effect: High Entropy/Turbulence explicitly suppresses neural confidence.
4.  **Attention Mechanism**: Soft-Attention for feature importance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
from app.core.config import settings

class Chomp1d(nn.Module):
    """Effectively trims the padding to ensure causal convolution (no peeking future)."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    Residual TCN Block.
    Input -> [DilatedConv -> WeightNorm -> Chomp -> ReLU -> Dropout] x2 -> + Input -> ReLU
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # Branch A: Convolution Path
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Branch B: Skip Connection (ResNet style)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res) # Residual Addition

class PhysicsGatingUnit(nn.Module):
    """
    [DOCTORAL INNOVATION]
    Modulates Neural Features based on Microstructure Entropy.
    
    Formula:
        Gate = Sigmoid( W_p * Physics_Tensor + b_p )
        Output = Neural_Features * Gate
        
    If Physics indicates high risk (e.g., Vacuum), the Gate closes (approaches 0),
    suppressing the Neural Signal and defaulting to uncertainty.
    """
    def __init__(self, feature_dim, physics_dim=4):
        super(PhysicsGatingUnit, self).__init__()
        # Physics Tensor: [OFI_Z, Illiquidity, Impact_Pressure, Kinetic_En]
        self.gate_fc = nn.Linear(physics_dim, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)
        
    def forward(self, neural_features, physics_tensor):
        # neural_features: (Batch, Hidden)
        # physics_tensor: (Batch, Physics_Dim)
        
        if physics_tensor is None:
            return neural_features # Open gate if no physics data
            
        gate_signal = torch.sigmoid(self.gate_fc(physics_tensor))
        return self.ln(neural_features * gate_signal)

class HybridProbabilistic(nn.Module):
    """
    [v4.0-DOCTORAL] The 'Cortex' of the System.
    """
    def __init__(self, input_dim=None, hidden_dim=128, num_classes=3, dropout=0.2):
        super(HybridProbabilistic, self).__init__()
        
        self.input_dim = input_dim or settings.NEURAL_INPUT_DIM
        self.tcn_channels = settings.NEURAL_TCN_CHANNELS
        self.hidden_dim = hidden_dim
        
        # 1. TCN Encoder (Temporal Features)
        self.tcn = nn.Sequential(
            TemporalBlock(self.input_dim, self.tcn_channels, 3, 1, 1, 2, dropout),
            TemporalBlock(self.tcn_channels, self.tcn_channels, 3, 1, 2, 4, dropout),
            TemporalBlock(self.tcn_channels, self.tcn_channels, 3, 1, 4, 8, dropout)
        )
        
        # 2. Bi-LSTM (Sequence Modeling)
        self.lstm = nn.LSTM(
            input_size=self.tcn_channels,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        # LSTM output is 2 * hidden_dim
        
        # 3. Attention Mechanism (Weighted Importance)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # 4. Latent Encoders (Microstructure)
        # We integrate the FEAT Encoder for spatial/form metrics
        from app.ml.models.feat_encoder import FeatEncoder
        self.feat_encoder = FeatEncoder(output_dim=32)
        
        # 5. Physics Gating Unit
        # Fuses the LSTM Context with Hard Physics Metrics
        self.physics_gate = PhysicsGatingUnit(feature_dim=hidden_dim*2, physics_dim=4)
        
        # 6. Heads
        fusion_dim = (hidden_dim * 2) + 32  # LSTM_Gated + Latent_State
        
        self.head_direction = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        self.head_volatility = nn.Linear(fusion_dim, 1) # Regime
        self.head_confidence = nn.Linear(fusion_dim, 1) # Probability (p_win)
        self.head_alpha = nn.Linear(fusion_dim, 1) # Excess Return prediction
        
        self.dropout_rate = dropout
        
        # Metadata
        self.metadata = {
            "version": "4.1.0-DOCTORAL",
            "arch": "Residual-TCN-BiLSTM-PhysicsGated",
            "physics_integration": "Active Gating (Multiplicative)"
        }

    def forward(self, x, feat_input=None, physics_tensor=None, force_dropout=False):
        """
        x: (Batch, Seq, Features)
        feat_input: Dict from FeatEncoder
        physics_tensor: (Batch, 4) [OFI, Illiq, Impact, Kinetic]
        """
        # 1. TCN
        x_tcn = x.permute(0, 2, 1) # (N, C, L)
        x_tcn = self.tcn(x_tcn)
        if force_dropout: x_tcn = F.dropout(x_tcn, p=self.dropout_rate, training=True)
        
        # 2. LSTM
        x_lstm = x_tcn.permute(0, 2, 1) # (N, L, C)
        lstm_out, _ = self.lstm(x_lstm) # (N, L, 2*H)
        
        # 3. Attention
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = (lstm_out * attn_weights).sum(dim=1) # (N, 2*H)
        
        # 4. Physics Gating (The "Doctoral" Step)
        # Modulate the context vector based on physical risk
        if physics_tensor is not None:
            context = self.physics_gate(context, physics_tensor)
            
        # 5. Latent Fusion
        if feat_input is not None:
             z_feat = self.feat_encoder(
                feat_input["form"], 
                feat_input["space"], 
                feat_input["accel"], 
                feat_input["time"],
                feat_input.get("kinetic")
            )
        else:
             z_feat = torch.zeros(x.size(0), 32).to(x.device)
             
        final_fusion = torch.cat([context, z_feat], dim=1)
        
        # 6. Outputs
        return {
            "logits": self.head_direction(final_fusion),
            "volatility": torch.sigmoid(self.head_volatility(final_fusion)),
            "p_win": torch.sigmoid(self.head_confidence(final_fusion)),
            "alpha": self.head_alpha(final_fusion)
        }
