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
from torch.nn.utils import parametrizations
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
        self.conv1 = parametrizations.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = parametrizations.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                   stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # [v6.0] Cortical Hardening: LayerNorm
        self.norm = nn.LayerNorm(n_outputs)
        
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
        
        # Permute for LayerNorm (B, C, L) -> (B, L, C)
        out = out.permute(0, 2, 1)
        out = self.norm(out)
        out = out.permute(0, 2, 1)
        
        return self.relu(out + res)

class MicrostructureGatingUnit(nn.Module):
    """
    [PHASE 13 - DOCTORAL EVOLUTION]
    Replaces simple gating with Cross-Attention between Neural context and Physics state.
    Allows the model to 'attend' to specific physical risks (e.g. illiquidity) 
    before committing to a prediction.
    """
    def __init__(self, feature_dim, physics_dim=6):
        super(MicrostructureGatingUnit, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(physics_dim, feature_dim)
        self.value = nn.Linear(physics_dim, feature_dim)
        self.scale = feature_dim ** 0.5
        self.ln = nn.LayerNorm(feature_dim)
        
    def forward(self, neural_features, physics_tensor):
        if physics_tensor is None: return neural_features
        
        # Cross-Attention: Neural Query, Physics Key/Value
        q = self.query(neural_features).unsqueeze(1) # (B, 1, D)
        k = self.key(physics_tensor).unsqueeze(1)    # (B, 1, D)
        v = self.value(physics_tensor).unsqueeze(1)  # (B, 1, D)
        
        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale
        attn_weights = F.softmax(attn, dim=-1)
        
        gated_context = torch.bmm(attn_weights, v).squeeze(1)
        return self.ln(neural_features + gated_context) # Residual gating

class HybridProbabilistic(nn.Module):
    """
    [v5.0-DOCTORAL IMPACT] The 'Immortal Cortex'.
    Upgraded with Aleatoric Uncertainty and Cross-Attention Gating.
    """
    def __init__(self, input_dim=None, hidden_dim=None, num_classes=None, dropout=0.2):
        super(HybridProbabilistic, self).__init__()
        
        self.input_dim = int(input_dim or settings.NEURAL_INPUT_DIM)
        self.tcn_channels = int(settings.NEURAL_TCN_CHANNELS)
        self.hidden_dim = int(hidden_dim or settings.NEURAL_HIDDEN_DIM)
        self.num_classes = int(num_classes or settings.NEURAL_NUM_CLASSES)
        
        # 1. TCN Encoder (Temporal Stability)
        self.tcn = nn.Sequential(
            TemporalBlock(self.input_dim, self.tcn_channels, 3, 1, 1, 2, dropout),
            TemporalBlock(self.tcn_channels, self.tcn_channels, 3, 1, 2, 4, dropout),
            TemporalBlock(self.tcn_channels, self.tcn_channels, 3, 1, 4, 8, dropout)
        )
        
        # 2. Bi-LSTM (State Persistence)
        self.lstm = nn.LSTM(
            input_size=self.tcn_channels,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # 3. [v6.0] Multi-Head Attention (Noise Filter)
        # Filters the LSTM sequence to highlight institutional signals
        self.mh_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 4. [v6.0] Global Residual Bridge (M1 Persistence)
        # Bridges the raw input directly to the fusion layer
        self.residual_bridge = nn.Linear(self.input_dim, self.hidden_dim * 2)
        
        # 5. Latent Encoders
        from app.ml.models.feat_encoder import FeatEncoder
        self.feat_encoder = FeatEncoder(output_dim=32)
        
        # 5. [v5.0] Microstructure Gating Unit (Cross-Attention)
        self.physics_gate = MicrostructureGatingUnit(feature_dim=self.hidden_dim*2, physics_dim=6)
        
        # 6. Heads with Aleatoric Uncertainty
        fusion_dim = (self.hidden_dim * 2) + 32
        
        # Directional Head
        self.head_direction = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(64),
            nn.Linear(64, self.num_classes)
        )
        
        # [v5.0] Probability and Uncertainty Heads
        self.head_p_win = nn.Linear(fusion_dim, 1)
        self.head_log_var = nn.Linear(fusion_dim, 1) # Aleatoric Uncertainty (Neural Fear)
        self.head_alpha = nn.Linear(fusion_dim, 1)
        
        self.dropout_rate = dropout
        
        self.metadata = {
            "version": "5.0.0-DOCTORAL-UNIFORM",
            "arch": "Bayesian-TCN-BiLSTM-CrossPhysics",
            "uncertainty": "Aleatoric (Log-Variance)"
        }

    def forward(self, x, feat_input=None, physics_tensor=None, force_dropout=False):
        # 1. TCN
        x_tcn = x.permute(0, 2, 1) # (N, C, L)
        x_tcn = self.tcn(x_tcn)
        if force_dropout: x_tcn = F.dropout(x_tcn, p=self.dropout_rate, training=True)
        
        # 2. LSTM
        x_lstm = x_tcn.permute(0, 2, 1) # (N, L, C)
        lstm_out, _ = self.lstm(x_lstm)
        
        # 3. [v6.0] Multi-Head Attention
        # self-attention on the temporal sequence
        attn_out, _ = self.mh_attention(lstm_out, lstm_out, lstm_out)
        context = attn_out[:, -1, :] # Take last timestep context
        
        # [v6.0] Global Residual Skip (M1 Persistence)
        # Directly connect the raw input to context to avoid signal decay
        # x is (B, L, C). We take the last bar (B, C)
        raw_input = x[:, -1, :]
        context = context + self.residual_bridge(raw_input)
        
        # 4. [v5.0] Cross-Attention Physics Gating
        context = self.physics_gate(context, physics_tensor)
            
        # 5. Latent Fusion
        if feat_input is not None:
             z_feat = self.feat_encoder(
                feat_input["form"] if "form" in feat_input else torch.zeros(x.size(0), 4).to(x.device), 
                feat_input["space"] if "space" in feat_input else torch.zeros(x.size(0), 3).to(x.device), 
                feat_input["accel"] if "accel" in feat_input else torch.zeros(x.size(0), 3).to(x.device), 
                feat_input["time"] if "time" in feat_input else torch.zeros(x.size(0), 4).to(x.device),
                feat_input.get("kinetic")
            )
        else:
             z_feat = torch.zeros(x.size(0), 32).to(x.device)
             
        final_fusion = torch.cat([context, z_feat], dim=1)
        
        # 6. Outputs
        p_win = torch.sigmoid(self.head_p_win(final_fusion))
        log_var = self.head_log_var(final_fusion) # Log-variance for stability
        uncertainty = torch.exp(log_var)          # Real Uncertainty [Epistemic proxy]
        
        return {
            "logits": self.head_direction(final_fusion),
            "p_win": p_win,
            "uncertainty": uncertainty,
            "log_var": log_var,
            "alpha": self.head_alpha(final_fusion)
        }
