import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    """
    TCN Block: Dilated Convolution + Chomp + ReLU + Dropout
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
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
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Removes the last elements of a sequence to ensure causal convolution."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class HybridProbabilistic(nn.Module):
    """
    [ARCH-HYBRID-PROB-V1] Probabilistic Hybrid Model (TCN-BiLSTM-Attention).
    Supports Monte Carlo Dropout for Epistemic Uncertainty Estimation.
    """
    def __init__(self, input_dim, hidden_dim=128, num_classes=3, dropout=0.2):
        super(HybridProbabilistic, self).__init__()
        
        
        # Block 1: TCN (Temporal Convolutional Network)
        # TCN accepts (Batch, Channels, Seq_Len)
        # input_dim now includes +7 PVP Features (approx 18-25 total)
        self.tcn_channels = 64
        self.tcn1 = TemporalBlock(input_dim, self.tcn_channels, kernel_size=3, stride=1, dilation=1, padding=2, dropout=dropout)
        self.tcn2 = TemporalBlock(self.tcn_channels, self.tcn_channels, kernel_size=3, stride=1, dilation=2, padding=4, dropout=dropout)
        self.tcn3 = TemporalBlock(self.tcn_channels, self.tcn_channels, kernel_size=3, stride=1, dilation=4, padding=8, dropout=dropout)
        
        # Block 2: Bi-LSTM (Context)
        self.lstm = nn.LSTM(
            input_size=self.tcn_channels,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        ) # Output dim = hidden_dim * 2
        
        lstm_out_dim = hidden_dim * 2
        
        # Block 3: Attention Mechanism (Dot-Product)
        self.attention = nn.Linear(lstm_out_dim, 1)
        
        # [LEVEL 41] PVP-FEAT Latent Encoder Integration
        from app.ml.models.feat_encoder import FeatEncoder
        self.feat_encoder = FeatEncoder(output_dim=32)
        
        # Output Layer
        # Input is Context (Hidden*2) + Latent State (32)
        self.fc = nn.Linear(lstm_out_dim + 32, num_classes) # [SELL, HOLD, BUY]
        
        self.dropout_rate = dropout
        
    def forward(self, x, feat_input=None, force_dropout=False):
        """
        Forward pass with optional forced dropout for MC sampling.
        feat_input: Dictionary of tensors {form, space, accel, time, kinetic}
        """
        # x: (Batch, Seq_Len, Features)
        
        # 1. TCN Processing
        # Requires (Batch, Channels, Seq_Len)
        x_tcn = x.permute(0, 2, 1)
        x_tcn = self.tcn1(x_tcn)
        if force_dropout: x_tcn = F.dropout(x_tcn, p=self.dropout_rate, training=True)
        
        x_tcn = self.tcn2(x_tcn)
        if force_dropout: x_tcn = F.dropout(x_tcn, p=self.dropout_rate, training=True)
        
        x_tcn = self.tcn3(x_tcn)
        if force_dropout: x_tcn = F.dropout(x_tcn, p=self.dropout_rate, training=True)
        
        # Back to (Batch, Seq_Len, Channels) for LSTM
        x_lstm_in = x_tcn.permute(0, 2, 1)
        
        # 2. Bi-LSTM Processing
        lstm_out, _ = self.lstm(x_lstm_in) # (Batch, Seq, Hidden*2)
        
        if force_dropout:
            lstm_out = F.dropout(lstm_out, p=self.dropout_rate, training=True)
        
        # 3. Attention
        attn_scores = self.attention(lstm_out) # (Batch, Seq, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Weighted sum (Temporal Context)
        context = (lstm_out * attn_weights).sum(dim=1) # (Batch, Hidden*2)
        
        if force_dropout:
            context = F.dropout(context, p=self.dropout_rate, training=True)

        # [LEVEL 41] Latent Fusion
        if feat_input is not None:
            # [LEVEL 50] Pass Kinetic Tensor
            z_t = self.feat_encoder(
                feat_input["form"], 
                feat_input["space"], 
                feat_input["accel"], 
                feat_input["time"],
                feat_input.get("kinetic") # Optional
            ) # (Batch, 32)
            
            # Fuse Temporal Context + Structural State
            fusion = torch.cat([context, z_t], dim=1)
        else:
            # Fallback (Zero Vector for Z_t)
            batch_size = x.size(0)
            z_dummy = torch.zeros(batch_size, 32).to(x.device)
            fusion = torch.cat([context, z_dummy], dim=1)

        # 4. Classification
        logits = self.fc(fusion)
        return logits
