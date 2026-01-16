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

class HybridSniper(nn.Module):
    """
    [ARCH-HYBRID-V1] Level 25 Engine using TCN-BiLSTM-Attention.
    Optimized for RTX 3060.
    """
    def __init__(self, input_dim, hidden_dim=128, num_classes=3, dropout=0.2):
        super(HybridSniper, self).__init__()
        
        # Block 1: TCN (Temporal Convolutional Network)
        # 3 Layers of Dilated Convolutions
        self.tcn_channels = 64
        self.tcn1 = TemporalBlock(input_dim, self.tcn_channels, kernel_size=3, stride=1, dilation=1, padding=2)
        self.tcn2 = TemporalBlock(self.tcn_channels, self.tcn_channels, kernel_size=3, stride=1, dilation=2, padding=4)
        self.tcn3 = TemporalBlock(self.tcn_channels, self.tcn_channels, kernel_size=3, stride=1, dilation=4, padding=8)
        
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
        
        # Output Layer
        self.fc = nn.Linear(lstm_out_dim, num_classes) # [SELL, HOLD, BUY]
        
    def forward(self, x):
        # x: (Batch, Seq_Len, Features)
        
        # 1. TCN Processing
        # Requires (Batch, Channels, Seq_Len)
        x_tcn = x.permute(0, 2, 1)
        x_tcn = self.tcn1(x_tcn)
        x_tcn = self.tcn2(x_tcn)
        x_tcn = self.tcn3(x_tcn)
        
        # Back to (Batch, Seq_Len, Channels) for LSTM
        x_lstm_in = x_tcn.permute(0, 2, 1)
        
        # 2. Bi-LSTM Processing
        lstm_out, _ = self.lstm(x_lstm_in) # (Batch, Seq, Hidden*2)
        
        # 3. Attention
        # Calculate attention weights
        attn_scores = self.attention(lstm_out) # (Batch, Seq, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Weighted sum
        context = (lstm_out * attn_weights).sum(dim=1) # (Batch, Hidden*2)
        
        # 4. Classification
        logits = self.fc(context)
        return logits # Softmax is applied in Loss or Inference wrapper
