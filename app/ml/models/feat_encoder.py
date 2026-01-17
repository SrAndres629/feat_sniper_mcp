import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatEncoder(nn.Module):
    """
    [LEVEL 41] PVP-FEAT Latent Encoder.
    Compresses Market Microstructure (Form, Space, Acceleration, Time) 
    into a dense latent vector Z_t.
    """
    def __init__(self, output_dim=32):
        super(FeatEncoder, self).__init__()
        
        # 1. FORM Sub-network (Profile Geometry)
        # Inputs: Skew, Kurtosis, Value Concentration, Entropy
        self.form_net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # 2. SPACE Sub-network (Liquidity Density)
        # Inputs: Dist to POC, Density > VAH, Density < VAL
        self.space_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # 3. ACCELERATION Sub-network (Energy)
        # Inputs: Energy Z-Score, POC Velocity, Volume Velocity
        self.accel_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # 4. TIME Sub-network (Cycle)
        # Inputs: Session (One-Hot x3), Cycle Progress
        self.time_net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # 5. KINETIC Sub-network (Cognitive Patterns) [LEVEL 50]
        # Input 1: Pattern ID (0-4) -> Embedding(4)
        self.pattern_embedding = nn.Embedding(5, 4)
        # Input 2: Coherence, Alignment, Bias Dist (3 floats)
        # Total to Dense: 4 (embed) + 3 (float) = 7
        self.kinetic_net = nn.Sequential(
            nn.Linear(7, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # Fusion Layer (Concatenates sub-vectors)
        # Total Input: 8 * 5 = 40
        self.fusion = nn.Linear(40, output_dim)
        
    def forward(self, form, space, accel, time, kinetic=None):
        """
        Takes raw metrics and returns Latent State Z_t.
        kinetic: (Batch, 4) -> [PatternID, Coherence, Alignment, BiasDist]
        """
        f = self.form_net(form)
        e = self.space_net(space)
        a = self.accel_net(accel)
        t = self.time_net(time)
        
        # Kinetic Handling
        if kinetic is not None:
            # Split ID and Float metrics
            p_id = kinetic[:, 0].long() # (Batch,)
            metrics = kinetic[:, 1:]    # (Batch, 3)
            
            p_emb = self.pattern_embedding(p_id) # (Batch, 4)
            k_in = torch.cat([p_emb, metrics], dim=1) # (Batch, 7)
            k = self.kinetic_net(k_in) # (Batch, 8)
        else:
            # Fallback for legacy calls (Zero Vector)
            k = torch.zeros_like(f)
        
        # Concatenate manifolds
        combined = torch.cat([f, e, a, t, k], dim=1)
        
        # Project to Latent Space Z
        z = self.fusion(combined)
        return torch.tanh(z) # Normalized State Vector [-1, 1]
