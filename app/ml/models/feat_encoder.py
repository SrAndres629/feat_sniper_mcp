import torch
import torch.nn as nn

class FeatEncoder(nn.Module):
    """
    [V6.2.0 QUANTUM COMPATIBLE]
    Encodes auxiliary market features into a latent representation.
    Aligned with SniperDataset V6 dimensions.
    """
    def __init__(self, output_dim=32, dims=None):
        super(FeatEncoder, self).__init__()
        
        # 1. Formation Vector (Input 10 per SniperDataset V6)
        self.form_net = nn.Sequential(
            nn.Linear(10, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 16),
            nn.LeakyReLU(0.1)
        )
        
        # 2. Spatial Grid (Input 50 per SniperDataset V6)
        self.space_net = nn.Sequential(
            nn.Linear(50, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh()
        )
        
        # 3. Acceleration Matrix (Input 4)
        self.accel_net = nn.Sequential(
            nn.Linear(4, 16),
            nn.SiLU(), 
            nn.Linear(16, 8),
            nn.SiLU()
        )
        
        # 4. Temporal Encoding (Input 2 per SniperDataset V6)
        self.time_net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )

        # 5. Kinetic Energy (Input 1 per SniperDataset V6)
        self.kin_net = nn.Sequential(
            nn.Linear(1, 8),
            nn.Softplus(),
            nn.Linear(8, 4)
        )

        # Output Projection
        # 16 (Form) + 16 (Space) + 8 (Accel) + 4 (Time) + 4 (Kin) = 48
        self.projection = nn.Linear(48, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, form, space, accel, time, kinetic):
        # Neural Manifolds
        e_form = self.form_net(form)
        e_space = self.space_net(space)
        e_accel = self.accel_net(accel)
        e_time = self.time_net(time)
        e_kin = self.kin_net(kinetic)
        
        # Concatenate & Project
        concat = torch.cat([e_form, e_space, e_accel, e_time, e_kin], dim=1)
        out = self.projection(concat)
        return self.norm(out)
