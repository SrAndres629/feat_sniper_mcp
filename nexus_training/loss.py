import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvergentSingularityLoss(nn.Module):
    """
    [LEVEL 55 - PATCHED] THE NEURO-MATHEMATICAL SINGULARITY LOSS.
    Corrected for Multi-Dimensional Physics & Negative Loss Prevention.
    """
    def __init__(self, weight=None, kinetic_lambda=0.5, spatial_lambda=0.3):
        super(ConvergentSingularityLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.k_lambda = kinetic_lambda
        self.s_lambda = spatial_lambda
        
    def forward(self, pred, target, physics_tensor, x_map=None):
        """
        pred: (Batch, 3) [SELL, HOLD, BUY]
        target: (Batch)
        physics_tensor: (Batch, 4) -> [Energy, Force, Entropy, Viscosity]
        x_map: (Batch, 1, 50, 50)
        """
        # 1. Base Classification Loss
        ce = self.ce_loss(pred, target)
        
        # 2. EXTRACT PHYSICS DIMENSIONS
        # We need to extract the specific metrics. 
        # Index 1 is Force/Thrust (from our Trainer stack)
        # We clamp it to ensure logic stability.
        force_raw = physics_tensor[:, 1]  # Range [0, 5.0]
        entropy = physics_tensor[:, 2]    # Range [0, 1.0]
        
        # [FIX 1] NORMALIZE FORCE to [0, 1] for penalty logic
        # We assume 5.0 is max force. We want 0.0 to 1.0 ratio.
        force_normalized = torch.clamp(force_raw / 5.0, 0.0, 1.0)
        
        # 3. KINETIC PENALTY (Laws of Physics)
        probs = F.softmax(pred, dim=1)
        prob_dir = probs[:, 2] - probs[:, 0] # Bias towards BUY(+1) or SELL(-1)
        
        # Logic: If Directional Conviction is HIGH (abs(prob_dir) -> 1)
        # BUT Physical Force is LOW (force -> 0), 
        # THEN Penalize. (You shouldn't move fast without force).
        
        # [FIX 2] ABSOLUTE PROTECTION against Negative Loss
        # formula: Violation = Conviction * (1 - Force_Strength)
        # If Force is 1.0 (Max), term becomes 0. No Penalty.
        # If Force is 0.0 (None), term becomes 1. Max Penalty.
        kinetic_violation = torch.abs(prob_dir) * (1.0 - force_normalized)
        
        # Extra: If Entropy is HIGH (Chaos), penalize strong conviction too.
        entropy_violation = torch.abs(prob_dir) * entropy 
        
        total_kinetic_loss = torch.mean(kinetic_violation + entropy_violation)
        
        # 4. SPATIAL PENALTY (Vision Consensus)
        spatial_violation_loss = 0.0
        if x_map is not None:
            # Spatial Heuristic: Density check
            # Top half (Resistance) vs Bottom half (Support)
            top_density = torch.sum(x_map[:, :, :25, :], dim=(1,2,3))
            bottom_density = torch.sum(x_map[:, :, 25:, :], dim=(1,2,3))
            
            # Normalize densities to 0-1 relative to total energy
            total_energy = top_density + bottom_density + 1e-9
            top_ratio = top_density / total_energy
            bottom_ratio = bottom_density / total_energy
            
            # Risk: Buying into Resistance (Top) or Selling into Support (Bot)
            buy_risk = probs[:, 2] * top_ratio
            sell_risk = probs[:, 0] * bottom_ratio
            
            spatial_violation_loss = torch.mean(buy_risk + sell_risk)
            
        # FINAL SUM
        return ce + (total_kinetic_loss * self.k_lambda) + (spatial_violation_loss * self.s_lambda)
