
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvergentSingularityLoss(nn.Module):
    """
    [LEVEL 54] THE NEURO-MATHEMATICAL SINGULARITY LOSS.
    Enforces triple-domain consistency:
    1.  Temporal Class Error (CrossEntropy)
    2.  Kinetic Violation (Acceleration vs Direction)
    3.  Spatial Anomaly (Vision vs Probability)
    """
    def __init__(self, kinetic_lambda=0.5, spatial_lambda=0.3):
        super(ConvergentSingularityLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.k_lambda = kinetic_lambda
        self.s_lambda = spatial_lambda
        
    def forward(self, pred, target, physics, x_map=None):
        """
        pred: (Batch, 3) [SELL, HOLD, BUY]
        target: (Batch)
        physics: (Batch) - Kinetic Acceleration factor
        x_map: (Batch, 1, 50, 50) - Spatial Liquidity Map
        """
        ce = self.ce_loss(pred, target)
        
        # 1. KINETIC PENALTY (Laws of Physics)
        probs = F.softmax(pred, dim=1)
        prob_dir = probs[:, 2] - probs[:, 0] # Net directional bias
        
        # If directional intent is high but acceleration is low -> Penalty
        kinetic_violation = torch.abs(prob_dir) * (1.0 - physics)
        
        # 2. SPATIAL PENALTY (Vision Consensus)
        # If the model predicts a BUY but there is an "Energy Wall" (high density) 
        # above price in the energy map, penalize the overconfidence.
        spatial_violation = 0.0
        if x_map is not None:
            # Simple Spatial Heuristic: Sum of density in top half vs bottom half
            # In a real PhD implementation, this would be a Cross-Attention mapping
            top_density = torch.sum(x_map[:, :, :25, :], dim=(1,2,3))
            bot_density = torch.sum(x_map[:, :, 25:, :], dim=(1,2,3))
            
            # Predict BUY (2) but high top density (Wall) -> Risk
            buy_risk = probs[:, 2] * (top_density / (top_density + bot_density + 1e-9))
            # Predict SELL (0) but high bottom density (Floor) -> Risk
            sell_risk = probs[:, 0] * (bot_density / (top_density + bot_density + 1e-9))
            spatial_violation = buy_risk + sell_risk
            
        return ce + (torch.mean(kinetic_violation) * self.k_lambda) + (torch.mean(spatial_violation) * self.s_lambda)
