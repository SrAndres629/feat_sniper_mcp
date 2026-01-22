import torch
import torch.nn as nn
import torch.nn.functional as F

class SovereignQuantLoss(nn.Module):
    """
    [V6.2.0 QUANTUM FORGE] - SOFT TARGET & CURRICULUM LOSS.
    Optimized for Soft Labels (Probabilities) and Phased Learning.
    
    Features:
    1. Soft Target Support: Accepts probability distributions [Batch, 3].
    2. Phase Masking: Curriculum-based penalty activation.
    3. Asymmetric Risk: 3.0x penalty for fighting physics.
    4. Focal Loss: Focus on hard examples.
    5. Trap Detection: Liquidity trap penalty.
    """
    def __init__(self, weight=None, gamma_focal=2.0, k_lambda=0.6, s_lambda=0.4, fvg_dual_mode=False, fractal_sync=False, temporal_decay=0.01):
        super(SovereignQuantLoss, self).__init__()
        self.gamma_focal = gamma_focal
        self.k_lambda = k_lambda
        self.s_lambda = s_lambda
        self.fvg_dual_mode = fvg_dual_mode
        self.fractal_sync = fractal_sync
        self.temporal_decay = temporal_decay
        self.weight = weight
        
    def forward(self, pred, target, physics_tensor, x_map=None, current_balance=20.0, alpha=1.0, timestamps=None, phase_mask=None):
        """
        pred: [Batch, 3] (Logits)
        target: [Batch, 3] (Soft Probabilities) OR [Batch] (Hard Indices)
        physics_tensor: (Batch, 6) -> [Energy, Force, Entropy, Viscosity, Volatility, Intensity]
        alpha: Global curriculum scalar.
        phase_mask: Dict with multipliers (kinetic, spatial, alpha) for Curriculum.
        """
        # 0. PHASE CONFIGURATION (Curriculum)
        if phase_mask is None:
            phase_mask = {"kinetic": 1.0, "spatial": 1.0, "alpha": 1.0}
            
        # 1. BASE LOSS (SOFT TARGET AWARE)
        probs = F.softmax(pred, dim=1)
        
        # Detect if target is soft (shape [B,3]) or hard (shape [B])
        if target.dim() == 1 or target.shape[-1] != pred.shape[-1]:
            # Hard target - convert to one-hot for soft loss calculation
            target_soft = F.one_hot(target.long(), num_classes=pred.shape[-1]).float()
        else:
            target_soft = target
            
        # Cross-Entropy for Soft Targets: -sum(target * log(softmax(pred)))
        log_probs = F.log_softmax(pred, dim=1)
        ce_raw = -torch.sum(target_soft * log_probs, dim=1)
        
        # Focal Loss Component (pt = probability of correct class)
        pt = torch.sum(probs * target_soft, dim=1)
        focal_weight = (1 - pt) ** self.gamma_focal
        
        # 2. EXTRACT PHYSICS & CONTEXT
        force_norm = torch.clamp(physics_tensor[:, 1] / 5.0, 0.0, 1.0)
        viscosity = physics_tensor[:, 3]
        volatility = physics_tensor[:, 4]
        intensity = physics_tensor[:, 5]
        
        # 3. ASYMMETRIC RISK (Phase 2+)
        physics_force = physics_tensor[:, 1]
        preds_max = torch.argmax(probs, dim=1)
        
        fighting_bulls = (physics_force > 0.1) & (preds_max == 0)
        fighting_bears = (physics_force < -0.1) & (preds_max == 2)
        
        asymmetry_weight = torch.ones_like(ce_raw)
        if phase_mask.get("kinetic", 1.0) > 0:
            asymmetry_weight[fighting_bulls] *= 3.0
            asymmetry_weight[fighting_bears] *= 3.0
        
        # 4. VOLATILITY-ADJUSTED GRADIENT
        vol_weight = 1.0 / (1.0 + torch.clamp(volatility, 0.0, 5.0))
        
        # 5. KINETIC PENALTIES (Phase 2+)
        prob_dir = probs[:, 2] - probs[:, 0]
        kinetic_violation = torch.abs(prob_dir) * (1.0 - force_norm) * (1.0 + viscosity)
        
        # 6. TRAP DETECTION (Phase 2+)
        trap_zone = (viscosity > 0.6) & (force_norm < 0.4)
        breakout_attempt = (preds_max != 1)
        
        trap_penalty = torch.zeros_like(ce_raw)
        trap_mask = trap_zone & breakout_attempt
        trap_penalty[trap_mask] = ce_raw[trap_mask] * 5.0
        
        # 7. ENTROPY PENALTY (Phase 3 - Alpha)
        entropy_val = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
        entropy_penalty = torch.clamp(entropy_val - 0.8, 0.0, 5.0)
        
        # 8. TEMPORAL IMPORTANCE (Recency Weighting)
        temporal_importance = torch.ones_like(ce_raw)
        if timestamps is not None:
            max_ts = torch.max(timestamps)
            temporal_importance = torch.exp(self.temporal_decay * (timestamps - max_ts))
            temporal_importance = temporal_importance / (torch.mean(temporal_importance) + 1e-9)
        
        # --- FINAL COMPOSITION WITH PHASE MASKS ---
        k_term = (kinetic_violation * self.k_lambda) * phase_mask.get("kinetic", 1.0) * alpha
        t_term = trap_penalty * phase_mask.get("kinetic", 1.0) * alpha
        a_term = entropy_penalty * phase_mask.get("alpha", 1.0) * alpha
        
        # Base Weighted Loss
        weighted_ce = ce_raw * focal_weight * asymmetry_weight * vol_weight * temporal_importance
        
        total_loss = torch.mean(weighted_ce + k_term + t_term + a_term)
        
        # [CORTEX HARDENING] NaN Shield
        if not torch.isfinite(total_loss):
            return torch.tensor(10.0, device=pred.device, requires_grad=True)
            
        return total_loss
