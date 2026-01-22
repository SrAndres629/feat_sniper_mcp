import torch
import torch.nn as nn
import torch.nn.functional as F

class SovereignQuantLoss(nn.Module):
    """
    [V6 ELITE QUANT] - THE SOVEREIGN PROTOCOL.
    Transitions from simple classification to institutional-grade risk/reward modeling.
    
    Features:
    1. Asymmetric Risk: 3.0x penalty for total directional reversals (Buy vs Sell).
    2. Volatility-Adjusted Gradient: Reduces learning weight during noise (ATR Gating).
    3. Focal Loss Integration: Forces focus on high-confluence hard examples.
    4. Killzone Gravity: 2.5x impact during London/NY.
    5. Drawdown Awareness: Scales with margin pressure.
    """
    def __init__(self, weight=None, gamma_focal=2.0, k_lambda=0.6, s_lambda=0.4, fvg_dual_mode=False, fractal_sync=False, temporal_decay=0.01):
        super(SovereignQuantLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')
        self.gamma_focal = gamma_focal
        self.k_lambda = k_lambda
        self.s_lambda = s_lambda
        self.fvg_dual_mode = fvg_dual_mode
        self.fractal_sync = fractal_sync
        self.temporal_decay = temporal_decay
        
    def forward(self, pred, target, physics_tensor, x_map=None, current_balance=20.0, alpha=1.0, timestamps=None):
        """
        physics_tensor: (Batch, 6) -> [Energy, Force, Entropy, Viscosity, Volatility, Intensity]
        alpha: Curriculum factor (0.0 to 1.0) to scale the intensity of elite penalties.
        timestamps: (Batch,) normalized or raw timestamps for recency weighting.
        """
        # 1. BASE CLASSIFICATION with Focal Adjustment
        ce_raw = self.ce_loss(pred, target)
        probs = F.softmax(pred, dim=1)
        pt = torch.exp(-ce_raw)  # probability of correct class
        focal_weight = (1 - pt) ** self.gamma_focal
        
        # 2. EXTRACT PHYSICS & CONTEXT
        # [AUDIT] Expected Order: [Energy, Force, Entropy, Viscosity, Volatility, Intensity]
        force_norm = torch.clamp(physics_tensor[:, 1] / 5.0, 0.0, 1.0)
        entropy = torch.abs(physics_tensor[:, 2])
        viscosity = physics_tensor[:, 3]
        volatility = physics_tensor[:, 4]  # ATR Context
        intensity = physics_tensor[:, 5]   # Killzone Intensity
        
        # 3. ASYMMETRIC RISK (Force-Aligned) [PHASE 1 REFINEMENT]
        # "Don't fight the physics."
        # Physics Force (Index 1) > 0.1 implies Bullish Momentum.
        # Physics Force (Index 1) < -0.1 implies Bearish Momentum.
        physics_force = physics_tensor[:, 1]
        preds_max = torch.argmax(pred, dim=1)
        
        # Penalize: Force is UP but Model says SELL (0)
        fighting_bulls = (physics_force > 0.1) & (preds_max == 0)
        # Penalize: Force is DOWN but Model says BUY (2)
        fighting_bears = (physics_force < -0.1) & (preds_max == 2)
        
        asymmetry_weight = torch.ones_like(ce_raw)
        asymmetry_weight[fighting_bulls] *= 3.0
        asymmetry_weight[fighting_bears] *= 3.0
        
        # Also keep total reversal penalty for non-force days? 
        # No, "Sovereign" means obeying Physics first.
        
        # 4. VOLATILITY-ADJUSTED GRADIENT (ATR Gating)
        # Higher volatility = Lower learning weight (to ignore news noise)
        vol_weight = 1.0 / (1.0 + torch.clamp(volatility, 0.0, 5.0))
        
        # 5. KILLZONE GRAVITY (2.5x impact + Fractal Sync)
        temporal_weight = 1.0 + (intensity * (2.5 if self.fractal_sync else 1.5))
        
        # 6. KINETIC & SPATIAL PENALTIES
        prob_dir = probs[:, 2] - probs[:, 0]
        # Strong direction with low force or high viscosity = Penalty
        kinetic_violation = torch.abs(prob_dir) * (1.0 - force_norm) * (1.0 + viscosity)

        # [ZERO-DAY PROTOCOL] 1. Liquidity Trap Detector (The Fakeout)
        # Condition: High Viscosity (Wicks/Choppiness) + Low Force (No Momentum)
        # If Model predicts Breakout (Sell=0 or Buy=2) in this zone -> 5.0x Penalty
        trap_zone = (viscosity > 0.6) & (force_norm < 0.4)
        breakout_attempt = (torch.argmax(pred, dim=1) != 1) # Trying to trade
        
        trap_penalty = torch.zeros_like(ce_raw)
        # We only penalize if it's a trap zone AND the model tries to trade
        trap_mask = trap_zone & breakout_attempt
        trap_penalty[trap_mask] = ce_raw[trap_mask] * 5.0 
        
        spatial_loss = torch.zeros_like(ce_raw)
        if x_map is not None:
            top_density = torch.sum(x_map[:, :, :25, :], dim=(1,2,3))
            bottom_density = torch.sum(x_map[:, :, 25:, :], dim=(1,2,3))
            total_density = top_density + bottom_density + 1e-9
            spatial_loss = (probs[:, 2] * (top_density / total_density)) + \
                          (probs[:, 0] * (bottom_density / total_density))
            
            # [DOCTORAL] FVG DUAL-OBJECTIVE
            if self.fvg_dual_mode:
                # Penalize BUY if price is at Top Density (potentially hitting Supply FVG)
                # Penalize SELL if price is at Bottom Density (potentially hitting Demand FVG)
                fvg_penalty = (probs[:, 2] * (top_density / total_density) * 2.0) + \
                              (probs[:, 0] * (bottom_density / total_density) * 2.0)
                spatial_loss += fvg_penalty

        # [ZERO-DAY PROTOCOL] 2. Stagnation Penalty (Time-Under-Risk)
        # Condition: Low Volatility (Dead Market) + Model is In-Trade (Buy/Sell)
        # "If the market sleeps, YOU sleep."
        stagnation_mask = (volatility < 0.5) & (breakout_attempt)
        stagnation_penalty = torch.zeros_like(ce_raw)
        stagnation_penalty[stagnation_mask] = ce_raw[stagnation_mask] * 2.0

        # [ZERO-DAY PROTOCOL] 3. Entropy Penalty (Indecision)
        # Penalize lack of conviction. High entropy = Bad.
        entropy_val = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
        # We only punish entropy if it's high (> 1.0 approx for 3 classes)
        entropy_penalty = torch.clamp(entropy_val - 0.8, 0.0, 5.0)

        # 7. FINANCIAL PRESSURE
        MARGIN_PER_001 = 8.0
        if isinstance(current_balance, (float, int)):
            balance_tensor = torch.tensor(float(current_balance), device=pred.device)
        else:
            balance_tensor = current_balance
        pressure_raw = 20.0 / (balance_tensor - MARGIN_PER_001 + 1e-9)
        balance_pressure = torch.clamp(pressure_raw, 1.0, 10.0)

        # [V6.1.3 CORTEX HARDENING] TEMPORAL IMPORTANCE (RECENCY)
        # Formula: W(t) = exp(alpha * (t - max_t))
        temporal_importance = torch.ones_like(ce_raw)
        if timestamps is not None:
            max_ts = torch.max(timestamps)
            temporal_importance = torch.exp(self.temporal_decay * (timestamps - max_ts))
            temporal_importance = temporal_importance / (torch.mean(temporal_importance) + 1e-9)

        # FINAL COMPOSITION
        # Focal Loss * Asymmetry * VolatilityGating * Killzone * Drawdown * Recency
        penalty_composition = focal_weight * asymmetry_weight * vol_weight * temporal_weight * balance_pressure * temporal_importance
        
        # We blend from standard CE (alpha=0) to full Sovereign context (alpha=1)
        weighted_ce = ce_raw * (1.0 + (penalty_composition - 1.0) * alpha)
        
        # [ELITE QUANT SUMMATION]
        total_loss = torch.mean(
            weighted_ce + 
            (kinetic_violation * self.k_lambda * alpha) + 
            (spatial_loss * self.s_lambda * alpha) + 
            (trap_penalty * alpha) +
            (stagnation_penalty * alpha) +
            (entropy_penalty * alpha)
        )
        
        # [CORTEX HARDENING] NaN Shield
        # If loss is Infinite or NaN, clamp it to avoid crashing the CUDA Driver
        if not torch.isfinite(total_loss):
            return torch.tensor(10.0, device=pred.device, requires_grad=True)
            
        return total_loss

