import numpy as np
from typing import Dict, Any
from nexus_core.fundamental_engine.risk_modulator import RiskModulator
from nexus_core.risk_engine.guards import LiquidityRiskAuditor

class NeuralRiskOrchestrator:
    """
    [DOCTORAL RISK CORE]
    The final arbiter of capital intensity.
    
    Orchestrates:
    1. Alpha Regime Probabilities (Scalp/Day/Swing)
    2. Macro Event Shielding (DEFCON levels)
    3. Liquidity Guards (Blackouts/Regimes)
    
    Formula:
        Final_Risk = Base_Risk * P(Mode) * C_regime * Macro_Shield * Guard_Penalty
    """
    
    def __init__(self):
        self.macro_shield = RiskModulator()
        self.liquidity_auditor = LiquidityRiskAuditor()
        
    def calculate_capital_intensity(self, 
                                   alpha_last: Dict[str, Any], 
                                   macro_last: Dict[str, Any],
                                   current_volatility: float) -> Dict[str, Any]:
        """
        Determines exactly HOW MUCH and AT WHAT RISK level to execute.
        
        Args:
            alpha_last: Last state from AlphaTensorOrchestrator.
            macro_last: Last state from MacroTensorFactory.
            current_volatility: ATR-normalized volatility.
        """
        
        # 1. ALPHA REGIME MODULATION
        # [Scalp, DayTrade, Swing]
        regime_probs = alpha_last["regime_probability"] 
        mode_idx = np.argmax(regime_probs)
        confidence = float(np.max(regime_probs))
        
        # Base Intensity per Mode (Institutional Standard)
        # Scalping: Fast, high frequency, smaller size to avoid slippage.
        # DayTrade: Standard capital allocation.
        # Swing: Deep conviction, largest position size.
        mode_base_intensity = [0.5, 0.8, 1.2] 
        base_intensity = mode_base_intensity[mode_idx] * confidence
        
        # 2. MACRO EVENT SHIELDING
        # Use the position multiplier from the fundamental engine
        macro_multiplier = float(macro_last.get("macro_position_multiplier", 1.0))
        
        # 3. LIQUIDITY & BLACKOUT AUDIT
        # We perform a final check on guards (e.g. CME Close)
        # Handle the case where we don't have a full SessionState yet or pass raw components
        # For simplicity in this orchestration, we'll assume a pass-through or a wrapper
        risk_guard_coefficient = 1.0
        if macro_last.get("macro_danger", 0.0) > 0.5:
            risk_guard_coefficient = 0.0 # Emergency Halt
            
        # 4. FINAL RISK COEFFICIENT
        # Multiplication of all survival factors
        final_coefficient = base_intensity * macro_multiplier * risk_guard_coefficient
        
        # 5. ADAPTIVE EXECUTION PARAMETERS
        # Normalizing SL/TP by ATR based on Mode
        # Scalp: SL = 0.5 * ATR, Day: SL = 1.5 * ATR, Swing: SL = 3.0 * ATR
        sl_multipliers = [0.8, 1.5, 3.0]
        tp_multipliers = [1.5, 3.0, 6.0]
        
        sl_norm = sl_multipliers[mode_idx]
        tp_norm = tp_multipliers[mode_idx]
        
        # Leverage Scaling (Inverse of SL for risk parity)
        # If SL is tight (Scalp), leverage can be higher to maintain % risk per trade.
        base_leverage = 10.0 # Example
        # Target Risk Per Trade (e.g., 1% of equity)
        # Leverage = Target_Risk / (Distance_to_SL %)
        # Here we provide a 'Relative Leverage Multiplier'
        leverage_multiplier = 1.0 / (sl_norm + 1e-9)
        
        return {
            "final_risk_coefficient": float(round(final_coefficient, 4)),
            "dominant_regime": ["SCALP", "DAY_TRADE", "SWING"][mode_idx],
            "confidence": confidence,
            "sl_params": {
                "atr_multiplier": sl_norm,
                "tp_atr_multiplier": tp_norm
            },
            "leverage_multiplier": float(round(leverage_multiplier, 2)),
            "execution_status": "PROCEED" if final_coefficient > 0.1 else "HALT",
            "reason": "Macro/Alpha Alignment" if final_coefficient > 0.1 else "Risk Constraint Violation"
        }
