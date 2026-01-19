from typing import Dict
from nexus_core.chronos_engine.phaser import SessionState, MicroPhase, VolatilityRegime

class LiquidityRiskAuditor:
    """
    [RISK MODULATOR - ADAPTIVE]
    Adapts Risk Coefficient based on Volatility Regime.
    BLACKOUT -> 0.0 (Hard Stop)
    LOW -> 0.5 (Asia / Scalp)
    HIGH -> 0.8-1.0 (Trend)
    EXTREME -> 1.0 (Full Expansion)
    """
    
    def audit_risk(self, 
                   state: SessionState, 
                   current_volume: float, 
                   avg_volume_30m: float) -> Dict[str, float]:
        
        # 1. THE BLACKOUT WINDOW (Sacred Rule)
        if state.vol_regime == VolatilityRegime.BLACKOUT:
            return {
                "risk_coefficient": 0.0,
                "reason": f"BLACKOUT WINDOW: {state.warning_label}"
            }

        # 2. Base Risk by Regime
        risk_coefficient = 1.0
        penalty_reason = "None"
        
        if state.vol_regime == VolatilityRegime.LOW:
            risk_coefficient = 0.5 # Mean Reversion / Scalping Sizing
            penalty_reason = "Low Volatility Regime (Asia/Lunch)"
            
        elif state.vol_regime == VolatilityRegime.HIGH:
            risk_coefficient = 1.0 # Standard sizing
            
        elif state.vol_regime == VolatilityRegime.EXTREME:
            risk_coefficient = 1.0 # Full sizing (managed via Stop Loss logic elsewhere)

        # 3. Contextual Penalties (The Probabilistic Layer)
        
        # London Raid Trap
        if state.micro_phase == MicroPhase.LONDON_RAID:
             risk_coefficient *= 0.4 # Significant reduction for Trap Zone
             penalty_reason = "London Raid Trap Probability"

        # NY Confirmation (Volume Check)
        elif state.micro_phase == MicroPhase.NY_CONFIRMATION:
            if current_volume < (1.2 * avg_volume_30m):
                 risk_coefficient *= 0.5
                 penalty_reason += " + Low Volume"
            else:
                 penalty_reason = "Volume Confirmed"

        # Explicit Guard Override
        if state.action_guard == "NO_TRADE":
            risk_coefficient = 0.0
            penalty_reason = "Guard: NO_TRADE"

        return {
            "risk_coefficient": float(round(risk_coefficient, 2)),
            "reason": penalty_reason
        }
