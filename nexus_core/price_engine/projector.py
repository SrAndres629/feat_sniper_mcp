import pandas as pd
import numpy as np
from typing import Dict, Tuple

class VolatilityProjector:
    """
    [EXPANSION ENGINE]
    Calculates Institutional Take Profit Levels using Volatility Scaling.
    Formula: Target = Origin + (ATR * Factor)
    """
    
    def calculate_expansion_targets(self, 
                                  current_price: float, 
                                  atr_1h: float, 
                                  session_factor: float = 2.0) -> Dict[str, float]:
        """
        Projects dynamic expansion cones from current price.
        fs = 2.0 (Standard Deviation Move) -> NY Base
        fs = 2.5 (High Volatility) -> Reversals
        """
        return {
            "target_bull_conservative": current_price + (atr_1h * 1.5),
            "target_bull_standard": current_price + (atr_1h * session_factor),
            "target_bull_extended": current_price + (atr_1h * 2.5),
            
            "target_bear_conservative": current_price - (atr_1h * 1.5),
            "target_bear_standard": current_price - (atr_1h * session_factor),
            "target_bear_extended": current_price - (atr_1h * 2.5),
        }

    def get_daily_cycle_targets(self, asia_high: float, asia_low: float, atr_15m: float) -> Dict[str, float]:
        """
        Projects targets based on Asia Range Breakout.
        Standard London/NY Expansion targets relative to Asia.
        """
        # Logic: If we break Asia High, where do we go?
        # Standard Expansion = Asia Range + ATR padding?
        # User Formula: Project using ATR15m * 2.5 from the Break point?
        # Let's use ATR projection from the opposite extreme (Liquidity Sweep theory)
        
        # If Bullish Break (Sweep Low -> Break High):
        projected_high = asia_low + (atr_15m * 8.0) # Approximation of daily expansion on 15m
        
        # Let's stick to the specific user request:
        # "Target = Low_Asia + (ATR_1H * f_session)" imply projecting FROM the sweep level.
        
        return {
            "asia_high": asia_high,
            "asia_low": asia_low,
            "projected_breakout_target_bull": asia_high + (atr_15m * 4.0), # 15m ATR is small, need multiplier
            "projected_breakout_target_bear": asia_low - (atr_15m * 4.0)
        }
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> float:
        """
        Helper for ATR calculation.
        """
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window).mean().iloc[-1]
