import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

logger = logging.getLogger("feat.acceleration.vectors")

class MomentumVector:
    """
    [A] COMPONENT - ACCELERATION (The Physics Engine)
    Calculates Newtonian physics of price: Velocity and Acceleration.
    """
    def __init__(self, threshold: float = 1.5):
        self.threshold = threshold

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Logic: 
        Velocity (V) = Delta Price
        Acceleration (Acc) = Delta Velocity
        """
        if len(df) < 5:
            return {"velocity": 0.0, "acceleration": 0.0, "is_valid": False, "is_trap": False}
            
        prices = df['close'].values
        
        # 1st Derivative: Velocity
        velocity = np.diff(prices)
        # 2nd Derivative: Acceleration
        acceleration = np.diff(velocity)
        
        v_current = velocity[-1]
        a_current = acceleration[-1]
        
        # Rule: Valid breakout requires Acceleration > Threshold
        is_breakout = abs(a_current) > self.threshold
        
        # DIVERGENCE_TRAP: Price moves (V is high) but Acceleration is low or negative
        # Signifies absorption or lack of institutional follow-through
        is_trap = abs(v_current) > (self.threshold * 0.8) and abs(a_current) < (self.threshold * 0.2)
        
        return {
            "velocity": float(v_current),
            "acceleration": float(a_current),
            "is_valid": is_breakout and not is_trap,
            "is_trap": is_trap,
            "status": "ACCELERATING" if is_breakout else ("DIVERGENCE_TRAP" if is_trap else "INERTIA")
        }
