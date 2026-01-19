import numpy as np
import pandas as pd
from typing import Dict
import datetime

from nexus_core.price_engine.projector import VolatilityProjector
from nexus_core.chronos_engine.macro import MacroCycleTracker

class TargetTensorFactory:
    """
    [EXPANSION HORIZON - NEURAL TARGETS]
    Feeds the AI with:
    1. 'Where' (Distance to ATR Targets).
    2. 'When (Macro)' (Weekly/Fractal alignment).
    """
    
    def __init__(self):
        self.projector = VolatilityProjector()
        self.macro_tracker = MacroCycleTracker()

    def process_targets(self, 
                       timestamp_utc: datetime.datetime, 
                       current_price: float, 
                       atr_1h: float) -> Dict[str, np.ndarray]:
        """
        Generates Probabilistic Target Tensors.
        Instead of 'Distance', we give the AI a 'Probability Density' map.
        """
        # 1. Macro Cycle Logic
        macro = self.macro_tracker.get_macro_state(timestamp_utc)
        
        # 2. Price Projections (The 'Mu' - Mean Expectation)
        targets = self.projector.calculate_expansion_targets(current_price, atr_1h)
        
        # 3. Gaussian Probability Field
        # We model the Target as a Zone, not a Point.
        # How likely is it that the price is "drawn" to the target?
        # Physics: Potential Energy increases as we approach?
        # Let's use Normalized Distance (Z-Score concept)
        
        # Distance Scalar
        dist_bull = (targets["target_bull_standard"] - current_price)
        dist_bear = (current_price - targets["target_bear_standard"])
        
        # Gaussian Activation: Peak (1.0) when AT target. Decay as we move away.
        # Sigma = ATR / 2. (Tight Zone)
        sigma = atr_1h / 2.0
        
        prob_bull_zone = np.exp(-0.5 * (dist_bull / sigma)**2)
        prob_bear_zone = np.exp(-0.5 * (dist_bear / sigma)**2)
        
        # But wait, for 'Expansion', we want to know "How much room to run?".
        # If Prob is 1.0 (We are at target), Expansion Potential is low (Distribution).
        # If Prob is low (We are far), Potential is high?
        # User wants: "Understand liqudity cycle as probability".
        
        # Let's provide BOTH:
        # 1. Normalized Room to run (Distance / ATR) -> "Potential Energy"
        # 2. Target Proximity (Gaussian) -> "Magnetic Pull / Arrived"
        
        room_bull_z = dist_bull / atr_1h # 2.0 means 2 ATRs to go.
        
        return {
            "expansion_potential_bull": np.array([room_bull_z], dtype=np.float32),
            "target_proximity_bull": np.array([prob_bull_zone], dtype=np.float32), 
            
            "macro_cycle_probability": np.array([macro.cycle_weight], dtype=np.float32),
            
            # One-Hot Encoding for H4 Candle (1-6)
            "h4_fractal_phase": np.eye(6)[macro.h4_candle_index - 1].astype(np.float32),
        }
