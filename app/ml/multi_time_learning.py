"""
MULTI-TIME LEARNING MANAGER (MIP v6.0)
======================================
Manages multitemporal datasets, hierarchical weights, and fractal alignment.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from app.ml.data_collector import TIMEFRAMES, TIMEFRAME_MAP

logger = logging.getLogger("QuantumLeap.MultiTimeLearning")

class MultiTimeLearningManager:
    """Orchestrates datasets and dynamic weights for MTF Intelligence."""
    
    def __init__(self):
        # Hierarchical Weights: Macro is heavy, Micro is sensitive
        self.weights = {
            "M1": 0.10,   # Timing
            "M5": 0.15,   # Momentum
            "M15": 0.15,
            "H1": 0.20,   # Trend Bias
            "H4": 0.25,   # Structural Bias
            "D1": 0.15    # Global Bias
        }

    def get_fractal_weights(self, hurst_map: Dict[str, float]) -> Dict[str, float]:
        """Adjusts weights dynamically based on market physics (Hurst).
        
        If a timeframe is highly trending (H > 0.6), its weight increases.
        """
        dynamic_weights = self.weights.copy()
        for tf, h in hurst_map.items():
            if h > 0.6: # Strong Trend
                dynamic_weights[tf] *= 1.2
            elif h < 0.4: # Mean Reversion
                dynamic_weights[tf] *= 0.8
                
        # Normalize weights to sum to 1.0
        total = sum(dynamic_weights.values())
        return {k: v / total for k, v in dynamic_weights.items()}

    def resolve_conflicts(self, signals: Dict[str, float], hurst_map: Dict[str, float]) -> float:
        """Applies Fusion Layer logic to generate a final Trade Probability.
        
        Args:
            signals: Dict of [TF: probability].
            hurst_map: Dict of [TF: Hurst Exponent].
            
        Returns:
            float: Final consolidated probability.
        """
        weights = self.get_fractal_weights(hurst_map)
        final_prob = 0.0
        
        for tf, prob in signals.items():
            final_prob += prob * weights.get(tf, 0.0)
            
        # Global Bias Filter: If D1/H4 is deeply inverse, dampen signal
        macro_bias = (signals.get("D1", 0.5) + signals.get("H4", 0.5)) / 2
        if abs(macro_bias - 0.5) > 0.2: # Strong Macro Bias
            if (macro_bias > 0.5 and final_prob < 0.5) or (macro_bias < 0.5 and final_prob > 0.5):
                logger.warning("FUSION: Macro Veto active. Signal conflicts with structural bias.")
                final_prob = 0.5 + (final_prob - 0.5) * 0.5 # Dampen towards neutrality
                
        return final_prob

# Singleton
mtf_manager = MultiTimeLearningManager()
