"""
MULTI-TIME LEARNING MANAGER (MIP v7.0 - Killzone Aware)
========================================================
Manages multitemporal datasets, hierarchical weights, and fractal alignment.

UPGRADES v7.0:
- Dynamic H4 weight boost when close is in killzone (0.25 → 0.40)
- Killzone-aware fusion layer
- Integration with temporal features
- Confirmation window logic
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

logger = logging.getLogger("QuantumLeap.MultiTimeLearning")


# =============================================================================
# H4 CONFIRMATION WINDOWS (Bolivia Time)
# =============================================================================

H4_KILLZONE_OVERLAP = {
    # H4 candles that include significant killzone portions
    # Format: H4_close_hour → contains_killzone, weight_boost
    8: {"contains_killzone": True, "weight_boost": 0.10, "note": "Ends at 08:00, includes 04-08 London"},
    12: {"contains_killzone": True, "weight_boost": 0.15, "note": "Ends at 12:00, includes 09-12 overlap ⭐"},
    16: {"contains_killzone": False, "weight_boost": 0.0, "note": "Ends at 16:00, NY afternoon"},
    20: {"contains_killzone": False, "weight_boost": 0.0, "note": "Ends at 20:00, includes globex open"},
    0: {"contains_killzone": False, "weight_boost": 0.0, "note": "Ends at 00:00, Asia"},
    4: {"contains_killzone": False, "weight_boost": 0.0, "note": "Ends at 04:00, pre-London"},
}


class MultiTimeLearningManager:
    """
    Orchestrates datasets and dynamic weights for MTF Intelligence.
    
    Now with killzone awareness for H4 weight boosting.
    """
    
    def __init__(self):
        # Base Hierarchical Weights
        self.base_weights = {
            "M1": 0.10,   # Timing
            "M5": 0.15,   # Momentum
            "M15": 0.15,
            "H1": 0.20,   # Trend Bias
            "H4": 0.25,   # Structural Bias
            "D1": 0.15    # Global Bias
        }
        
        # Current dynamic weights (updated by killzone context)
        self.weights = self.base_weights.copy()
        
        # Killzone context
        self.in_killzone = False
        self.h4_close_in_killzone = False
        
    def update_killzone_context(
        self,
        in_killzone: bool,
        h4_close_in_killzone: bool = False,
        current_hour: int = None
    ):
        """
        Update weights based on killzone context.
        
        If H4 close is within killzone (09:00-13:00 Bolivia),
        boost H4 weight significantly.
        """
        self.in_killzone = in_killzone
        self.h4_close_in_killzone = h4_close_in_killzone
        
        # Reset to base weights
        self.weights = self.base_weights.copy()
        
        # Boost H4 if close is in killzone
        if h4_close_in_killzone:
            # H4 close during overlap = institutional authority
            boost = 0.15  # 0.25 → 0.40
            self.weights["H4"] += boost
            
            # Reduce micro timeframes to compensate
            self.weights["M1"] -= boost / 3
            self.weights["M5"] -= boost / 3
            self.weights["M15"] -= boost / 3
            
            logger.info(f"[MTF] H4 weight boosted to {self.weights['H4']:.2f} (killzone close)")
        
        # Normalize
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        
    def get_fractal_weights(self, hurst_map: Dict[str, float]) -> Dict[str, float]:
        """
        Adjusts weights dynamically based on market physics (Hurst).
        
        If a timeframe is highly trending (H > 0.6), its weight increases.
        """
        dynamic_weights = self.weights.copy()
        
        for tf, h in hurst_map.items():
            if h > 0.6:  # Strong Trend
                dynamic_weights[tf] *= 1.2
            elif h < 0.4:  # Mean Reversion
                dynamic_weights[tf] *= 0.8
                
        # Normalize weights to sum to 1.0
        total = sum(dynamic_weights.values())
        return {k: v / total for k, v in dynamic_weights.items()}
    
    def calculate_alignment_score(
        self,
        d1_direction: str,
        h4_direction: str,
        h1_direction: str
    ) -> Dict[str, Any]:
        """
        Calculate alignment score for size multiplier.
        
        Returns:
            Dict with score, size_multiplier, and alignment_type
        """
        def dir_to_val(d):
            d = d.upper() if d else "NEUTRAL"
            if d == "BULLISH": return 1.0
            if d == "BEARISH": return -1.0
            return 0.0
        
        d1_val = dir_to_val(d1_direction)
        h4_val = dir_to_val(h4_direction)
        h1_val = dir_to_val(h1_direction)
        
        # Weighted alignment factor
        alignment_factor = (
            d1_val * self.weights.get("D1", 0.15) +
            h4_val * self.weights.get("H4", 0.25) +
            h1_val * self.weights.get("H1", 0.20)
        ) / (self.weights.get("D1", 0.15) + self.weights.get("H4", 0.25) + self.weights.get("H1", 0.20))
        
        # Full alignment check
        all_aligned = (d1_val == h4_val == h1_val) and d1_val != 0
        
        # Conflict check
        has_conflict = (d1_val * h4_val < 0) or (h4_val * h1_val < 0)
        
        # Size multiplier
        if all_aligned:
            size_mult = 1.0
            alignment_type = "FULL"
        elif has_conflict:
            size_mult = 0.25
            alignment_type = "CONFLICT"
        elif abs(alignment_factor) > 0.5:
            size_mult = 0.75
            alignment_type = "MOSTLY_ALIGNED"
        elif h4_val != 0 and d1_val == 0:
            size_mult = 0.50
            alignment_type = "H4_ONLY"
        else:
            size_mult = 0.50
            alignment_type = "PARTIAL"
        
        return {
            "alignment_factor": round(alignment_factor, 3),
            "alignment_type": alignment_type,
            "size_multiplier": size_mult,
            "all_aligned": all_aligned,
            "has_conflict": has_conflict,
            "direction": "BULLISH" if alignment_factor > 0.2 else ("BEARISH" if alignment_factor < -0.2 else "NEUTRAL")
        }

    def resolve_conflicts(
        self,
        signals: Dict[str, float],
        hurst_map: Dict[str, float],
        h4_in_killzone: bool = False
    ) -> float:
        """
        Applies Fusion Layer logic to generate a final Trade Probability.
        
        Args:
            signals: Dict of [TF: probability].
            hurst_map: Dict of [TF: Hurst Exponent].
            h4_in_killzone: If H4 close was in killzone, boost its weight.
            
        Returns:
            float: Final consolidated probability.
        """
        # Update context if H4 in killzone
        if h4_in_killzone and not self.h4_close_in_killzone:
            self.update_killzone_context(True, True)
        
        weights = self.get_fractal_weights(hurst_map)
        final_prob = 0.0
        
        for tf, prob in signals.items():
            final_prob += prob * weights.get(tf, 0.0)
            
        # Global Bias Filter: Macro Veto
        macro_bias = (signals.get("D1", 0.5) + signals.get("H4", 0.5)) / 2
        
        if abs(macro_bias - 0.5) > 0.2:  # Strong Macro Bias
            if (macro_bias > 0.5 and final_prob < 0.5) or (macro_bias < 0.5 and final_prob > 0.5):
                logger.warning("FUSION: Macro Veto active. Signal conflicts with structural bias.")
                
                # Dampen towards neutrality (but less if in killzone)
                dampen_factor = 0.6 if self.in_killzone else 0.5
                final_prob = 0.5 + (final_prob - 0.5) * dampen_factor
        
        # Killzone boost
        if self.in_killzone and final_prob > 0.5:
            final_prob = min(0.95, final_prob * 1.05)  # Small boost
            
        return final_prob
    
    def get_confirmation_requirements(
        self,
        timeframe: str,
        in_killzone: bool = False
    ) -> Dict[str, Any]:
        """
        Get confirmation requirements for a timeframe signal.
        
        Based on domain knowledge:
        - H1: Close + 5-30min retest + vol ≥1.3x
        - H4: Close in killzone = instant confirmation, else wait 4h
        - D1: Confirm in next killzone (09:00-13:00)
        """
        if timeframe == "H1":
            return {
                "wait_minutes_min": 5,
                "wait_minutes_max": 30,
                "volume_threshold": 1.3,
                "requires_retest": True,
                "killzone_reduces_wait": in_killzone
            }
        elif timeframe == "H4":
            return {
                "wait_hours": 0 if in_killzone else 4,
                "confirm_with_h1_count": 0 if in_killzone else 4,
                "killzone_close_is_confirmation": in_killzone
            }
        elif timeframe == "D1":
            return {
                "wait_for_next_killzone": True,
                "killzone_hours": [9, 10, 11, 12, 13],
                "fix_adds_confidence": True
            }
        else:
            return {"wait_minutes": 15, "volume_threshold": 1.2}


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

mtf_manager = MultiTimeLearningManager()
