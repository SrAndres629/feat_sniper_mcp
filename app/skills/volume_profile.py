"""
Volume Profile Skill - wrapper for Numba-Accelerated Math Engine
================================================================
Exposes high-performance Volume Profile calculations to the rest of the system.
Uses nexus_core.math_engine (JIT) for millisecond-latency profiling.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from nexus_core.math_engine import bin_volume_fast, fast_bin_indices
import logging

logger = logging.getLogger("FEAT.VolumeProfile")

class VolumeProfile:
    """
    High-Performance Volume Profile Engine.
    Wraps nexus_core.math_engine for safe consumption by Structure Engine.
    """
    
    def __init__(self, bins: int = 50):
        self.default_bins = bins
        self.cache = {}
    
    def get_profile(self, candles: pd.DataFrame, bins: int = 50) -> Dict[str, Any]:
        """
        Calculates Volume Profile from DataFrame.
        Returns POC, VAH, VAL, and full profile.
        """
        if len(candles) < 10:
            return {}
            
        try:
            prices = candles['close'].values.astype(np.float64)
            # Use tick_volume if available (Forex), else volume (Crypto)
            if 'tick_volume' in candles.columns:
                volumes = candles['tick_volume'].values.astype(np.float64)
            else:
                volumes = candles['volume'].values.astype(np.float64)
                
            range_min = np.min(prices)
            range_max = np.max(prices)
            
            if range_max == range_min:
                return {}
                
            bin_size = (range_max - range_min) / bins
            
            # --- JIT ACCELERATED CALCULATION ---
            centers, counts = bin_volume_fast(prices, volumes, bin_size)
            # -----------------------------------
            
            # Identify POC (Point of Control)
            poc_idx = np.argmax(counts)
            poc_price = centers[poc_idx]
            max_vol = counts[poc_idx]
            
            # Value Area (70% of volume)
            total_vol = np.sum(counts)
            target_vol = total_vol * 0.70
            
            # Value Area Calculation (Start at POC and expand)
            current_vol = max_vol
            lower_idx = poc_idx
            upper_idx = poc_idx
            
            while current_vol < target_vol:
                # Try expanding down
                next_lower_vol = counts[lower_idx - 1] if lower_idx > 0 else 0
                # Try expanding up
                next_upper_vol = counts[upper_idx + 1] if upper_idx < len(counts) - 1 else 0
                
                if next_lower_vol == 0 and next_upper_vol == 0:
                    break
                    
                if next_lower_vol > next_upper_vol:
                    lower_idx = max(0, lower_idx - 1)
                    current_vol += next_lower_vol
                else:
                    upper_idx = min(len(counts) - 1, upper_idx + 1)
                    current_vol += next_upper_vol
                    
            val_price = centers[lower_idx]
            vah_price = centers[upper_idx]
            
            return {
                "poc": poc_price,
                "vah": vah_price,
                "val": val_price,
                "total_volume": total_vol,
                "profile_centers": centers,
                "profile_counts": counts,
                "max_volume": max_vol
            }
            
        except Exception as e:
            logger.error(f"Volume Profile Calc Error: {e}")
            return {}

    def get_volume_at_price(self, price: float, profile: Dict[str, Any]) -> float:
        """
        Returns estimated liquidity/volume at a specific price level.
        Useful for validating breakdown/breakout zones.
        """
        if not profile or 'profile_centers' not in profile:
            return 0.0
            
        centers = profile['profile_centers']
        counts = profile['profile_counts']
        
        # Find closest bin
        idx = (np.abs(centers - price)).argmin()
        return counts[idx]

    def get_zone_quality(self, top: float, bottom: float, profile: Dict[str, Any]) -> float:
        """
        Returns a score (0.0 - 5.0) indicating how much volume exists in this zone relative to avg.
        Used to multiply zone strength.
        """
        if not profile: return 1.0
        
        midpoint = (top + bottom) / 2
        zone_vol = self.get_volume_at_price(midpoint, profile)
        
        avg_vol = profile['total_volume'] / len(profile['profile_counts'])
        if avg_vol == 0: return 1.0
        
        ratio = zone_vol / avg_vol
        return min(5.0, ratio)

# Singleton Export
volume_profile = VolumeProfile()
