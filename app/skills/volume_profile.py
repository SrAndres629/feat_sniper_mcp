"""
Volume Profile Skill - wrapper for Numba-Accelerated Math Engine
================================================================
Exposes high-performance Volume Profile calculations to the rest of the system.
Uses nexus_core.math_engine (JIT) for millisecond-latency profiling.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from nexus_core.math_engine import (
    calculate_kde_jit, 
    calculate_moments, 
    classify_distribution_shape, 
    calculate_value_area_fast,
    normalize_tensor_minmax
)
import logging

logger = logging.getLogger("FEAT.VolumeProfile")

class VolumeProfile:
    """
    High-Performance Volume Profile Engine (Liquid Density Edition).
    Uses Kernel Density Estimation (KDE) and JIT for doctoral-grade signals.
    """
    
    def __init__(self):
        self.cache = {}
    
    def get_profile(self, candles: pd.DataFrame, resolution: int = 64) -> Dict[str, Any]:
        """
        Calculates Doctoral Volume Profile using KDE.
        Returns:
            - poc: Exact price peak.
            - vah, val: Value Area boundaries.
            - shape: P, b, or D shape string.
            - profile_tensor: 1D normalized vector for Neural Vision.
        """
        if len(candles) < 10:
            return {}
            
        try:
            prices = candles['close'].values.astype(np.float64)
            volumes = (candles['tick_volume'] if 'tick_volume' in candles.columns else candles['volume']).values.astype(np.float64)
                
            p_min, p_max = np.min(prices), np.max(prices)
            if p_max == p_min: return {}
            
            # 1. Weights and Grid for KDE
            weights = volumes / (volumes.sum() + 1e-9)
            grid = np.linspace(p_min, p_max, resolution)
            
            # 2. Adaptive Bandwidth (Scott's Rule approximation)
            std = np.std(prices)
            n_eff = 1.0 / (np.sum(weights**2) + 1e-9)
            bandwidth = std * (n_eff**(-0.2)) # Scott's Rule
            if bandwidth < 1e-9: bandwidth = (p_max - p_min) / resolution
            
            # 3. JIT-Accelerated KDE
            density = calculate_kde_jit(prices, weights, grid, bandwidth)
            
            # 4. Moments and Shape Classification
            mean, std_w, skew, kurt = calculate_moments(grid, density)
            shape_desc = classify_distribution_shape(skew, kurt)
            
            # 5. Peak and Value Area
            peak_idx = np.argmax(density)
            poc_price = grid[peak_idx]
            
            vah_price, val_price = calculate_value_area_fast(grid, density, np.sum(density))
            
            # 6. Neural Tensorization
            profile_tensor = normalize_tensor_minmax(density)
            
            return {
                "poc": poc_price,
                "vah": vah_price,
                "val": val_price,
                "shape": shape_desc,
                "skew": skew,
                "kurtosis": kurt,
                "profile_tensor": profile_tensor,
                "grid": grid,
                "density": density,
                "total_volume": volumes.sum()
            }
            
        except Exception as e:
            logger.error(f"Volume Profile (KDE) Error: {e}")
            return {}

    def get_volume_at_price(self, price: float, profile: Dict[str, Any]) -> float:
        """
        Returns KDE density at a specific price.
        """
        if not profile or 'grid' not in profile: return 0.0
        grid = profile['grid']
        density = profile['density']
        idx = (np.abs(grid - price)).argmin()
        return density[idx]

    def get_zone_quality(self, top: float, bottom: float, profile: Dict[str, Any]) -> float:
        """
        Doctoral Zone Quality: Ratio of local density vs average density.
        """
        if not profile: return 1.0
        midpoint = (top + bottom) / 2
        local_density = self.get_volume_at_price(midpoint, profile)
        avg_density = np.mean(profile['density'])
        if avg_density < 1e-9: return 1.0
        return min(5.0, local_density / avg_density)

# Singleton Export
volume_profile = VolumeProfile()
