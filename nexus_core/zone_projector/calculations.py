import numpy as np
import pandas as pd

def calculate_dynamic_band(price: float, atr: float, k_factor: float = 0.5) -> tuple:
    """
    [DOCTORAL PHYSICS] Calculates zone boundaries based on Volatility (ATR).
    Width is not fixed in pips, but breathing with the market.
    """
    width = atr * k_factor
    return price + width, price - width

def calculate_mean_threshold(high: float, low: float, open_p: float, close_p: float, kind: str = "WICK") -> float:
    """
    [INSTITUTIONAL GEOMETRY] Calculates the 50% Mean Threshold.
    Valid for Wicks (Shadows) and Order Blocks (Body).
    """
    if kind == "WICK":
        # Determine if upper or lower wick dominates or implies direction
        # Simplified: We usually call this with specific wick coordinates
        return (high + low) / 2
    elif kind == "BODY":
        return (open_p + close_p) / 2
    return (high + low) / 2

def calculate_gaussian_proximity(price: float, zone_center: float, sigma: float) -> float:
    """
    [TENSOR MATH] Radial Basis Function (RBF).
    Returns value 0.0 to 1.0 representing proximity intensity.
    """
    if sigma == 0: return 0.0
    return np.exp(-((price - zone_center)**2) / (2 * (sigma**2)))

def calculate_volume_gravity(zone_vol: float, avg_vol: float) -> float:
    """
    [ENERGY] Relative Volume Gravity.
    """
    if avg_vol == 0: return 1.0
    return min(zone_vol / avg_vol, 5.0) # Cap at 5x gravity
