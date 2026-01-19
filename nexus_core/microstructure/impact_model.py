import numpy as np
import pandas as pd
from typing import Dict, Tuple

class PriceImpactModel:
    """
    [MICROSTRUCTURE PHYSICS]
    Models the 'Cost of Liquidity'.
    Formula: Delta_P = eta * V^alpha
    
    Where:
    - eta (η): The Impact Coefficient (Illiquidity Proxy).
    - alpha (α): The Concavity Factor (usually 0.5 - 0.7).
    """
    
    def __init__(self, window_size: int = 20, fixed_alpha: float = 0.6):
        self.window_size = window_size
        self.alpha = fixed_alpha 
        self.eta_history = []

    def calculate_impact_coefficient(self, close_prices: pd.Series, volumes: pd.Series) -> float:
        """
        Estimates 'eta' (Liquidity Cost) over a rolling window.
        High Eta = Thin Liquidity (Price moves easily on low vol).
        Low Eta = Deep Liquidity (Absorbs volume).
        """
        if len(close_prices) < self.window_size:
            return 0.0
            
        # Get recent window
        recent_closes = close_prices.iloc[-self.window_size:].values
        recent_vols = volumes.iloc[-self.window_size:].values
        
        # Calculate Delta P (Absolute returns)
        delta_p = np.abs(np.diff(recent_closes))
        
        # Calculate V^alpha (exclude last volume to match diff length)
        # Note: diff reduces length by 1, so we take vols[1:]
        # Actually aligned: DeltaP[i] = P[i] - P[i-1]. Corresponds to Volume[i]?
        # Usually Impact is contemporaneous. P[t]-P[t-1] caused by V[t].
        v_pow = np.power(recent_vols[1:], self.alpha)
        
        # Avoid division by zero
        v_pow = np.where(v_pow == 0, 1.0, v_pow)
        
        # Eta = Delta_P / V^alpha
        # We take the median or mean of this ratio for robustness
        raw_etas = delta_p / v_pow
        
        # Clean infinite values
        raw_etas = raw_etas[np.isfinite(raw_etas)]
        
        if len(raw_etas) == 0:
            return 0.0
            
        estimated_eta = float(np.median(raw_etas))
        
        # Normalize/Clip for system sanity (e.g. 0 to 10 scale if needed, for now raw)
        return estimated_eta

    def estimate_slippage(self, volume: float, current_eta: float) -> float:
        """
        Predicts price movement for a given volume.
        """
        return current_eta * (volume ** self.alpha)
