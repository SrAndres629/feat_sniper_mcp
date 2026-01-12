"""
FEAT NEXUS: LABELING SYSTEM (Triple-Barrier Method)
===================================================
Implementation of Marcos Lopez de Prado's labeling for robust financial ML.
"""

import numpy as np
from numba import njit
from typing import Tuple, List

@njit(cache=True)
def apply_triple_barrier(
    prices: np.ndarray, 
    entry_idx: int, 
    pt_ratio: float, 
    sl_ratio: float, 
    horizon: int
) -> float:
    """
    Core Triple Barrier Logic.
    1.0: Profit Target hit first.
    -1.0: Stop Loss hit first.
    0.0: Time horizon reached.
    """
    if entry_idx >= len(prices):
        return 0.0
        
    entry_price = prices[entry_idx]
    upper_barrier = entry_price * (1 + pt_ratio)
    lower_barrier = entry_price * (1 - sl_ratio)
    
    end_idx = min(entry_idx + horizon, len(prices) - 1)
    
    for i in range(entry_idx + 1, end_idx + 1):
        if prices[i] >= upper_barrier:
            return 1.0 # Success (Bullish)
        elif prices[i] <= lower_barrier:
            return -1.0 # Failure (Bearish)
            
    return 0.0 # Neutral/Time-out

def label_dataset_triple_barrier(
    prices: np.ndarray, 
    indices: np.ndarray, 
    pt: float = 0.005, 
    sl: float = 0.005, 
    h: int = 100
) -> np.ndarray:
    """
    Labels a set of entry points using the Triple Barrier protocol.
    """
    labels = np.zeros(len(indices))
    for i, idx in enumerate(indices):
        labels[i] = apply_triple_barrier(prices, idx, pt, sl, h)
    return labels

def get_labeling_parameters(atr_value: float, volatility_multiplier: float = 2.0) -> dict:
    """
    Dynamically adjust barriers based on ATR.
    """
    return {
        "pt_ratio": (atr_value * volatility_multiplier) / 2000.0, # Approximate scaling
        "sl_ratio": (atr_value * volatility_multiplier) / 2000.0,
        "horizon": 100 # Default bars
    }
