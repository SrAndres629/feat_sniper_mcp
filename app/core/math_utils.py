"""
FEAT MATH ENGINE - Optimized with Numba
=======================================
High-performance numerical utilities for Market Physics.
"""

import numpy as np
from numba import njit
from scipy.stats import gaussian_kde

@njit
def fast_bin_indices(prices, bin_size, min_price):
    """
    Numba-accelerated pricing bin calculation.
    """
    n = prices.shape[0]
    idx = np.empty(n, dtype=np.int64)
    for i in range(n):
        idx[i] = int(np.floor((prices[i] - min_price) / bin_size))
    return idx

@njit
def bin_volume_numba(prices, volumes, bin_size):
    """
    High-speed Volume Profile binning.
    """
    if len(prices) == 0:
        return np.array([0.0]), np.array([0.0])
    
    min_p = np.min(prices)
    max_p = np.max(prices)
    
    n_bins = int(np.floor((max_p - min_p) / bin_size)) + 1
    counts = np.zeros(n_bins, dtype=np.float64)
    
    for i in range(len(prices)):
        idx = int(np.floor((prices[i] - min_p) / bin_size))
        if idx < n_bins:
            counts[idx] += volumes[i]
            
    centers = (np.arange(n_bins) + 0.5) * bin_size + min_p
    return centers, counts

def calculate_weighted_kde(prices, volumes, grid, bandwidth_factor=1.0):
    """
    Weighted KDE using effective sample size (n_eff).
    """
    if len(prices) < 5:
        return np.zeros_like(grid)
    
    # Effective sample size calculation
    weights = volumes / volumes.sum()
    n_eff = 1.0 / np.sum(weights**2)
    
    # Adaptive Bandwidth (Scott's Rule variation)
    std = np.sqrt(np.cov(prices, aweights=volumes))
    bw = bandwidth_factor * std * (n_eff**(-1./5.))
    
    try:
        kde = gaussian_kde(prices, weights=volumes, bw_method=bw/std)
        return kde(grid)
    except:
        return np.zeros_like(grid)

@njit
def triple_barrier_label(prices, entry_idx, pt_ratio, sl_ratio, horizon):
    """
    Triple Barrier Method for Institutional Labeling.
    - +1: Hits Profit Target
    - -1: Hits Stop Loss
    -  0: Time Horizon Exhausted (Vertical Barrier)
    """
    entry_price = prices[entry_idx]
    upper_barrier = entry_price * (1 + pt_ratio)
    lower_barrier = entry_price * (1 - sl_ratio)
    
    end_idx = min(entry_idx + horizon, len(prices) - 1)
    
    for i in range(entry_idx + 1, end_idx + 1):
        if prices[i] >= upper_barrier:
            return 1.0
        if prices[i] <= lower_barrier:
            return -1.0
            
    # If neither hit, it's a 0 (Time expiration) or we could use price delta
    return 0.0
