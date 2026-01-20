"""
FEAT SNIPER: SHANNON ENTROPY MODULE
====================================
Calculates information entropy from price series to detect market noise.

High Entropy (> 0.6) = Chaotic/Random market -> BLOCK TRADING
Low Entropy (< 0.4) = Focused/Trending market -> ALLOW TRADING
"""

import numpy as np
from numba import jit
from typing import Tuple


@jit(nopython=True, cache=True)
def _calculate_bins(data: np.ndarray, num_bins: int) -> np.ndarray:
    """JIT-compiled histogram binning for Shannon entropy."""
    if len(data) < 2:
        return np.zeros(num_bins, dtype=np.float64)
    
    min_val = np.min(data)
    max_val = np.max(data)
    
    if max_val == min_val:
        result = np.zeros(num_bins, dtype=np.float64)
        result[0] = len(data)
        return result
    
    bin_width = (max_val - min_val) / num_bins
    counts = np.zeros(num_bins, dtype=np.float64)
    
    for val in data:
        bin_idx = int((val - min_val) / bin_width)
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
        counts[bin_idx] += 1
    
    return counts


@jit(nopython=True, cache=True)
def _shannon_entropy(counts: np.ndarray) -> float:
    """Computes Shannon entropy from bin counts."""
    total = np.sum(counts)
    if total == 0:
        return 0.0
    
    probs = counts / total
    entropy = 0.0
    
    for p in probs:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy


class ShannonEntropyAnalyzer:
    """
    Analyzes market entropy to detect noise vs trend regimes.
    
    Uses rolling window Shannon entropy on price returns.
    Entropy is normalized to [0, 1] range.
    """
    
    def __init__(self, window: int = 50, num_bins: int = 10):
        """
        Args:
            window: Rolling window size for entropy calculation.
            num_bins: Number of histogram bins (affects granularity).
        """
        self.window = window
        self.num_bins = num_bins
        self.max_entropy = np.log2(num_bins)  # Maximum possible entropy
        
    def calculate_returns_entropy(self, prices: np.ndarray) -> float:
        """
        Calculates normalized entropy of price returns.
        
        Args:
            prices: Array of closing prices.
            
        Returns:
            Normalized entropy score [0.0 - 1.0].
            Higher = more random/noisy market.
        """
        # [FIX] Dynamic Warmup: mitigate "hardcoded" feel by allowing calculation on fewer bars
        # Requirement: At least 10 bars to have statistical significance
        min_warmup = 10
        if len(prices) < min_warmup:
            return 0.5  # Neutral if truly insufficient data
        
        # Use available data up to self.window
        # If we have 20 bars, we use 20. If we have 100, we use 50 (self.window).
        lookback = min(len(prices)-1, self.window)
        
        # Calculate returns on the dynamic window
        # [-lookback-1:] gets us lookback+1 prices to compute 'lookback' returns
        subset = prices[-lookback-1:]
        returns = np.diff(subset) / subset[:-1]
        
        # Compute histogram
        counts = _calculate_bins(returns, self.num_bins)
        
        # Compute raw entropy
        raw_entropy = _shannon_entropy(counts)
        
        # Normalize to [0, 1]
        normalized = raw_entropy / self.max_entropy if self.max_entropy > 0 else 0.0
        
        return float(np.clip(normalized, 0.0, 1.0))
    
    def calculate_tick_entropy(self, tick_deltas: np.ndarray) -> float:
        """
        Entropy of tick-by-tick price changes (for HFT).
        
        Args:
            tick_deltas: Array of (bid - previous_bid) or (ask - previous_ask).
            
        Returns:
            Normalized entropy [0.0 - 1.0].
        """
        if len(tick_deltas) < self.window:
            return 0.5
        
        data = tick_deltas[-self.window:]
        counts = _calculate_bins(data, self.num_bins)
        raw_entropy = _shannon_entropy(counts)
        
        return float(np.clip(raw_entropy / self.max_entropy, 0.0, 1.0))
    
    def get_regime_signal(self, entropy: float) -> Tuple[str, bool]:
        """
        Interprets entropy as a trading regime signal.
        
        Returns:
            Tuple of (regime_name, is_tradeable).
        """
        if entropy > 0.7:
            return ("HIGH_NOISE", False)  # Block trading
        elif entropy > 0.6:
            return ("ELEVATED_NOISE", False)  # Block trading
        elif entropy > 0.4:
            return ("NEUTRAL", True)  # Proceed with caution
        else:
            return ("LOW_NOISE", True)  # Ideal for trading
    
    def get_entropy_score(self, prices: np.ndarray) -> dict:
        """
        Full entropy analysis with regime classification.
        
        Returns:
            Dictionary with entropy score and regime info.
        """
        entropy = self.calculate_returns_entropy(prices)
        regime, tradeable = self.get_regime_signal(entropy)
        
        return {
            "entropy_score": round(entropy, 4),
            "regime": regime,
            "tradeable": tradeable,
            "interpretation": "Market is focused" if entropy < 0.4 else 
                            "Market is neutral" if entropy < 0.6 else 
                            "Market is noisy - BLOCK TRADES"
        }


# Singleton for global access
entropy_analyzer = ShannonEntropyAnalyzer(window=50, num_bins=10)
