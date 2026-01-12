"""
FRACTAL ANALYSIS ENGINE - Market Physics Hub
============================================
Calculation of Hurst Exponent and Fractal Dimension to detect market regimes.

Hurst Exponent (H):
- H > 0.5: Persistent (Trending)
- H < 0.5: Anti-persistent (Mean-reverting / Range)
- H = 0.5: Random Walk (Brownian Motion / Noise)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("QuantumLeap.FractalAnalysis")

class FractalAnalyzer:
    """Computes fractal metrics for a given time series."""
    
    @staticmethod
    def compute_hurst(series: np.ndarray, max_lags: int = 20) -> float:
        """
        Calculates the Hurst Exponent using Rescaled Range (R/S) analysis.
        
        Args:
            series: Price series (e.g., closing prices).
            max_lags: Maximum number of lags for R/S calculation.
            
        Returns:
            float: Hurst exponent.
        """
        if len(series) < 100:
            return 0.5 # Default to random walk if not enough data
            
        lags = range(2, max_lags)
        
        # Calculate the array of the variances of the lagged differences
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        
        # Use a linear fit to estimate the Hurst Exponent
        poly = np.polyfit(np.log10(lags), np.log10(tau), 1)
        
        return poly[0] * 2.0

    @staticmethod
    def detect_regime(hurst: float) -> str:
        """Categorizes market regime based on Hurst exponent."""
        if hurst > 0.55:
            return "TRENDING"
        elif hurst < 0.45:
            return "MEAN_REVERTING"
        else:
            return "RANDOM_WALK"

    def analyze_timeframe(self, df: pd.DataFrame) -> Dict[str, any]:
        """Performs full fractal analysis on a dataframe of price data.
        
        Args:
            df: Dataframe with 'close' column.
            
        Returns:
            Dict: Hurst and regime information.
        """
        if df.empty or 'close' not in df.columns:
            return {"hurst": 0.5, "regime": "UNKNOWN"}
            
        close_prices = df['close'].values
        hurst = self.compute_hurst(close_prices)
        regime = self.detect_regime(hurst)
        
        return {
            "hurst": float(hurst),
            "regime": regime,
            "complexity": float(2.0 - hurst) # Simple Fractal Dimension estimate
        }

# Singleton
fractal_analyzer = FractalAnalyzer()
