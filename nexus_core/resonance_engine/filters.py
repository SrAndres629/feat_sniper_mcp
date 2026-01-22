"""
[MODULE 04 - DOCTORAL DSP]
Zero-Lag Moving Average Filters.
Implements Hull Moving Average (HMA) and Arnaud Legoux Moving Average (ALMA)
for phase-lag elimination in spectral analysis.
"""
import numpy as np
import pandas as pd
from typing import Union

def weighted_ma(series: pd.Series, period: int) -> pd.Series:
    """
    Weighted Moving Average (WMA).
    Weights are linearly distributed: most recent has highest weight.
    """
    if period < 1:
        return series
    
    weights = np.arange(1, period + 1)
    
    def wma_calc(window):
        if len(window) < period:
            return np.nan
        return np.dot(window, weights) / weights.sum()
    
    return series.rolling(window=period).apply(wma_calc, raw=True)


def hull_ma(series: pd.Series, period: int) -> pd.Series:
    """
    [ZERO-LAG] Hull Moving Average (HMA).
    Formula: HMA = WMA(2×WMA(n/2) − WMA(n), √n)
    
    Properties:
    - Eliminates phase lag almost entirely
    - Maintains smoothness
    - Reacts faster to price impulses than EMA
    """
    if period < 2:
        return series
    
    half_period = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    
    # WMA of half period
    wma_half = weighted_ma(series, half_period)
    
    # WMA of full period
    wma_full = weighted_ma(series, period)
    
    # Raw HMA signal: 2 * WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final smoothing with WMA of sqrt(period)
    hma = weighted_ma(raw_hma, sqrt_period)
    
    return hma


def alma(series: pd.Series, period: int = 200, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    """
    [INSTITUTIONAL ANCHOR] Arnaud Legoux Moving Average (ALMA).
    
    Properties:
    - Gaussian-weighted average with configurable offset
    - Reduced lag compared to SMA/EMA at same period
    - Smoother output with less overshoot
    
    Parameters:
    - period: Lookback window
    - offset: Position of Gaussian peak (0-1, higher = more responsive)
    - sigma: Gaussian width (higher = smoother)
    """
    if period < 1:
        return series
    
    # Gaussian weights
    m = offset * (period - 1)
    s = period / sigma
    
    weights = np.array([np.exp(-((i - m) ** 2) / (2 * s * s)) for i in range(period)])
    weights = weights / weights.sum()
    
    def alma_calc(window):
        if len(window) < period:
            return np.nan
        return np.dot(window, weights)
    
    return series.rolling(window=period).apply(alma_calc, raw=True)


def exponential_ma(series: pd.Series, period: int) -> pd.Series:
    """
    Standard EMA for comparison/fallback.
    Included for backward compatibility but NOT recommended for primary use.
    """
    return series.ewm(span=period, adjust=False).mean()


def tema(series: pd.Series, period: int) -> pd.Series:
    """
    Triple Exponential Moving Average (TEMA).
    Formula: TEMA = 3×EMA − 3×EMA(EMA) + EMA(EMA(EMA))
    
    Less lag than EMA but more than HMA. Useful for trend confirmation.
    """
    ema1 = exponential_ma(series, period)
    ema2 = exponential_ma(ema1, period)
    ema3 = exponential_ma(ema2, period)
    
    return 3 * ema1 - 3 * ema2 + ema3
