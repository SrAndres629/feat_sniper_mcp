import numpy as np
import pandas as pd

class HurstExponent:
    """
    [FRACTAL GEOMETRY]
    Calculates the Hurst Exponent (H) to detect market regime.
    
    H < 0.5: Mean Reverting (Range/Chop) -> Asia Trap
    H ~ 0.5: Random Walk (Noise)
    H > 0.5: Persistent (Trend) -> NY Expansion
    """
    
    def __init__(self, min_window: int = 100):
        self.min_window = min_window

    def calculate(self, price_series: pd.Series) -> float:
        """
        Calculates H using the simplified R/S analysis on log returns.
        """
        if len(price_series) < self.min_window:
            return 0.5 # Default to Random Walk if insufficient data
            
        # Log returns
        prices = price_series.values
        returns = np.log(prices[1:] / prices[:-1])
        
        # We need to compute R/S over multiple lag sizes to estimate the slope
        # Simplified operational approximation:
        # H = log(R/S) / log(N) (Basic Mandelbrot relation for a single window)
        # For a robust rolling metric, we can use a fixed massive window or sub-windows.
        
        # Let's use the single window approximation for speed in real-time,
        # but standardized over a lookback.
        
        # 1. Mean Return
        mean_ret = np.mean(returns)
        
        # 2. Cumulative Deviations
        deviations = returns - mean_ret
        cum_deviations = np.cumsum(deviations)
        
        # 3. Range (R)
        R = np.max(cum_deviations) - np.min(cum_deviations)
        
        # 4. Standard Deviation (S)
        S = np.std(returns, ddof=1)
        
        if S == 0:
            return 0.5
            
        RS = R / S
        
        # 5. Hurst (Approx)
        # RS ~ (N/2)^H
        # log(RS) = H * log(N/2)
        # H = log(RS) / log(N/2)
        
        N = len(returns)
        try:
            h_val = np.log(RS) / np.log(N / 2)
            # Clip bound [0, 1]
            return max(0.0, min(1.0, float(h_val)))
        except:
            return 0.5
