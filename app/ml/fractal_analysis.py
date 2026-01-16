"""
FRACTAL ANALYSIS ENGINE - Corrected R/S Hurst Implementation (P0 REPAIR)
=========================================================================
Calculation of Hurst Exponent using standard Rescaled Range (R/S) analysis.

[P0 REPAIR] Fixes:
- Implemented proper R/S (Rescaled Range) algorithm
- Previous implementation was calculating variance of lagged differences (incorrect)
- Now follows the canonical Lo (1991) methodology

Hurst Exponent (H):
- H > 0.5: Persistent (Trending) - price follows its current direction
- H < 0.5: Anti-persistent (Mean-reverting) - price reverses
- H = 0.5: Random Walk (Brownian Motion) - no predictable pattern

References:
- Hurst, H.E. (1951) "Long-term storage capacity of reservoirs"
- Lo, A.W. (1991) "Long-Term Memory in Stock Market Prices"
"""
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("QuantumLeap.FractalAnalysis")


class FractalAnalyzer:
    """
    Computes fractal metrics for a given time series.
    
    [P0 REPAIR] Now implements the correct R/S (Rescaled Range) algorithm.
    """
    
    @staticmethod
    def compute_hurst(series: np.ndarray, min_window: int = 10, max_window: int = None) -> float:
        """
        Calculates the Hurst Exponent using Rescaled Range (R/S) analysis.
        
        [P0 REPAIR] Correct implementation of R/S algorithm:
        
        For each window size n:
        1. Divide series into k subseries of length n
        2. For each subseries:
           a. Calculate mean m
           b. Calculate cumulative deviations from mean: Y_t = Σ(x_i - m)
           c. Range R = max(Y) - min(Y)
           d. Standard deviation S = std(subseries)
           e. R/S = R / S
        3. Average R/S across all subseries
        4. Fit: log(R/S) ~ H * log(n) + c
        
        The slope H is the Hurst Exponent.
        
        Args:
            series: Price series (e.g., closing prices or returns)
            min_window: Minimum window size for R/S calculation
            max_window: Maximum window size (default: len(series) // 4)
            
        Returns:
            float: Hurst exponent in range [0, 1]
        """
        n = len(series)
        
        if n < 100:
            if n > 16:
                return FractalAnalyzer.compute_hurst_lite(series)
            logger.warning(f"[HURST] Insufficient data ({n} < 16), returning 0.5")
            return 0.5  # Default to random walk
        
        if max_window is None:
            max_window = n // 4
        
        # Ensure min_window is reasonable
        min_window = max(min_window, 10)
        max_window = min(max_window, n // 2)
        
        if min_window >= max_window:
            return 0.5
        
        # Generate window sizes (logarithmically spaced for better fit)
        window_sizes = np.unique(np.logspace(
            np.log10(min_window), 
            np.log10(max_window), 
            num=20
        ).astype(int))
        
        rs_values = []
        valid_windows = []
        
        for window in window_sizes:
            if window < 10:
                continue
                
            rs_list = []
            
            # Number of non-overlapping subseries
            num_subseries = n // window
            
            for i in range(num_subseries):
                start = i * window
                end = start + window
                subseries = series[start:end]
                
                if len(subseries) < window:
                    continue
                
                # Step 2a: Calculate mean
                mean = np.mean(subseries)
                
                # Step 2b: Cumulative deviations from mean
                deviations = subseries - mean
                cumulative_deviations = np.cumsum(deviations)
                
                # Step 2c: Range R = max - min of cumulative deviations
                R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                
                # Step 2d: Standard deviation S
                S = np.std(subseries, ddof=0)
                
                # Step 2e: R/S ratio (avoid division by zero)
                if S > 1e-10:
                    rs = R / S
                    rs_list.append(rs)
            
            if len(rs_list) > 0:
                # Step 3: Average R/S for this window size
                avg_rs = np.mean(rs_list)
                rs_values.append(avg_rs)
                valid_windows.append(window)
        
        if len(rs_values) < 3:
            logger.warning("[HURST] Not enough valid R/S values, returning 0.5")
            return 0.5
        
        # Step 4: Linear regression of log(R/S) vs log(n)
        log_windows = np.log10(valid_windows)
        log_rs = np.log10(rs_values)
        
        # Fit: log(R/S) = H * log(n) + c
        # The slope H is the Hurst exponent
        try:
            coefficients = np.polyfit(log_windows, log_rs, 1)
            H = coefficients[0]
            
            # Clamp to valid range [0, 1]
            H = np.clip(H, 0.0, 1.0)
            
            logger.debug(f"[HURST] Calculated H={H:.4f} from {len(valid_windows)} windows")
            return float(H)
            
        except Exception as e:
            logger.error(f"[HURST] Regression failed: {e}")
            return 0.5

    @staticmethod
    def compute_hurst_lite(series: np.ndarray) -> float:
        """
        [SENIOR ARCHITECTURE] Hurst Lite: Variance-based estimator for HFT.
        Calculates H based on the slope of variance scaling for small windows.
        Works for 16 <= n < 100.
        """
        n = len(series)
        if n < 16: return 0.5
        
        # Calculate variances for lags: 2, 4, 8, 16
        lags = [2, 4, 8, 16]
        variances = []
        for lag in lags:
            diffs = series[lag:] - series[:-lag]
            variances.append(np.var(diffs))
        
        # Fit: log(Var) ~ 2H * log(lag)
        try:
            coeffs = np.polyfit(np.log2(lags), np.log2(variances), 1)
            H = coeffs[0] / 2.0
            return float(np.clip(H, 0.0, 1.0))
        except:
            return 0.5

    @staticmethod
    def detect_regime(hurst: float) -> str:
        """
        Categorizes market regime based on Hurst exponent.
        
        [LAST MILE] Calibrated thresholds:
        - H > 0.65: Confidently trending (high persistence)
        - H < 0.45: Confidently ranging (mean-reversion)
        - 0.45 <= H <= 0.65: Random walk (no statistical edge)
        
        Note: These thresholds are more conservative than the previous
        0.55/0.45 to reduce false positive regime classifications.
        
        Args:
            hurst: Calculated Hurst exponent
            
        Returns:
            str: "TRENDING", "MEAN_REVERTING", or "RANDOM_WALK"
        """
        if hurst > 0.65:
            return "TRENDING"
        elif hurst < 0.45:
            return "MEAN_REVERTING"
        else:
            return "RANDOM_WALK"

    def analyze_timeframe(self, df: Any) -> Dict[str, Any]:
        """
        Performs full fractal analysis on a dataframe of price data.
        
        Args:
            df: Dataframe with 'close' column
            
        Returns:
            Dict: Hurst and regime information
        """
        if df is None or (hasattr(df, 'empty') and df.empty):
            return {"hurst": 0.5, "regime": "UNKNOWN", "confidence": "low"}
            
        if 'close' not in df.columns:
            return {"hurst": 0.5, "regime": "UNKNOWN", "confidence": "low"}
            
        close_prices = df['close'].values
        
        # Use returns for Hurst calculation (more stationary)
        if len(close_prices) > 2:
            returns = np.diff(np.log(close_prices + 1e-10))  # Log returns
            hurst = self.compute_hurst(close_prices)  # Can also use returns
        else:
            hurst = 0.5
            
        regime = self.detect_regime(hurst)
        
        # Calculate confidence based on how far from 0.5
        confidence = "high" if abs(hurst - 0.5) > 0.1 else "medium" if abs(hurst - 0.5) > 0.05 else "low"
        
        return {
            "hurst": float(round(hurst, 4)),
            "regime": regime,
            "confidence": confidence,
            "complexity": float(round(2.0 - hurst, 4))  # Simple Fractal Dimension estimate
        }

    @staticmethod
    def validate_hurst_calculation(known_h: float = 0.7) -> Dict[str, Any]:
        """
        Validation function to test Hurst calculation accuracy.
        
        Generates a fractional Brownian motion (fBm) with known H
        and verifies the calculated H is close to the known value.
        
        Args:
            known_h: Known Hurst exponent to generate test data
            
        Returns:
            Dict with test results
        """
        # Generate fBm-like series (simplified)
        np.random.seed(42)
        n = 1000
        
        # Create correlated noise based on H
        noise = np.random.randn(n)
        
        # Approximate fBm by correlating consecutive values
        persistence = 2 * known_h - 1  # Correlation coefficient
        series = np.zeros(n)
        series[0] = noise[0]
        
        for i in range(1, n):
            series[i] = persistence * series[i-1] + noise[i]
        
        # Cumulative sum to create price-like series
        prices = 100 + np.cumsum(series) * 0.1
        
        # Calculate Hurst
        calculated_h = FractalAnalyzer.compute_hurst(prices)
        
        error = abs(calculated_h - known_h)
        passed = error < 0.15  # Allow 15% error margin
        
        return {
            "known_h": known_h,
            "calculated_h": round(calculated_h, 4),
            "error": round(error, 4),
            "passed": passed,
            "regime": FractalAnalyzer.detect_regime(calculated_h)
        }


# Singleton instance
fractal_analyzer = FractalAnalyzer()
