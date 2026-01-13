"""
MARKET PHYSICS ENGINE (FEAT NEXUS PRIME) - v2.0
==============================================
Calculates the "Physics" of the market: Mass (Volume), Velocity (Displacement),
and Energy (Liquidity).

Upgrades (PVP FEAT):
1. CVD (Cumulative Volume Delta) & Flow Dynamics.
2. Normalized POC Delta (Z-Score & ATR Adjusted).
3. KDE (Kernel Density Estimation) for smoothed profiles.
4. Breakout Probability Tensors.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from scipy.stats import gaussian_kde
from app.core.math_utils import bin_volume_numba, calculate_weighted_kde

logger = logging.getLogger("FEAT.MarketPhysics")

class MarketPhysicsEngine:
    """
    Core engine for calculating market physics metrics.
    Uses pandas/numpy for vectorized performance.
    """
    
    def __init__(self):
        self.acceptance_threshold = 0.6
        self.probe_threshold = 0.3

    # =========================================================================
    # 1. PVP FEAT (Advanced Volume Profile)
    # =========================================================================
    
    def calculate_pvp_feat(self, df: pd.DataFrame, use_kde: bool = True) -> Dict[str, Any]:
        """
        PVP FEAT v2.2: Numba-accelerated profile with Centered Value Area.
        """
        if df.empty or len(df) < 5:
            return {}

        prices = df['close'].to_numpy()
        volumes = df['volume'].to_numpy()
        current_price = prices[-1]
        
        # 1. Faster Binning with Numba
        bin_size = max(0.0001, (df['high'].max() - df['low'].min()) / 50)
        bins, vol_profile = bin_volume_numba(prices, volumes, bin_size)
        
        total_vol = vol_profile.sum()
        poc_idx = np.argmax(vol_profile)
        poc_price = bins[poc_idx]

        # 2. KDE Smoothing (Weighted)
        if use_kde:
            kde_density = calculate_weighted_kde(prices, volumes, bins)
            kde_poc_idx = np.argmax(kde_density)
            kde_poc = bins[kde_poc_idx]
        else:
            kde_poc = poc_price
            kde_density = vol_profile / (total_vol + 1e-9)

        # 3. Value Area Centered on POC
        dist_to_poc = np.abs(bins - poc_price)
        # Sort bins by distance (Secondary sort: volume descending)
        sort_idx = np.lexsort((-vol_profile, dist_to_poc))
        
        cum_vol = 0
        va_bins = []
        for idx in sort_idx:
            cum_vol += vol_profile[idx]
            va_bins.append(bins[idx])
            if cum_vol >= 0.70 * total_vol:
                break
        
        vah, val = max(va_bins), min(va_bins)
        
        # 4. Normalized Delta (Z-Score)
        vol_weighted_mean = np.average(prices, weights=volumes)
        vol_weighted_std = np.sqrt(np.average((prices - vol_weighted_mean)**2, weights=volumes))
        z_score = (current_price - poc_price) / (vol_weighted_std + 1e-9)
        
        return {
            "poc": float(poc_price),
            "kde_poc": float(kde_poc),
            "vah": float(vah),
            "val": float(val),
            "z_score": float(z_score),
            "total_volume": float(total_vol),
            "skew": self._calculate_skew(vol_profile, poc_idx),
            "current_rel_pos": (current_price - val) / (vah - val + 1e-9)
        }
    
    def _calculate_skew(self, vol_profile: np.ndarray, poc_idx: int) -> float:
        vol_above = vol_profile[poc_idx+1:].sum()
        vol_below = vol_profile[:poc_idx].sum()
        return (vol_above - vol_below) / (vol_above + vol_below + 1e-9)

    # =========================================================================
    # 2. CVD DYNAMICS (Cumulative Volume Delta)
    # =========================================================================

    def calculate_cvd_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        BVC (Bulk Volume Classification): Refined CVD calculation 
        with noise filtering and tick size estimation.
        """
        if df.empty: return {}
        
        # 1. BVC Improved Tick Sign (with noise filter)
        price_change = df['close'].diff()
        std_diff = price_change.std()
        tick_filter = std_diff * 0.1 # Filter out 10% of volatility as noise
        
        tick_sign = np.where(np.abs(price_change) > tick_filter, np.sign(price_change), 0)
        
        # Fallback to mid-candle for zeros (only if move is zero but candle has body)
        tick_sign = np.where(
            tick_sign == 0, 
            np.sign(df['close'] - df['open']), 
            tick_sign
        )
        
        deltas = tick_sign * df['volume']
        cvd = deltas.cumsum()
        
        # 2. Flow Dynamics
        velocity = cvd.diff(3).fillna(0)
        acceleration = velocity.diff(3).fillna(0)
        
        # 3. Imbalance Ratio
        total_vol = df['volume'].rolling(window=14).sum()
        imbalance_ratio = cvd.diff(14) / (total_vol + 1e-9)

        return {
            "cvd": float(cvd.iloc[-1]),
            "cvd_velocity": float(velocity.iloc[-1]),
            "cvd_acceleration": float(acceleration.iloc[-1]),
            "imbalance_ratio": float(imbalance_ratio.iloc[-1])
        }

    def detect_absorption(self, df: pd.DataFrame) -> float:
        """
        Calculates the Absorption Tension: [High Volume / Narrow Range].
        1.0 = Pure Absorption, 0.0 = Normal expansion.
        """
        if len(df) < 20: return 0.0
        
        # Volatility normalization
        candle_range = (df['high'] - df['low'])
        avg_range = candle_range.rolling(20).mean()
        
        # Volume Spike
        avg_vol = df['volume'].rolling(20).mean()
        std_vol = df['volume'].rolling(20).std()
        vol_spike = (df['volume'] > avg_vol + 1.5 * std_vol)
        
        # Narrow Range (Compressing)
        narrow_range = (candle_range < 0.6 * avg_range)
        
        is_absorbed = (vol_spike & narrow_range)
        
        # Return a "tension" score based on the last 5 bars
        return float(is_absorbed.tail(5).mean())

    def calculate_energy_map(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        FEAT ENERGY MAP: Synthesizes Volume, Volatility, and Flow.
        E = V * (Momentum/ATR) * CVD_Bias
        """
        if len(df) < 14: return {"total_energy": 0, "hotspots": []}
        
        # 1. Flux Intensity
        cvd_data = self.calculate_cvd_metrics(df)
        momentum = abs(cvd_data["cvd_velocity"])
        
        # 2. Volatility Normalizer
        from app.skills.indicators import get_technical_indicator, IndicatorRequest
        atr = get_technical_indicator(df, IndicatorRequest(symbol="SYM", indicator="ATR")).get("value", 1.0)
        
        # 3. Energy Composite
        vol_norm = df['volume'].iloc[-1] / (df['volume'].rolling(20).mean().iloc[-1] + 1e-9)
        energy_score = vol_norm * (momentum / (atr + 1e-9)) * (1 + abs(cvd_data["imbalance_ratio"]))
        
        # 4. Absorption state
        absorption = self.detect_absorption(df)
        
        return {
            "energy_score": float(energy_score),
            "absorption_tension": absorption,
            "imbalance_bias": cvd_data["imbalance_ratio"],
            "state": "HOT" if energy_score > 2.0 else "COLD"
        }

    # =========================================================================
    # 3. BREAKOUT PROBABILITY TENSOR
    # =========================================================================

    def estimate_breakout_probability(self, pvp: Dict, cvd: Dict, atr: float) -> Dict[str, float]:
        """
        Heuristic / Probabilistic model for Breakout.
        P(breakout) = f(Dist to POC, CVD Imbalance, Z-Score)
        """
        if not pvp or not cvd: return {"p_up": 0.5, "p_down": 0.5}

        # Z-Score Impact: High Z-Score (>2) suggests overextension or breakout initiative
        # CVD Impact: Positive acceleration confirms buyer aggression
        
        p_up = 0.5
        p_down = 0.5
        
        # Logic: If price is near VAH and CVD Acceleration is positive -> P(Breakout Up) increases
        if pvp["current_rel_pos"] > 0.8: # Near VAH
            if cvd["cvd_acceleration"] > 0:
                p_up += 0.2
            if cvd["imbalance_ratio"] > 0.1:
                p_up += 0.1
        
        elif pvp["current_rel_pos"] < 0.2: # Near VAL
            if cvd["cvd_acceleration"] < 0:
                p_down += 0.2
            if cvd["imbalance_ratio"] < -0.1:
                p_down += 0.1

        # Z-Score normalization: If Z > 2, we are outside standard value
        if abs(pvp["z_score"]) > 2.0:
            p_up *= 1.1 if pvp["z_score"] > 0 else 0.9
            p_down *= 1.1 if pvp["z_score"] < 0 else 0.9

        return {
            "p_up": min(0.95, max(0.05, p_up)),
            "p_down": min(0.95, max(0.05, p_down))
        }

    # =========================================================================
    # 4. MCI & LIQUIDITY (Simplified for Integration)
    # =========================================================================

    def calculate_mci(self, df: pd.DataFrame, lookback: int = 5) -> Dict[str, Any]:
        if len(df) < lookback + 1: return {"mci_score": 0.0, "mci_type": "NEUTRAL"}
        
        last = df.iloc[-1]
        prev = df.iloc[-(lookback+1):-1]
        
        recent_high = prev['high'].max()
        recent_low = prev['low'].min()
        
        is_high_sweep = last['high'] > recent_high and last['close'] < recent_high
        is_low_sweep = last['low'] < recent_low and last['close'] > recent_low
        
        mci_type = "NEUTRAL"
        if is_high_sweep: mci_type = "BEARISH_SWEEP"
        elif is_low_sweep: mci_type = "BULLISH_SWEEP"
        else: mci_type = "CONTINUATION"
        
        return {"mci_score": 0.0, "mci_type": mci_type, "is_sweep": is_high_sweep or is_low_sweep}

    # =========================================================================
    # 5. FRACTAL PHYSICS (Hurst & Entropy)
    # =========================================================================

    def calculate_fractal_metrics(self, df: pd.DataFrame, window: int = 100) -> Dict[str, float]:
        """
        Calculates Fractal efficiency metrics.
        - Hurst Exponent: Persistency (0.5=Random, >0.5=Trend, <0.5=Revert)
        - Shannon Entropy: Information density / Disorder
        """
        if len(df) < window:
             return {"hurst_exponent": 0.5, "shannon_entropy": 0.0}
        
        # 1. Shannon Entropy (Price Distribution)
        # Bins prices to calculate probability distribution
        clean_prices = df['close'].tail(window).values
        hist, bin_edges = np.histogram(clean_prices, bins=20, density=True)
        # Filter non-zero probs
        probs = hist[hist > 0]
        # Normalize to sum to 1 just in case
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log2(probs))
        # Normalize Entropy (0-1 approx)
        max_entropy = np.log2(20)
        norm_entropy = entropy / max_entropy
        
        # 2. Simplified Hurst (R/S Analysis approximation)
        # H = log(R/S) / log(n)
        try:
            returns = np.diff(np.log(clean_prices))
            if len(returns) < 2: return {"hurst_exponent": 0.5, "shannon_entropy": norm_entropy}
            
            # Divide into chunks? No, simple rolling window R/S
            # R = Range of Cumulative Deviation
            # S = Std Dev
            
            mean_ret = np.mean(returns)
            cum_dev = np.cumsum(returns - mean_ret)
            R = np.max(cum_dev) - np.min(cum_dev)
            S = np.std(returns)
            
            if S == 0: S = 1e-9
            
            # Using E[R/S] ~ c * n^H -> log(R/S) = log(c) + H*log(n)
            # This is a point-estimation, not full regression, but suitable for realtime stream
            
            # Correct range n
            n = len(returns)
            H = np.log(R/S) / np.log(n)
            
            # Cap H between 0 and 1
            hurst = min(1.0, max(0.0, H))
            
        except Exception as e:
            logger.error(f"Fractal Math Error: {e}")
            hurst = 0.5
            
        return {
            "hurst_exponent": float(hurst),
            "shannon_entropy": float(norm_entropy)
        }

# Singleton
market_physics = MarketPhysicsEngine()

