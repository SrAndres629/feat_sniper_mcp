"""
MARKET PHYSICS ENGINE (FEAT NEXUS PRIME)
========================================
Calculates the "Physics" of the market: Mass (Volume), Velocity (Displacement),
and Energy (Liquidity).

Modules:
1. PVP Vectorial: Dynamic Volume Profile (POC, VAH, VAL).
2. Liquidity Primitives: Mathematical detection of Pools, Imbalances, Sweeps.
3. MCI (Manipulation Confirmation Index): Sweep-to-Displacement Ratio.

"Price is the shadow, Volume is the object."
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger("FEAT.MarketPhysics")

class MarketPhysicsEngine:
    """
    Core engine for calculating market physics metrics.
    Uses pandas/numpy for vectorized performance.
    """
    
    def __init__(self):
        pass

    # =========================================================================
    # 1. PVP VECTORIAL (Dynamic Volume Profile)
    # =========================================================================
    
    def calculate_pvp_vectorial(
        self, 
        df: pd.DataFrame, 
        value_area_pct: float = 0.70, 
        tick_size: float = 0.01
    ) -> Dict[str, float]:
        """
        Calculates the Vectorial PVP (POC, VAH, VAL) for a given candle set.
        Returns dictionary with profile metrics.
        """
        if df.empty or 'close' not in df.columns or 'volume' not in df.columns:
            return None

        # Create bins for volume profile
        try:
            low = df['low'].min()
            high = df['high'].max()
            
            # Ensure we have a valid range
            if high == low:
                high += tick_size
                low -= tick_size
                
            bins = np.arange(
                np.floor(low / tick_size) * tick_size, 
                np.ceil(high / tick_size) * tick_size + tick_size, 
                tick_size
            )
            
            # Digitizing prices to bins
            prices = df['close'].values
            volumes = df['volume'].values
            
            # -1 because digitize returns 1-based index for bins
            bin_indices = np.digitize(prices, bins) - 1
            
            # Aggregate volume per bin
            vol_profile = np.zeros(len(bins))
            for i, v in zip(bin_indices, volumes):
                if 0 <= i < len(vol_profile):
                    vol_profile[i] += v
            
            # Calculate POC (Point of Control)
            total_vol = vol_profile.sum()
            if total_vol == 0:
                return None
                
            poc_idx = np.argmax(vol_profile)
            poc_price = bins[poc_idx]
            
            # Calculate Value Area (VAH, VAL)
            # Sort bins by distance from POC to accumulate central volume first
            bin_centers = bins # approximation
            
            # Using indices to expand out from POC is more accurate for VA usually 
            # but standard implementation sums highest volume nodes. 
            # Let's use the provided logic: accumulate by distance from POC
            
            sorted_indices_by_dist = np.argsort(np.abs(bins - poc_price))
            
            accumulated_vol = 0.0
            included_indices = []
            
            for i in sorted_indices_by_dist:
                if 0 <= i < len(vol_profile):
                    accumulated_vol += vol_profile[i]
                    included_indices.append(i)
                    if accumulated_vol / total_vol >= value_area_pct:
                        break
            
            if not included_indices:
                val_price = poc_price
                vah_price = poc_price
            else:
                included_bins = bins[included_indices]
                val_price = included_bins.min()
                vah_price = included_bins.max()
            
            return {
                "poc": float(poc_price),
                "vah": float(vah_price),
                "val": float(val_price),
                "total_volume": float(total_vol),
                "va_volume_pct": float(accumulated_vol / total_vol),
                "profile_skew": self._calculate_skew(vol_profile, poc_idx)
            }
            
        except Exception as e:
            logger.error(f"Error calculating PVP: {e}")
            return None

    def _calculate_skew(self, vol_profile: np.ndarray, poc_idx: int) -> float:
        """Calculates volume skewness relative to POC."""
        vol_above = vol_profile[poc_idx+1:].sum()
        vol_below = vol_profile[:poc_idx].sum()
        total = vol_above + vol_below + 1e-9
        return (vol_above - vol_below) / total


    # =========================================================================
    # 2. MCI (MANIPULATION CONFIRMATION INDEX)
    # =========================================================================

    def calculate_mci(self, df: pd.DataFrame, lookback: int = 5) -> Dict[str, Any]:
        """
        Calculates Manipulation Confirmation Index.
        Ratio of Sweep Magnitude to Subsequent Displacement.
        
        Hypothesis: 
        - High Sweep + Low Displacement = PASSIVE ABSORPTION (Fakeout/Trap)
        - High Sweep + High Displacement = AGGRESSIVE INITIATIVE (Real Move)
        """
        if len(df) < lookback + 1:
            return {"mci": 0.0, "type": "INSUFFICIENT_DATA"}
            
        # Analyze the recent "event" candle (e.g., the sweep candle)
        # We look at the last closed candle as the potential sweep/breakout event
        last_candle = df.iloc[-1]
        prev_candles = df.iloc[-(lookback+1):-1]
        
        # Detect Swing High/Low in previous window
        recent_high = prev_candles['high'].max()
        recent_low = prev_candles['low'].min()
        
        current_high = last_candle['high']
        current_low = last_candle['low']
        current_close = last_candle['close']
        current_open = last_candle['open']
        
        # Calculate Vectors
        body_size = abs(current_close - current_open)
        upper_wick = current_high - max(current_open, current_close)
        lower_wick = min(current_open, current_close) - current_low
        total_range = current_high - current_low
        
        # Detect Sweep Scenario
        is_high_sweep = current_high > recent_high and current_close < recent_high
        is_low_sweep = current_low < recent_low and current_close > recent_low
        
        mci_score = 0.0
        mci_type = "NEUTRAL"
        displacement_vector = 0.0
        absorption_factor = 0.0
        
        if is_high_sweep:
            # Bearish Sweep Logic
            sweep_magnitude = current_high - recent_high
            displacement = recent_high - current_close # How far back inside it closed
            
            # Ratio: How much it swept vs how much it rejected
            if sweep_magnitude > 0:
                rejection_ratio = displacement / sweep_magnitude
                # High rejection ratio means strong reaction -> Real Reversal likely
                # Low rejection ratio (closed just below) -> Weak reaction -> Absorption?
                
                mci_score = rejection_ratio
                displacement_vector = -displacement
                mci_type = "BEARISH_SWEEP"
                
                # Volume check (if available)
                # High volume on sweep + high rejection = Strong Reversal
        
        elif is_low_sweep:
            # Bullish Sweep Logic
            sweep_magnitude = recent_low - current_low
            displacement = current_close - recent_low
            
            if sweep_magnitude > 0:
                rejection_ratio = displacement / sweep_magnitude
                mci_score = rejection_ratio
                displacement_vector = displacement
                mci_type = "BULLISH_SWEEP"

        else:
            # Breakout Scenario (Displacement Analysis)
            # High Breakout + High Volume + Close near extreme = Real
            mci_type = "CONTINUATION"  # Logic to be expanded
            
        return {
            "mci_score": float(mci_score),
            "mci_type": mci_type,
            "displacement_vector": float(displacement_vector),
            "is_sweep": is_high_sweep or is_low_sweep
        }


    # =========================================================================
    # 3. LIQUIDITY PRIMITIVES
    # =========================================================================

    def detect_liquidity_primitives(self, df: pd.DataFrame, 
                                  pool_tolerance: float = 0.0005, 
                                  lookback: int = 50) -> Dict[str, int]:
        """
        Detects Pools, Imbalances, and Sweeps in the given dataframe.
        """
        if df.empty:
            return {"n_pools": 0, "n_imbalances": 0, "n_sweeps": 0}
            
        closes = df['close'].values
        # 1. Detect Liquidity Pools (Clusters of touches)
        # Simplified vectorized approach
        # Bin prices and count frequency
        
        # Naive approach: count approximate equalities
        # For efficiency in python, we might skip full grid search in realtime
        # Using a simple heuristic for now: nearby swing points
        
        # 2. Detect Imbalances (FVGs)
        # Body > 2x average, wicks small relative to body
        bodies = np.abs(df['close'] - df['open'])
        avg_body = bodies.mean()
        is_large_body = bodies > (avg_body * 1.5)
        # Logic for strict FVG (gap between wicks) requires adjacent candles
        # Simplified count of "Impulse Candles"
        n_imbalances = np.sum(is_large_body)
        
        # 3. Detect Sweeps (Long Wicks)
        ranges = df['high'] - df['low']
        avg_range = ranges.mean()
        # Large range but small body = large wick (potential sweep)
        is_sweep_candle = (ranges > avg_range * 1.5) & (bodies < avg_range * 0.4)
        n_sweeps = np.sum(is_sweep_candle)
        
        return {
            "n_pools": 0, # Placeholder for more complex logic
            "n_imbalances": int(n_imbalances),
            "n_sweeps": int(n_sweeps)
        }

# Singleton
market_physics = MarketPhysicsEngine()
