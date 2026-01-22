"""
[MODULE 04 - DOCTORAL RESONANCE ENGINE]
Spectral Frequency Analysis with Zero-Lag Filters.

Philosophy:
- Moving averages are not indicators, they are Impulse Response Filters.
- The Sniper must see frequencies of power, not lagging lines.
- Elasticity measures the tension between price and institutional anchors.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from app.core.config import settings
from .filters import hull_ma, alma, weighted_ma, tema

logger = logging.getLogger("Nexus.ResonanceEngine")


class ResonanceEngine:
    """
    [v6.0 - DOCTORAL DSP]
    Spectral Resonance Engine with Zero-Lag HMA and Elasticity Channel.
    
    Neural Channels:
    - 15: resonance_alignment (HMA order score)
    - 16: spectral_dispersion (std dev between frequency layers)
    - 17: elasticity_z (distance to ALMA-200, Z-scored)
    - 18: resonance_slope (weighted slope of all layers)
    """
    
    def __init__(self):
        self.eps = 1e-9
        
        # Load periods from config (NO MAGIC NUMBERS)
        self.hma_fast = settings.RESONANCE_HMA_FAST
        self.hma_medium = settings.RESONANCE_HMA_MEDIUM
        self.hma_slow = settings.RESONANCE_HMA_SLOW
        self.alma_period = settings.RESONANCE_ALMA_PERIOD
        self.alma_offset = settings.RESONANCE_ALMA_OFFSET
        self.alma_sigma = settings.RESONANCE_ALMA_SIGMA
        self.elasticity_lookback = settings.RESONANCE_ELASTICITY_LOOKBACK
        self.dispersion_threshold = settings.RESONANCE_DISPERSION_THRESHOLD
        
        logger.info(f"[ResonanceEngine] Initialized with HMA({self.hma_fast}/{self.hma_medium}/{self.hma_slow}), ALMA({self.alma_period})")

    def compute_spectral_tensor(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [CORE] Computes full spectral tensor for neural consumption.
        Returns DataFrame with resonance columns for channels 15-18.
        """
        if df.empty or "close" not in df.columns:
            return df
        
        df = df.copy()
        close = df["close"]
        
        # 1. ZERO-LAG HMA Layers
        df["hma_fast"] = hull_ma(close, self.hma_fast)
        df["hma_medium"] = hull_ma(close, self.hma_medium)
        df["hma_slow"] = hull_ma(close, self.hma_slow)
        
        # 2. INSTITUTIONAL ANCHOR (ALMA-200)
        df["alma_anchor"] = alma(close, self.alma_period, self.alma_offset, self.alma_sigma)
        
        # 3. ATR for Scale Invariance
        atr = (df["high"] - df["low"]).rolling(14, min_periods=1).mean().ffill().replace(0, self.eps)
        df["atr"] = atr
        
        # =============================================
        # CHANNEL 15: RESONANCE ALIGNMENT (HMA Order)
        # =============================================
        # Bullish: Fast > Medium > Slow
        # Bearish: Fast < Medium < Slow
        bull_order = (df["hma_fast"] > df["hma_medium"]) & (df["hma_medium"] > df["hma_slow"])
        bear_order = (df["hma_fast"] < df["hma_medium"]) & (df["hma_medium"] < df["hma_slow"])
        
        df["resonance_alignment"] = 0.0
        df.loc[bull_order, "resonance_alignment"] = 1.0
        df.loc[bear_order, "resonance_alignment"] = -1.0
        
        # Partial alignment (2 of 3)
        partial_bull = (df["hma_fast"] > df["hma_medium"]) | (df["hma_medium"] > df["hma_slow"])
        partial_bear = (df["hma_fast"] < df["hma_medium"]) | (df["hma_medium"] < df["hma_slow"])
        df.loc[partial_bull & ~bull_order, "resonance_alignment"] = 0.5
        df.loc[partial_bear & ~bear_order, "resonance_alignment"] = -0.5
        
        # =============================================
        # CHANNEL 16: SPECTRAL DISPERSION
        # =============================================
        # Std dev between the 3 HMA layers normalized by ATR
        hma_stack = pd.concat([df["hma_fast"], df["hma_medium"], df["hma_slow"]], axis=1)
        dispersion_raw = hma_stack.std(axis=1)
        df["spectral_dispersion"] = (dispersion_raw / atr).clip(0, 5)
        
        # =============================================
        # CHANNEL 17: ELASTICITY Z-SCORE
        # =============================================
        # Distance from price to ALMA-200, normalized by ATR, then Z-scored
        distance_to_anchor = close - df["alma_anchor"]
        distance_normalized = distance_to_anchor / atr
        
        # Rolling Z-Score
        mean_dist = distance_normalized.rolling(self.elasticity_lookback, min_periods=1).mean()
        std_dist = distance_normalized.rolling(self.elasticity_lookback, min_periods=1).std().replace(0, self.eps)
        df["elasticity_z"] = ((distance_normalized - mean_dist) / std_dist).clip(-3, 3)
        
        # =============================================
        # CHANNEL 18: RESONANCE SLOPE
        # =============================================
        # Weighted average of HMA slopes
        def calc_slope(series, window=5):
            return series.diff(window) / window
        
        slope_fast = calc_slope(df["hma_fast"]) / atr
        slope_medium = calc_slope(df["hma_medium"]) / atr
        slope_slow = calc_slope(df["hma_slow"]) / atr
        
        # Weighted: Fast has highest weight
        df["resonance_slope"] = (slope_fast * 0.5 + slope_medium * 0.3 + slope_slow * 0.2).clip(-2, 2)
        
        # =============================================
        # MEAN REVERSION SIGNAL (Bonus)
        # =============================================
        # If elasticity is extreme, flag potential mean reversion
        df["mean_reversion_signal"] = (df["elasticity_z"].abs() > self.dispersion_threshold).astype(float)
        
        # Fill NaNs from warmup
        for col in ["hma_fast", "hma_medium", "hma_slow", "alma_anchor", 
                    "resonance_alignment", "spectral_dispersion", "elasticity_z", "resonance_slope"]:
            df[col] = df[col].fillna(0.0)
        
        return df

    def get_live_resonance(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Returns the latest resonance metrics for live trading.
        """
        df = self.compute_spectral_tensor(df)
        if df.empty:
            return {}
        
        last = df.iloc[-1]
        return {
            "resonance_alignment": float(last.get("resonance_alignment", 0.0)),
            "spectral_dispersion": float(last.get("spectral_dispersion", 0.0)),
            "elasticity_z": float(last.get("elasticity_z", 0.0)),
            "resonance_slope": float(last.get("resonance_slope", 0.0)),
            "mean_reversion_signal": float(last.get("mean_reversion_signal", 0.0)),
            "hma_fast": float(last.get("hma_fast", 0.0)),
            "hma_medium": float(last.get("hma_medium", 0.0)),
            "hma_slow": float(last.get("hma_slow", 0.0)),
            "alma_anchor": float(last.get("alma_anchor", 0.0)),
        }


# Singleton
resonance_engine = ResonanceEngine()
