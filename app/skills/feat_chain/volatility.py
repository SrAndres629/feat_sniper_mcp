import pandas as pd
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger("feat.neuro.volatility")

class VolatilityChannel:
    """
    🧬 Neuro-Channel: Volatility & Regime Detection
    Defines the 'Atmosphere' of the market to scale risk and exits.
    """
    def compute_regime_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        if len(df) < 20:
            return {"atr": 0.0, "z_score": 0.0, "volatility_bias": 0.0}
            
        # ATR Calculation
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        # Z-Score of Close Price relative to 20 SMA
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        z_score = (df['close'].iloc[-1] - sma20.iloc[-1]) / (std20.iloc[-1] + 1e-9)
        
        # Volatility Bias (Normalized ATR relative to its history)
        atr_history = tr.rolling(14).mean()
        vol_bias = (atr - atr_history.mean()) / (atr_history.std() + 1e-9)
        
        return {
            "atr": float(atr),
            "z_score": float(z_score),
            "volatility_bias": float(vol_bias)
        }

    def get_dynamic_boundaries(self, df: pd.DataFrame) -> Dict[str, float]:
        """Keltner Channel inspired dynamic boundaries."""
        sma20 = df['close'].rolling(20).mean().iloc[-1]
        
        # Approximate ATR
        tr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
        
        return {
            "upper": float(sma20 + (tr * 2.0)),
            "lower": float(sma20 - (tr * 2.0)),
            "mid": float(sma20)
        }
