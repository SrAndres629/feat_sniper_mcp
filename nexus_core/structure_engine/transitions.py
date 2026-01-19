import pandas as pd
import numpy as np
from .fractals import identify_fractals
from .sessions import identify_trading_session, get_session_weight
from app.skills.volume_profile import volume_profile

def detect_structural_shifts(df: pd.DataFrame) -> pd.DataFrame:
    """
    [v5.1 - VECTORIZED TOPOLOGY] Precise BOS/CHOCH Detection.
    Factors: Vectorized Session Mapping, Rolling POC, and Hierarchical Fractals.
    """
    if df.empty: return df
    df = identify_fractals(df)

    # 1. Track Hierarchical Levels
    df["swing_h"] = df["high"].where(df["major_h"]).ffill()
    df["swing_l"] = df["low"].where(df["major_l"]).ffill()
    df["internal_h"] = df["high"].where(df["minor_h"]).ffill()
    df["internal_l"] = df["low"].where(df["minor_l"]).ffill()

    # 2. VECTORIZED SESSION MAPPING (No .map bottleneck)
    # We use index attributes for speed
    hours = df.index.hour
    # Simplified mapping for vectorization:
    # 3-11: LONDON, 13-20: NY, Else: ASIA
    df["session_type"] = "ASIA"
    df.loc[(hours >= 3) & (hours <= 11), "session_type"] = "LONDON"
    df.loc[(hours >= 13) & (hours <= 20), "session_type"] = "NY"
    
    # Session Weights (Doctoral Institutional Importance)
    weights = {"LONDON": 1.5, "NY": 1.2, "ASIA": 0.5}
    df["session_weight"] = df["session_type"].map(weights).fillna(1.0)

    # 3. ROLLING POC (Fixes Lookahead Bias)
    # Instead of one profile for all time, we use a rolling Window (24h)
    # A simplified proxy for POC is a high-volume price area.
    # We'll use a 20-period rolling high-volume price
    df["rolling_poc"] = df["close"].rolling(24).mean() # Simplified but causal

    # 4. SWING BOS (Structural Confirmation)
    # Rule: Candle Close > Previous Swing High
    raw_bos_bull = (df["close"] > df["swing_h"].shift(1)) & (df["close"].shift(1) <= df["swing_h"].shift(1))
    raw_bos_bear = (df["close"] < df["swing_l"].shift(1)) & (df["close"].shift(1) >= df["swing_l"].shift(1))
    
    # Validation: Institutional Force (Trending POC)
    df["bos_bull"] = raw_bos_bull & (df["close"] > df["rolling_poc"])
    df["bos_bear"] = raw_bos_bear & (df["close"] < df["rolling_poc"])

    # 5. INTERNAL CHOCH (Change of Character)
    # CHoCH is the first sign of reversal: price breaking the last internal structural point
    # of a preceding move.
    df["choch_bull"] = (df["close"] > df["internal_h"].shift(1)) & (df["swing_h"].shift(1) < df["swing_h"].shift(2))
    df["choch_bear"] = (df["close"] < df["internal_l"].shift(1)) & (df["swing_l"].shift(1) > df["swing_l"].shift(2))
    
    # 6. BOS STRENGTH (For Neural Ingest)
    is_bos = (df["bos_bull"] | df["bos_bear"])
    df["bos_strength"] = df["session_weight"] * np.where(is_bos, 1.2, 0.0)

    # 7. RANGE EQUILIBRIUM (Premium vs Discount)
    range_dist = (df["swing_h"] - df["swing_l"]) + 1e-9
    df["range_pos"] = (df["close"] - df["swing_l"]) / range_dist
    
    # Refined Strength: Buy in Discount, Sell in Premium
    df.loc[df["bos_bull"] & (df["range_pos"] < 0.5), "bos_strength"] *= 1.5
    df.loc[df["bos_bear"] & (df["range_pos"] > 0.5), "bos_strength"] *= 1.5

    return df
