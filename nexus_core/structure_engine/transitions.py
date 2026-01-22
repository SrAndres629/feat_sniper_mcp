import pandas as pd
import numpy as np
from .fractals import identify_fractals

def detect_structural_shifts(df: pd.DataFrame) -> pd.DataFrame:
    """
    [v6.0 - INSTITUTIONAL TOPOLOGY] Real-Close BOS/CHoCH Core.
    No wick signals. Structure is only broken when the auction CLOSES past the level.
    """
    if df.empty: return df
    df = identify_fractals(df)

    # 1. Track Hierarchical Levels
    df["swing_h"] = df["high"].where(df["major_h"]).ffill()
    df["swing_l"] = df["low"].where(df["major_l"]).ffill()
    df["internal_h"] = df["high"].where(df["minor_h"]).ffill()
    df["internal_l"] = df["low"].where(df["minor_l"]).ffill()

    # 2. Institutional Session Analysis
    # Fallback if settings not available or for tests
    try:
        from app.core.config import settings
        session_weights = settings.SMC_SESSION_WEIGHTS
    except:
        session_weights = {"LONDON": 1.5, "NY": 1.2, "ASIA": 0.5}

    if hasattr(df.index, 'hour'):
        hours = df.index.hour
    else:
        hours = pd.Series(0, index=df.index)

    df["session_type"] = "ASIA"
    df.loc[(hours >= 7) & (hours <= 11), "session_type"] = "LONDON"
    df.loc[(hours >= 12) & (hours <= 16), "session_type"] = "NY"
    
    df["session_weight"] = df["session_type"].map(session_weights).fillna(0.5)

    # 3. [DOCTORAL] Real-Close BOS (Structural Confirmation)
    # Rule: Price CLOSES past the previous swing high/low
    df["bos_bull"] = (df["close"] > df["swing_h"].shift(1)) & (df["close"].shift(1) <= df["swing_h"].shift(1))
    df["bos_bear"] = (df["close"] < df["swing_l"].shift(1)) & (df["close"].shift(1) >= df["swing_l"].shift(1))

    # 4. Change of Character (CHoCH)
    # The first break of internal structure signaling potential reversal
    df["choch_bull"] = (df["close"] > df["internal_h"].shift(1)) & (df["swing_h"].shift(1) < df["swing_h"].shift(2))
    df["choch_bear"] = (df["close"] < df["internal_l"].shift(1)) & (df["swing_l"].shift(1) > df["swing_l"].shift(2))
    
    # 5. BOS STRENGTH (Scale Invariant)
    is_bos = (df["bos_bull"].fillna(False) | df["bos_bear"].fillna(False)).astype(bool)
    df["bos_strength"] = df["session_weight"] * np.where(is_bos, 1.0, 0.0)

    # 6. Range Equilibrium (Premium vs Discount)
    range_dist = (df["swing_h"] - df["swing_l"]) + 1e-9
    df["range_pos"] = (df["close"] - df["swing_l"]) / range_dist
    
    # Optimization: Stronger BOS if occurring in appropriate range zone
    df.loc[df["bos_bull"].fillna(False) & (df["range_pos"] < 0.5), "bos_strength"] *= 1.2
    df.loc[df["bos_bear"].fillna(False) & (df["range_pos"] > 0.5), "bos_strength"] *= 1.2

    return df
