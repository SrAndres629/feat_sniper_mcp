import numpy as np
import pandas as pd
from .fractals import identify_fractals

def detect_zones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Supply/Demand zones based on fractal clustering with Time-Based Decay.
    """
    if "close" not in df.columns or len(df) < 20:
        df["zone_type"] = "NONE"
        df["zone_high"] = 0.0
        df["zone_low"] = 0.0
        df["zone_strength"] = 0.0
        return df
    
    # [PVP INTEGRATION]
    from app.skills.volume_profile import volume_profile
    profile = volume_profile.get_profile(df, bins=50)

    # Ensure fractals are identified
    df = identify_fractals(df)
    
    atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
    zone_tolerance = atr * 0.5 if atr > 0 else 0.0001
    
    df["zone_type"] = "NONE"
    df["zone_high"] = 0.0
    df["zone_low"] = 0.0
    df["zone_strength"] = 0.0
    
    current_idx = len(df) - 1
    lookback = min(100, len(df))
    history = df.tail(lookback)
    
    # 1. Supply Zones
    high_positions = np.where(history["fractal_high"])[0]
    if len(high_positions) >= 2:
        for pos in reversed(high_positions):
            level = history.iloc[pos]["high"]
            dist = np.abs(history["high"].values - level)
            matches = np.where(dist < zone_tolerance)[0]
            touches = len(matches)
            if touches >= 2:
                actual_idx = history.index[pos]
                idx_pos = df.index.get_loc(actual_idx)
                if not isinstance(idx_pos, int):
                    idx_pos = idx_pos.start if isinstance(idx_pos, slice) else idx_pos[0]
                
                age = current_idx - idx_pos
                decay = 1.0 / (1.0 + 0.01 * age)
                vol_boost = volume_profile.get_zone_quality(level+zone_tolerance, level-zone_tolerance, profile)
                
                df.loc[df.index[-1], "zone_type"] = "SUPPLY"
                df.loc[df.index[-1], "zone_high"] = level + zone_tolerance
                df.loc[df.index[-1], "zone_low"] = level - zone_tolerance
                df.loc[df.index[-1], "zone_strength"] = min(1.0, (touches / 5.0) * decay * vol_boost)
                break
                
    # 2. Demand Zones
    if df.iloc[-1]["zone_type"] == "NONE":
        low_positions = np.where(history["fractal_low"])[0]
        if len(low_positions) >= 2:
            for pos in reversed(low_positions):
                level = history.iloc[pos]["low"]
                dist = np.abs(history["low"].values - level)
                matches = np.where(dist < zone_tolerance)[0]
                touches = len(matches)
                if touches >= 2:
                    actual_idx = history.index[pos]
                    idx_pos = df.index.get_loc(actual_idx)
                    if not isinstance(idx_pos, int):
                        idx_pos = idx_pos.start if isinstance(idx_pos, slice) else idx_pos[0]
                        
                    age = current_idx - idx_pos
                    decay = 1.0 / (1.0 + 0.01 * age)
                    vol_boost = volume_profile.get_zone_quality(level+zone_tolerance, level-zone_tolerance, profile)
                    
                    df.loc[df.index[-1], "zone_type"] = "DEMAND"
                    df.loc[df.index[-1], "zone_high"] = level + zone_tolerance
                    df.loc[df.index[-1], "zone_low"] = level - zone_tolerance
                    df.loc[df.index[-1], "zone_strength"] = min(1.0, (touches / 5.0) * decay * vol_boost)
                    break
    return df
