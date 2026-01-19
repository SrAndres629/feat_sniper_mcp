import numpy as np
import pandas as pd
from .fractals import identify_fractals

def detect_zones(df: pd.DataFrame) -> pd.DataFrame:
    """
    [v5.0 - SMC DOCTORAL] Supply/Demand Zones with 'Memory of Pain'.
    Rule: Zones weaken with each test. 3+ touches = imminent rupture.
    """
    if "close" not in df.columns or len(df) < 20:
        df["zone_type"], df["zone_strength"], df["test_count"] = "NONE", 0.0, 0
        return df
    
    # [PVP INTEGRATION]
    from app.skills.volume_profile import volume_profile
    profile = volume_profile.get_profile(df, bins=50)

    # Ensure fractals are identified
    df = identify_fractals(df)
    
    atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
    zone_tolerance = atr * 0.5 if atr > 0 else 0.0001
    
    df["zone_type"], df["zone_high"], df["zone_low"], df["zone_strength"], df["test_count"] = "NONE", 0.0, 0.0, 0.0, 0
    
    current_idx = len(df) - 1
    lookback = min(100, len(df))
    history = df.tail(lookback)
    
    # 1. Supply Zones
    high_positions = np.where(history["major_h"])[0] # Use Major Swing for strong zones
    if len(high_positions) >= 1:
        for pos in reversed(high_positions):
            level = history.iloc[pos]["high"]
            # Count touches in the lookback
            dist = np.abs(history["high"].values - level)
            matches = np.where(dist < zone_tolerance)[0]
            test_count = len(matches)
            
            if test_count >= 1:
                actual_idx = history.index[pos]
                # Calculate Strength with 'Memory of Pain' Decay
                # Touches: 1=Strongest, 2=Medium, 3=Fragile, 4+=Imminent Breakout
                fragility_multiplier = 1.0 if test_count <= 1 else 0.6 if test_count == 2 else 0.2 if test_count == 3 else 0.05
                
                age = current_idx - df.index.get_loc(actual_idx)
                if not isinstance(age, int): age = age.start if isinstance(age, slice) else age[0]
                time_decay = 1.0 / (1.0 + 0.005 * age) # Institutional levels last longer
                
                vol_boost = volume_profile.get_zone_quality(level+zone_tolerance, level-zone_tolerance, profile)
                
                df.loc[df.index[-1], "zone_type"] = "SUPPLY"
                df.loc[df.index[-1], "zone_high"] = level + zone_tolerance
                df.loc[df.index[-1], "zone_low"] = level - zone_tolerance
                df.loc[df.index[-1], "test_count"] = test_count
                df.loc[df.index[-1], "zone_strength"] = min(1.0, fragility_multiplier * time_decay * vol_boost)
                break
                
    # 2. Demand Zones (Same logic for Lows)
    if df.iloc[-1]["zone_type"] == "NONE":
        low_positions = np.where(history["major_l"])[0]
        if len(low_positions) >= 1:
            for pos in reversed(low_positions):
                level = history.iloc[pos]["low"]
                dist = np.abs(history["low"].values - level)
                test_count = len(np.where(dist < zone_tolerance)[0])
                
                if test_count >= 1:
                    fragility_multiplier = 1.0 if test_count <= 1 else 0.6 if test_count == 2 else 0.2 if test_count == 3 else 0.05
                    actual_idx = history.index[pos]
                    age = current_idx - df.index.get_loc(actual_idx)
                    if not isinstance(age, int): age = age.start if isinstance(age, slice) else age[0]
                    time_decay = 1.0 / (1.0 + 0.005 * age)
                    
                    vol_boost = volume_profile.get_zone_quality(level+zone_tolerance, level-zone_tolerance, profile)
                    
                    df.loc[df.index[-1], "zone_type"] = "DEMAND"
                    df.loc[df.index[-1], "zone_high"] = level + zone_tolerance
                    df.loc[df.index[-1], "zone_low"] = level - zone_tolerance
                    df.loc[df.index[-1], "test_count"] = test_count
                    df.loc[df.index[-1], "zone_strength"] = min(1.0, fragility_multiplier * time_decay * vol_boost)
                    break
    return df
