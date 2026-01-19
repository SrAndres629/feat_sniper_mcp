import pandas as pd
import numpy as np

def detect_shadow_zones(df: pd.DataFrame) -> pd.DataFrame:
    """
    [v5.0 - DOCTORAL SPACE] Shadow Zones (ZS) Detection.
    Definition: Long rejection wicks representing rapid capital injection.
    Logic:
    - Wick > 60% of Total Range.
    - Zone: Starts at Wick Open (Candle Close/Open) and ends at Wick High/Low.
    - Use: Wick Fill targets (Magnet).
    """
    if df.empty: return df
    df = df.copy()
    
    # Initialize
    df["shadow_bull"] = False
    df["shadow_bull_top"] = 0.0
    df["shadow_bull_bottom"] = 0.0
    df["shadow_bear"] = False
    df["shadow_bear_top"] = 0.0
    df["shadow_bear_bottom"] = 0.0
    
    # Metrics
    body_size = abs(df["close"] - df["open"])
    total_range = df["high"] - df["low"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    
    # 1. Bullish Shadow (Long Lower Wick - Rejection from Lows)
    # Wick > 60% of Total Range
    is_long_lower = (lower_wick > (total_range * 0.6))
    
    # Mark the Shadow Zone (Mean Threshold - 50% of Wick)
    # Zone: From Wick Low to 50% of Wick
    df.loc[is_long_lower, "shadow_bull"] = True
    df.loc[is_long_lower, "shadow_bull_top"] = df.loc[is_long_lower, "low"] + (lower_wick[is_long_lower] * 0.5)
    df.loc[is_long_lower, "shadow_bull_bottom"] = df.loc[is_long_lower, "low"]
    
    # 2. Bearish Shadow (Long Upper Wick - Rejection from Highs)
    is_long_upper = (upper_wick > (total_range * 0.6))
    
    # Zone: From 50% of Wick to Wick High
    df.loc[is_long_upper, "shadow_bear"] = True
    df.loc[is_long_upper, "shadow_bear_top"] = df.loc[is_long_upper, "high"]
    df.loc[is_long_upper, "shadow_bear_bottom"] = df.loc[is_long_upper, "high"] - (upper_wick[is_long_upper] * 0.5)
    
    return df
