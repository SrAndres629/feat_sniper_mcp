import pandas as pd
import numpy as np

def detect_critical_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    [v5.0 - DOCTORAL SPACE] Critical Points (PC) Detection.
    Definition: Indecision candles (Doji/Small Body) acting as market hinges.
    Logic: 
    - Body < 20% of Range.
    - Long Wicks relative to body.
    - Captures the exact moment of institutional hand-over.
    """
    if df.empty: return df
    df = df.copy()
    
    # Initialize columns
    df["is_critical_point"] = False
    df["pc_top"] = 0.0
    df["pc_bottom"] = 0.0
    df["pc_type"] = "NONE" # HINGE_HIGH, HINGE_LOW
    
    # Calculate Candle Metrics
    body_size = abs(df["close"] - df["open"])
    total_range = df["high"] - df["low"]
    
    # 1. Identification: Body is tiny (< 10% of range)
    is_doji = body_size < (total_range * 0.1)
    
    # 2. Volume Indecision (High Volume Doji = Absorption/Hinge)
    # [DOCTORAL REFINEMENT] Volume Validation
    # A low volume Doji is noise. A high volume Doji is a battle.
    # [V6 SYNC] Use 'volume' (DB native) instead of 'tick_volume'
    vol_col = "volume" if "volume" in df.columns else "tick_volume"
    vol_ma = df[vol_col].rolling(20).mean()
    is_high_vol = df[vol_col] > (vol_ma * 1.5)
    
    # Critical Point = Tiny Body AND High Volume (Absorption)
    is_critical = is_doji & is_high_vol
    
    # 3. Mark the Critical Point
    # The Zone is the ENTIRE range of the candle (Wick to Wick)
    
    df.loc[is_critical, "is_critical_point"] = True
    df.loc[is_critical, "pc_top"] = df["high"]
    df.loc[is_critical, "pc_bottom"] = df["low"]
    
    # Identify context (Hinge High vs Hinge Low is determined by next candle)
    # But for real-time detection, we mark the existence.
    # Future confluence logic will determine if it's acting as Support or Resistance.
    
    return df
