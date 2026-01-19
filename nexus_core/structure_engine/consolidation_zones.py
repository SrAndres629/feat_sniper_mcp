import pandas as pd
import numpy as np

def detect_consolidation_zones(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    [v5.0 - DOCTORAL SPACE] Accumulation Zones (ZA) Detection.
    Definition: Tight range clusters representing contract buildup.
    Logic: Rango (H-L) < ATR * 0.5 continuously.
    """
    if df.empty: return df
    df = df.copy()
    
    # Initialize
    df["is_consolidation"] = False
    
    # [DOCTORAL HARDENING] POC (Point of Control)
    from app.skills.volume_profile import volume_profile
    
    # Calculate POC for the rolling window (e.g., last 50 candles)
    # This is expensive, so we do it only for the last bar or specific checks.
    # For DataFrame vectorization, we'll use a simplified approximation:
    # Weighted Average Price of the last N bars where volume is high.
    
    window = 50
    df["typ_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["vol_price"] = df["typ_price"] * df["tick_volume"]
    
    # Rolling VWAP as a proxy for POC in M5/M1 context
    df["rolling_vwap"] = df["vol_price"].rolling(window).sum() / df["tick_volume"].rolling(window).sum()
    
    # Logic: Consolidation is when price oscillates AROUND the POC/VWAP
    # Distortion: Distance from VWAP is small
    dist_to_vwap = abs(df["close"] - df["rolling_vwap"])
    atr = (df["high"] - df["low"]).rolling(14).mean()
    
    is_consolidation = dist_to_vwap < (atr * 0.5)
    
    # Count consecutive candles near POC
    df["tight_count"] = is_consolidation.astype(int).groupby((is_consolidation != is_consolidation.shift()).cumsum()).cumsum()
    
    df["is_consolidation"] = (df["tight_count"] >= 10) & is_consolidation # Needs longer time to be valid accumulation
    
    # Optional: Mark the boundaries of the box
    # For now, simplistic boolean marking
    
    return df
