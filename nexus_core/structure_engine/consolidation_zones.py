import pandas as pd
import numpy as np

def detect_consolidation_zones(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """
    [v6.0 - DOCTORAL LIQUID DENSITY] Accumulation Zones Detection.
    Logic: High Kurtosis (concentration) in the KDE Volume Profile + Price near POC.
    """
    if df.empty or len(df) < window: return df
    df = df.copy()
    
    from app.skills.volume_profile import volume_profile
    
    df["is_consolidation"] = False
    df["poc_dist"] = 0.0
    df["profile_kurtosis"] = 0.0
    
    # We apply this over a rolling window. For performance, we can skip bars.
    # Doctoral Optimization: Only calculate every 5 bars to reduce load, or on-demand.
    for i in range(window, len(df)):
        sub_df = df.iloc[i-window:i]
        profile = volume_profile.get_profile(sub_df)
        
        if not profile: continue
        
        poc = profile["poc"]
        kurt = profile["kurtosis"]
        close = df.iloc[i]["close"]
        
        # Consolidation Logic:
        # 1. High Kurtosis (> 3.0) means volume is tightly packed (Aceptación).
        # 2. Price is within 0.2 ATR of the POC.
        atr = (df.iloc[i]["high"] - df.iloc[i]["low"]) # local bar range as proxy
        
        is_near_poc = abs(close - poc) < (atr * 0.5)
        is_dense = kurt > 3.0
        
        df.iloc[i, df.columns.get_loc("is_consolidation")] = is_near_poc and is_dense
        df.iloc[i, df.columns.get_loc("poc_dist")] = abs(close - poc)
        df.iloc[i, df.columns.get_loc("profile_kurtosis")] = kurt

    # Count consecutive consolidation candles
    df["tight_count"] = df["is_consolidation"].astype(int).groupby((df["is_consolidation"] != df["is_consolidation"].shift()).cumsum()).cumsum()
    df["is_consolidation"] = (df["tight_count"] >= 5) & df["is_consolidation"]
    
    return df
