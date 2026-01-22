import pandas as pd
import numpy as np
from app.core.config import settings

def detect_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    [v6.0 - DOCTORAL PERSISTENCE] 
    Institutional Order Block (OB) Detection with Persistent Node Logic.
    State Machine:
    - 1: ACTIVE (Price has not returned)
    - 2: MITIGATED (Price touched the zone)
    - 3: VIOLATED (Price closed past the zone -> Potential Breaker)
    """
    if df.empty: return df
    df = df.copy()
    
    # 0. Initialization
    df["ob_bull"] = 0.0 # 0: None, 1: Active, 2: Mitigated, 3: Breaker
    df["ob_bear"] = 0.0
    df["ob_top"] = np.nan
    df["ob_bottom"] = np.nan
    
    if "bos_bull" not in df.columns: return df

    # Prepare logic helpers
    is_bearish = df["close"] < df["open"]
    is_bullish = df["close"] > df["open"]
    atr = (df["high"] - df["low"]).rolling(14).mean().ffill().fillna(df["close"] * 0.001)
    
    # Track indices for last opposite candles
    last_bear_idx_series = pd.Series(np.where(is_bearish, np.arange(len(df)), np.nan)).ffill()
    last_bull_idx_series = pd.Series(np.where(is_bullish, np.arange(len(df)), np.nan)).ffill()
    
    last_bear_idx = last_bear_idx_series.values
    last_bull_idx = last_bull_idx_series.values

    # Displacement validation (Must have expansion > ATR scalar)
    expansion_bull = (df["close"] - df["open"]) > (atr * settings.SMC_BOS_THRESHOLD)
    expansion_bear = (df["open"] - df["close"]) > (atr * settings.SMC_BOS_THRESHOLD)

    # Detect New OBs at the point of BOS
    bull_mask = df["bos_bull"].fillna(False) & expansion_bull.fillna(False)
    bear_mask = df["bos_bear"].fillna(False) & expansion_bear.fillna(False)
    
    ob_bull_col = df.columns.get_loc("ob_bull")
    ob_bear_col = df.columns.get_loc("ob_bear")
    ob_top_col = df.columns.get_loc("ob_top")
    ob_bot_col = df.columns.get_loc("ob_bottom")
    
    for i in np.where(bull_mask)[0]:
        val = last_bear_idx[i]
        if np.isnan(val): continue
        ob_idx = int(val)
        df.iloc[ob_idx, ob_bull_col] = 1.0 # Mark as ACTIVE
        df.iloc[ob_idx, ob_top_col] = df.iloc[ob_idx]["high"]
        df.iloc[ob_idx, ob_bot_col] = df.iloc[ob_idx]["low"]

    for i in np.where(bear_mask)[0]:
        val = last_bull_idx[i]
        if np.isnan(val): continue
        ob_idx = int(val)
        df.iloc[ob_idx, ob_bear_col] = 1.0 # Mark as ACTIVE
        df.iloc[ob_idx, ob_top_col] = df.iloc[ob_idx]["high"]
        df.iloc[ob_idx, ob_bot_col] = df.iloc[ob_idx]["low"]

    # [PERSISTENCE ENGINE] 
    # Propagate active zones and check for mitigation/violation
    active_bull_zones = []
    active_bear_zones = []
    
    for i in range(len(df)):
        low, high, close = df.iloc[i]["low"], df.iloc[i]["high"], df.iloc[i]["close"]
        
        # Add new zones to tracker
        if df.iloc[i, ob_bull_col] == 1.0:
            active_bull_zones.append({"top": df.iloc[i]["ob_top"], "bot": df.iloc[i]["ob_bottom"], "mitigated": False})
        if df.iloc[i, ob_bear_col] == 1.0:
            active_bear_zones.append({"top": df.iloc[i]["ob_top"], "bot": df.iloc[i]["ob_bottom"], "mitigated": False})

        # Check existing bull zones
        for zone in active_bull_zones:
            if not zone["mitigated"]:
                 if low <= zone["top"]:
                     zone["mitigated"] = True
                     # If closed below bottom -> Violated (Breaker)
                     if close < zone["bot"]:
                         df.iloc[i, ob_bull_col] = 3.0
                     else:
                         df.iloc[i, ob_bull_col] = 2.0

        # Check existing bear zones
        for zone in active_bear_zones:
            if not zone["mitigated"]:
                if high >= zone["bot"]:
                    zone["mitigated"] = True
                    if close > zone["top"]:
                        df.iloc[i, ob_bear_col] = 3.0
                    else:
                        df.iloc[i, ob_bear_col] = 2.0
                        
    return df
