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
    bull_mask = (df["bos_bull"].fillna(False) & expansion_bull.fillna(False)).values
    bear_mask = (df["bos_bear"].fillna(False) & expansion_bear.fillna(False)).values
    
    # Pre-fetch numpy arrays for performance
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    # 1. MARK INITIAL ACTIVE NODES
    bull_ob_indices = []
    bear_ob_indices = []

    for i in np.where(bull_mask)[0]:
        val = last_bear_idx[i]
        if not np.isnan(val):
            ob_idx = int(val)
            df.iat[ob_idx, df.columns.get_loc("ob_bull")] = 1.0 # Mark as ACTIVE
            df.iat[ob_idx, df.columns.get_loc("ob_top")] = highs[ob_idx]
            df.iat[ob_idx, df.columns.get_loc("ob_bottom")] = lows[ob_idx]
            bull_ob_indices.append(ob_idx)

    for i in np.where(bear_mask)[0]:
        val = last_bull_idx[i]
        if not np.isnan(val):
            ob_idx = int(val)
            df.iat[ob_idx, df.columns.get_loc("ob_bear")] = 1.0 # Mark as ACTIVE
            df.iat[ob_idx, df.columns.get_loc("ob_top")] = highs[ob_idx]
            df.iat[ob_idx, df.columns.get_loc("ob_bottom")] = lows[ob_idx]
            bear_ob_indices.append(ob_idx)

    # 2. [VECTORIZED PERSISTENCE ENGINE]
    # For each OB, find its first mitigation or violation point using Numpy
    
    # Process Bull OBs
    for ob_idx in bull_ob_indices:
        top = highs[ob_idx]
        bot = lows[ob_idx]
        
        # Look for first touch (Low <= Top)
        future_lows = lows[ob_idx+1:]
        touches = np.where(future_lows <= top)[0]
        
        if len(touches) > 0:
            hit_idx = ob_idx + 1 + touches[0]
            # Check for violation (Close < Bot) at mitigation point
            if closes[hit_idx] < bot:
                df.iat[hit_idx, df.columns.get_loc("ob_bull")] = 3.0 # Breaker
            else:
                df.iat[hit_idx, df.columns.get_loc("ob_bull")] = 2.0 # Mitigated

    # Process Bear OBs
    for ob_idx in bear_ob_indices:
        top = highs[ob_idx]
        bot = lows[ob_idx]
        
        # Look for first touch (High >= Bot)
        future_highs = highs[ob_idx+1:]
        touches = np.where(future_highs >= bot)[0]
        
        if len(touches) > 0:
            hit_idx = ob_idx + 1 + touches[0]
            # Check for violation (Close > Top) at mitigation point
            if closes[hit_idx] > top:
                df.iat[hit_idx, df.columns.get_loc("ob_bear")] = 3.0 # Breaker
            else:
                df.iat[hit_idx, df.columns.get_loc("ob_bear")] = 2.0 # Mitigated
                        
    return df
