import pandas as pd
import numpy as np

def identify_fractals(df: pd.DataFrame, sensitivity: float = 1.2) -> pd.DataFrame:
    """
    [v6.0 - TOPOLOGICAL RIGOR] 
    Dynamic Structural Mapping.
    A MajorSwing is verified using ATR-normalized displacement.
    """
    if df.empty: return df
    df = df.copy()

    # 1. DYNAMIC DEPTH (ATR-Based)
    atr = (df["high"] - df["low"]).rolling(14, min_periods=1).mean().ffill()
    atr = atr.replace(0, 1e-9).fillna(1e-9)
    
    # --- 2. MINOR FLOW ---
    h, l = df["high"], df["low"]
    df["minor_h"] = (h > h.shift(1)) & (h > h.shift(2)) & (h > h.shift(-1)) & (h > h.shift(-2))
    df["minor_l"] = (l < l.shift(1)) & (l < l.shift(2)) & (l < l.shift(-1)) & (l < l.shift(-2))

    df["internal_h"] = df["high"].where(df["minor_h"]).ffill()
    df["internal_l"] = df["low"].where(df["minor_l"]).ffill()

    # --- 3. MAJOR SWING ---
    highs, lows = df["high"].values, df["low"].values
    atrs = atr.values
    major_h, major_l = np.zeros(len(df), dtype=bool), np.zeros(len(df), dtype=bool)
    
    last_h_val, last_l_val = -np.inf, np.inf
    last_h_idx, last_l_idx = 0, 0
    is_searching_h = True 
    has_swept_idm = False 
    
    # Internal H/L for IDM check
    ih = df["internal_h"].values
    il = df["internal_l"].values

    for i in range(len(df)):
        if i > 0:
            if not np.isnan(ih[i-1]) and highs[i] > ih[i-1]: has_swept_idm = True
            if not np.isnan(il[i-1]) and lows[i] < il[i-1]: has_swept_idm = True

        if is_searching_h:
            if highs[i] > last_h_val:
                last_h_val, last_h_idx = highs[i], i
            
            displacement = last_h_val - lows[i]
            if displacement > (atrs[i] * sensitivity) and has_swept_idm:
                major_h[last_h_idx] = True
                is_searching_h, has_swept_idm = False, False
                last_l_val, last_l_idx = lows[i], i
        else:
            if lows[i] < last_l_val:
                last_l_val, last_l_idx = lows[i], i
            
            displacement = highs[i] - last_l_val
            if displacement > (atrs[i] * sensitivity) and has_swept_idm:
                major_l[last_l_idx] = True
                is_searching_h, has_swept_idm = True, False
                last_h_val, last_h_idx = highs[i], i
                
    df["major_h"], df["major_l"] = major_h, major_l
    df["anchor_h"] = df["high"].where(df["major_h"]).ffill()
    df["anchor_l"] = df["low"].where(df["major_l"]).ffill()
    
    return df
