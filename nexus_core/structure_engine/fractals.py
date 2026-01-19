import pandas as pd
import numpy as np

def identify_fractals(df: pd.DataFrame, depth: int = 12) -> pd.DataFrame:
    """
    [v5.0 - SMC DOCTORAL] 
    Refined Structural Mapping with Inducement Context.
    A MajorSwing is only validated AFTER price sweeps minor liquidity (Inducement).
    """
    if df.empty: return df
    df = df.copy()

    # --- 1. MINOR FLOW (Target Liquidity) ---
    df["minor_h"] = (df["high"].shift(2) > df["high"].shift(4)) & \
                    (df["high"].shift(2) > df["high"].shift(3)) & \
                    (df["high"].shift(2) > df["high"].shift(1)) & \
                    (df["high"].shift(2) > df["high"])
    
    df["minor_l"] = (df["low"].shift(2) < df["low"].shift(4)) & \
                    (df["low"].shift(2) < df["low"].shift(3)) & \
                    (df["low"].shift(2) < df["low"].shift(1)) & \
                    (df["low"].shift(2) < df["low"])

    # --- 2. MAJOR SWING (Protected Structure) ---
    highs, lows = df["high"].values, df["low"].values
    major_h, major_l = np.zeros(len(df), dtype=bool), np.zeros(len(df), dtype=bool)
    
    last_h_val, last_l_val = -np.inf, np.inf
    last_h_idx, last_l_idx = 0, 0
    is_searching_h = True 
    
    # Track the 'Last Inducement' (price breaking a minor high/low)
    has_swept_liquidity = False
    
    for i in range(len(df)):
        # Check for Inducement (Minor Liquidity Sweep)
        if i > 0:
            if highs[i] > df["internal_h"].iloc[i-1] or lows[i] < df["internal_l"].iloc[i-1]:
                has_swept_liquidity = True

        if is_searching_h:
            if highs[i] > last_h_val:
                last_h_val, last_h_idx = highs[i], i
            # Confirm High only if price drops AND we had a liquidity sweep (Inducement)
            if i - last_h_idx >= depth and lows[i] < last_h_val - (last_h_val * 0.005):
                if has_swept_liquidity:
                    major_h[last_h_idx] = True
                    is_searching_h, has_swept_liquidity = False, False
                    last_l_val, last_l_idx = lows[i], i
        else:
            if lows[i] < last_l_val:
                last_l_val, last_l_idx = lows[i], i
            if i - last_l_idx >= depth and highs[i] > last_l_val + (last_l_val * 0.005):
                if has_swept_liquidity:
                    major_l[last_l_idx] = True
                    is_searching_h, has_swept_liquidity = False, False
                    last_h_val, last_h_idx = highs[i], i
                
    df["major_h"], df["major_l"] = major_h, major_l
    
    # 3. METADATA FOR MANIFOLD MATHEMATICIAN
    # Calculate age of current structural range
    df["anchor_h"] = df["high"].where(df["major_h"]).ffill()
    df["anchor_l"] = df["low"].where(df["major_l"]).ffill()
    
    # Fractal Stability (How long has this level held?)
    df["struct_age"] = df.groupby((df["major_h"] | df["major_l"]).cumsum()).cumcount()
    
    return df
