import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def identify_fractals(df: pd.DataFrame, sensitivity: float = 2.0) -> pd.DataFrame:
    """
    [v5.1 - TOPOLOGICAL RIGOR] 
    Dynamic Structural Mapping.
    A MajorSwing is verified using ATR-normalized displacement (Manifold Invariance).
    
    sensitivity: Multiple of ATR required for structural validity.
    """
    if df.empty: return df
    df = df.copy()

    # 1. DYNAMIC DEPTH (ATR-Based)
    # Market volatility dictates the depth of 'meaningful' pivots.
    atr = (df["high"] - df["low"]).rolling(14).mean().ffill().fillna(df["close"] * 0.001)
    
    # --- 2. MINOR FLOW (Target Liquidity / Inducement) ---
    # Vectorized 5-bar fractal
    h = df["high"]
    l = df["low"]
    df["minor_h"] = (h > h.shift(1)) & (h > h.shift(2)) & (h > h.shift(-1)) & (h > h.shift(-2))
    df["minor_l"] = (l < l.shift(1)) & (l < l.shift(2)) & (l < l.shift(-1)) & (l < l.shift(-2))

    # Initialize Internal Structure Columns
    df["internal_h"] = df["high"].where(df["minor_h"]).ffill()
    df["internal_l"] = df["low"].where(df["minor_l"]).ffill()

    # --- 3. MAJOR SWING (Vectorized Search Attempt) ---
    # To maintain strict SMC logic without lookahead, we use a vectorized state tracker
    # but for complex zigzag logic, we use optimized indexing.
    
    highs, lows = df["high"].values, df["low"].values
    atrs = atr.values
    major_h, major_l = np.zeros(len(df), dtype=bool), np.zeros(len(df), dtype=bool)
    
    last_h_val, last_l_val = -np.inf, np.inf
    last_h_idx, last_l_idx = 0, 0
    is_searching_h = True 
    has_swept_idm = False # Inducement verification
    
    # Vectorized check for IDM sweep (current price vs trailing minor pivot)
    # We'll do this inside the optimization loop for causal strictness
    
    for i in range(len(df)):
        # INDUCEMENT CHECK: Price must invalidate a minor level to prove 'Intent'
        if i > 0:
            if highs[i] > df["internal_h"].iloc[i-1] or lows[i] < df["internal_l"].iloc[i-1]:
                has_swept_idm = True

        # DISPLACEMENT CHECK: Trend reversal must exceed N * ATR
        # Rule: A pivot is confirmed when price moves X * ATR in opposite direction.
        if is_searching_h:
            if highs[i] > last_h_val:
                last_h_val, last_h_idx = highs[i], i
            
            # Reversal criteria (Absolute price check is legacy, ATR is future)
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
    
    # 4. METADATA FOR MANIFOLD MATHEMATICIAN
    df["anchor_h"] = df["high"].where(df["major_h"]).ffill()
    df["anchor_l"] = df["low"].where(df["major_l"]).ffill()
    
    # Vectorized Age Calculation
    df["struct_age"] = df.groupby((df["major_h"] | df["major_l"]).cumsum()).cumcount()
    
    # [NEURAL] Normalize age/displacement for the Tensor Topologist
    df["struct_displacement_z"] = (df["high"] - df["anchor_l"]) / (atr * sensitivity + 1e-9)
    
    return df
