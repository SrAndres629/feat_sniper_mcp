import pandas as pd
import numpy as np

def detect_liquidity_pools(df: pd.DataFrame, tolerance_atr: float = 0.1) -> pd.DataFrame:
    """
    [v5.0 - SMC DOCTORAL] Liquidity Pool Detection (EQH / EQL).
    Identifies 'Retail Double Tops/Bottoms' which act as magnets for institutional sweeps.
    """
    if df.empty: return df
    df = df.copy()
    
    # Calculate ATR for adaptive tolerance
    atr = (df["high"] - df["low"]).rolling(14).mean().ffill()
    tolerance = atr * tolerance_atr
    
    df["is_eqh"] = False # Equal Highs
    df["is_eql"] = False # Equal Lows
    
    # We look for Major Highs/Lows that are close to each other
    major_h_indices = df.index[df["major_h"]].tolist()
    major_l_indices = df.index[df["major_l"]].tolist()
    
    # Check Highs (EQH)
    for i in range(len(major_h_indices)):
        for j in range(i + 1, len(major_h_indices)):
            idx_i, idx_j = major_h_indices[i], major_h_indices[j]
            dist = abs(df.at[idx_i, "high"] - df.at[idx_j, "high"])
            if dist <= tolerance.at[idx_j]:
                df.at[idx_j, "is_eqh"] = True
                
    # Check Lows (EQL)
    for i in range(len(major_l_indices)):
        for j in range(i + 1, len(major_l_indices)):
            idx_i, idx_j = major_l_indices[i], major_l_indices[j]
            dist = abs(df.at[idx_i, "low"] - df.at[idx_j, "low"])
            if dist <= tolerance.at[idx_j]:
                df.at[idx_j, "is_eql"] = True
                
    # Liquidity Magnet Force (Calculated distance to nearest pool)
    df["liq_target_h"] = df["high"].where(df["is_eqh"]).ffill()
    df["liq_target_l"] = df["low"].where(df["is_eql"]).ffill()
    
    return df
