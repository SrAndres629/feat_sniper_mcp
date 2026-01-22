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
    
    # [OPTIMIZATION] Pure Numpy Iteration to avoid .at[] Series Ambiguity
    highs = df["high"].values
    lows = df["low"].values
    tol_values = tolerance.values
    
    # We need integer indices for fast array access
    # We assume 'major_h' is boolean column
    if "major_h" in df.columns:
        major_h_bools = df["major_h"].fillna(False).values
        major_h_indices = np.where(major_h_bools)[0]
    else:
        major_h_indices = []

    if "major_l" in df.columns:
        major_l_bools = df["major_l"].fillna(False).values
        major_l_indices = np.where(major_l_bools)[0]
    else:
        major_l_indices = []
        
    # Collect EQH/EQL candidates as lists to avoid df.iat overhead
    eqh_indices = []
    eql_indices = []
    
    # Check Highs (EQH)
    for i in range(len(major_h_indices)):
        idx_i = major_h_indices[i]
        val_i = highs[idx_i]
        for j in range(i + 1, len(major_h_indices)):
            idx_j = major_h_indices[j]
            if abs(val_i - highs[idx_j]) <= tol_values[idx_j]:
                eqh_indices.append(idx_j)
                
    # Check Lows (EQL)
    for i in range(len(major_l_indices)):
        idx_i = major_l_indices[i]
        val_i = lows[idx_i]
        for j in range(i + 1, len(major_l_indices)):
            idx_j = major_l_indices[j]
            if abs(val_i - lows[idx_j]) <= tol_values[idx_j]:
                eql_indices.append(idx_j)

    # Bulk update flags
    if eqh_indices:
        df.loc[df.index[eqh_indices], "is_eqh"] = True
    if eql_indices:
        df.loc[df.index[eql_indices], "is_eql"] = True
                
    # Liquidity Magnet Force (Calculated distance to nearest pool)
    df["liq_target_h"] = df["high"].where(df["is_eqh"]).ffill()
    df["liq_target_l"] = df["low"].where(df["is_eql"]).ffill()
    
    return df
