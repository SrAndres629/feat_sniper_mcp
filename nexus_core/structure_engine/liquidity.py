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
        
    # Check Highs (EQH)
    for i in range(len(major_h_indices)):
        for j in range(i + 1, len(major_h_indices)):
            idx_i = major_h_indices[i]
            idx_j = major_h_indices[j]
            
            # Distance logic
            dist = abs(highs[idx_i] - highs[idx_j])
            current_tol = tol_values[idx_j]
            
            # Simple Scalar Comparison
            if dist <= current_tol:
                # Set flag at J (the second high)
                # Use iat for integer index
                df.iat[idx_j, df.columns.get_loc("is_eqh")] = True
                
    # Check Lows (EQL)
    for i in range(len(major_l_indices)):
        for j in range(i + 1, len(major_l_indices)):
            idx_i = major_l_indices[i]
            idx_j = major_l_indices[j]
            
            dist = abs(lows[idx_i] - lows[idx_j])
            current_tol = tol_values[idx_j]
            
            if dist <= current_tol:
                df.iat[idx_j, df.columns.get_loc("is_eql")] = True
                
    # Liquidity Magnet Force (Calculated distance to nearest pool)
    df["liq_target_h"] = df["high"].where(df["is_eqh"]).ffill()
    df["liq_target_l"] = df["low"].where(df["is_eql"]).ffill()
    
    return df
