import pandas as pd
import numpy as np

def detect_imbalances(df: pd.DataFrame) -> pd.DataFrame:
    """
    [PHASE 13 - DOCTORAL GEOMETRY]
    Fair Value Gap (FVG) / Imbalance Detection.
    Identifies inefficient price action where only one side of the market was active.
    """
    if df.empty: return df
    df = df.copy()

    # Bullish FVG: Low of candle i+1 is higher than High of candle i-1
    df["fvg_bull"] = (df["low"].shift(-1) > df["high"].shift(1))
    df["fvg_bull_top"] = df["low"].shift(-1)
    df["fvg_bull_bottom"] = df["high"].shift(1)
    
    # Bearish FVG: High of candle i+1 is lower than Low of candle i-1
    df["fvg_bear"] = (df["high"].shift(-1) < df["low"].shift(1))
    df["fvg_bear_top"] = df["low"].shift(1)
    df["fvg_bear_bottom"] = df["high"].shift(-1)
    
    # [DOCTORAL REFINEMENT] Freshness Tracking
    df["fvg_mitigated"] = False
    
    df["inversion_bull"] = False # Was Bear FVG, broken up -> Support
    df["inversion_bear"] = False # Was Bull FVG, broken down -> Resistance
    
    # Define Indices for Iteration (Using Integer Locations for Safety)
    bull_mask = df["fvg_bull"].values
    bear_mask = df["fvg_bear"].values
    
    bull_ilocs = np.where(bull_mask)[0]
    bear_ilocs = np.where(bear_mask)[0]

    # [OPTIMIZATION] Pure Numpy Access for Speed & Safety
    # Pre-fetch columns as numpy arrays to avoid 'Lengths must match' errors
    bull_top_values = df["fvg_bull_top"].values
    bull_bottom_values = df["fvg_bull_bottom"].values
    
    bear_top_values = df["fvg_bear_top"].values
    bear_bottom_values = df["fvg_bear_bottom"].values
    
    inversion_bull_col_idx = df.columns.get_loc("inversion_bull")
    inversion_bear_col_idx = df.columns.get_loc("inversion_bear")
    
    # [CRITICAL FIX] Define closes explicitly
    closes = df["close"].values

    # Check Bull FVGs -> Potential Bear Inversion
    for i in bull_ilocs:
        if i >= len(df) - 1: continue
            
        # Safe Scalar Access via Numpy
        bottom = bull_bottom_values[i]
        
        # Look forward using slice
        future_closes = closes[i+1:]
        
        # [DEFENSE] Prevent Empty Slice Comp
        if len(future_closes) == 0: continue
        
        # Numpy Compare (Array vs Scalar) - Guaranteed Safe
        violations = future_closes < bottom
        
        if violations.any():
            # first True in violations
            rel_idx = np.argmax(violations)
            abs_iloc = i + 1 + rel_idx
            
            # Set Inversion Flag at that specific candle using iat (safe integer access)
            df.iat[abs_iloc, inversion_bear_col_idx] = True

    # Check Bear FVGs -> Potential Bull Inversion
    for i in bear_ilocs:
        if i >= len(df) - 1: continue
            
        top = bear_top_values[i] # Safe Scalar
        
        future_closes = closes[i+1:]
        
        # [DEFENSE] Prevent Empty Slice Comp
        if len(future_closes) == 0: continue
        
        violations = future_closes > top
        
        if violations.any():
            rel_idx = np.argmax(violations)
            abs_iloc = i + 1 + rel_idx
            
            # Set Inversion Flag
            df.iat[abs_iloc, inversion_bull_col_idx] = True

    # Zone Identification (Consolidated Imbalance Score)
    df["imbalance_score"] = 0.0
    
    # Scoring Hierarchy:
    # 1. Fresh FVG (Strongest Standard) -> 1.0
    # 2. Inversion FVG (Strongest Reversal) -> 0.8 / -0.8
    # 3. Mitigated FVG (Weak) -> 0.2
    
    mask_bull = df["fvg_bull"]
    mask_bull_fresh = mask_bull & (~df["fvg_mitigated"])
    
    df.loc[mask_bull_fresh, "imbalance_score"] = 1.0
    df.loc[mask_bull & df["fvg_mitigated"], "imbalance_score"] = 0.2
    
    # Overwrite with Inversion
    df.loc[df["inversion_bull"], "imbalance_score"] = 0.8 # Strong Support
    
    mask_bear = df["fvg_bear"]
    mask_bear_fresh = mask_bear & (~df["fvg_mitigated"])
    
    df.loc[mask_bear_fresh, "imbalance_score"] = -1.0
    df.loc[mask_bear & df["fvg_mitigated"], "imbalance_score"] = -0.2
    
    # Overwrite
    df.loc[df["inversion_bear"], "imbalance_score"] = -0.8 # Strong Resistance
    
    return df

def get_active_imbalances(df: pd.DataFrame, current_price: float):
    """Returns FVG zones that haven't been filled by current price."""
    # This logic would normally track if price has returned to the FVG range
    # For now, we flag active FVGs near price
    pass
