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
    # GAP = Low[i+1] - High[i-1]
    df["fvg_bull"] = (df["low"].shift(-1) > df["high"].shift(1))
    df["fvg_bull_top"] = df["low"].shift(-1)
    df["fvg_bull_bottom"] = df["high"].shift(1)
    
    # Bearish FVG: High of candle i+1 is lower than Low of candle i-1
    # GAP = Low[i-1] - High[i+1]
    df["fvg_bear"] = (df["high"].shift(-1) < df["low"].shift(1))
    df["fvg_bear_top"] = df["low"].shift(1)
    df["fvg_bear_bottom"] = df["high"].shift(-1)
    
    # [DOCTORAL REFINEMENT] Freshness Tracking
    # An FVG is "Fresh" if price has not returned continuously to fill it.
    df["fvg_mitigated"] = False
    
    # 1. Track Bullish FVGs (Supports)
    # If price drops BELOW the FVG Top (gap fill), it's mitigated.
    # Note: Vectorizing this fully is complex, using an iterative approach for precision on recent bars.
    
    # 3. Inversion FVG Logic (The Flip)
    # If a Bullish FVG is closed below, it becomes Bearish Inversion (Resistance)
    # If a Bearish FVG is closed above, it becomes Bullish Inversion (Support)
    
    df["inversion_bull"] = False # Was Bear FVG, broken up -> Support
    df["inversion_bear"] = False # Was Bull FVG, broken down -> Resistance
    
    # Define Indices for Iteration (Using Integer Locations for Safety)
    # df.index can be Datetime, so we avoid idx+1 logic
    
    # Get boolean mask and find integer locations
    bull_mask = df["fvg_bull"].values
    bear_mask = df["fvg_bear"].values
    
    # Integer indices where FVG is True
    bull_ilocs = np.where(bull_mask)[0]
    bear_ilocs = np.where(bear_mask)[0]

    close_values = df["close"].values

    # Check Bull FVGs -> Potential Bear Inversion
    # We iterate, but we need to reference the original DataFrame for setting values or use iat
    # To enable safe slicing, we use iloc logic
    
    for i in bull_ilocs:
        # Avoid out of bounds
        if i >= len(df) - 1: continue
            
        top = df.at[df.index[i], "fvg_bull_top"]
        bottom = df.at[df.index[i], "fvg_bull_bottom"]
        
        # Look forward using slice
        # subset_close = close_values[i+1:]
        # Violation: Close < Bottom
        
        # We need to find the INDEX of the violation to mark the inversion
        # Let's find relative index
        future_closes = close_values[i+1:]
        violations = future_closes < bottom
        
        if violations.any():
            # first True in violations
            rel_idx = np.argmax(violations)
            abs_iloc = i + 1 + rel_idx
            
            # Set Inversion Flag at that specific candle
            df.iat[abs_iloc, df.columns.get_loc("inversion_bear")] = True

    # Check Bear FVGs -> Potential Bull Inversion
    for i in bear_ilocs:
        if i >= len(df) - 1: continue
            
        top = df.at[df.index[i], "fvg_bear_top"]
        bottom = df.at[df.index[i], "fvg_bear_bottom"]
        
        future_closes = close_values[i+1:]
        violations = future_closes > top
        
        if violations.any():
            rel_idx = np.argmax(violations)
            abs_iloc = i + 1 + rel_idx
            
            # Set Inversion Flag
            df.iat[abs_iloc, df.columns.get_loc("inversion_bull")] = True

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
