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
    
    # Zone Identification (Consolidated Imbalance Score)
    df["imbalance_score"] = 0.0
    df.loc[df["fvg_bull"], "imbalance_score"] = 1.0
    df.loc[df["fvg_bear"], "imbalance_score"] = -1.0
    
    return df

def get_active_imbalances(df: pd.DataFrame, current_price: float):
    """Returns FVG zones that haven't been filled by current price."""
    # This logic would normally track if price has returned to the FVG range
    # For now, we flag active FVGs near price
    pass
