import pandas as pd
import numpy as np

def detect_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    [v5.0 - SMC DOCTORAL] Institutional Order Block (OB) Detection.
    Definition: The last opposite candle before a validated Major BOS/Expansion.
    Includes Mitigation Tracking.
    """
    if df.empty: return df
    df = df.copy()
    
    df["ob_bull"] = False
    df["ob_bull_top"] = 0.0
    df["ob_bull_bottom"] = 0.0
    df["ob_bear"] = False
    df["ob_bear_top"] = 0.0
    df["ob_bear_bottom"] = 0.0
    df["is_mitigated"] = False

    # A Major BOS is the trigger for an OB
    bos_bull_indices = df.index[df["bos_bull"]].tolist()
    bos_bear_indices = df.index[df["bos_bear"]].tolist()

    # 1. Bullish Order Blocks (Last Bearish Candle before Bullish BOS)
    for idx in bos_bull_indices:
        # Look back for the start of the expansion
        found_ob = False
        for i in range(df.index.get_loc(idx)-1, 0, -1):
            if df.iloc[i]["close"] < df.iloc[i]["open"]: # Bearish candle
                df.at[idx, "ob_bull"] = True
                df.at[idx, "ob_bull_top"] = df.iloc[i]["high"]
                df.at[idx, "ob_bull_bottom"] = df.iloc[i]["low"]
                found_ob = True
                break
            if i < df.index.get_loc(idx) - 10: break # Max 10 candles back
            
    # 2. Bearish Order Blocks (Last Bullish Candle before Bearish BOS)
    for idx in bos_bear_indices:
        found_ob = False
        for i in range(df.index.get_loc(idx)-1, 0, -1):
            if df.iloc[i]["close"] > df.iloc[i]["open"]: # Bullish candle
                df.at[idx, "ob_bear"] = True
                df.at[idx, "ob_bear_top"] = df.iloc[i]["high"]
                df.at[idx, "ob_bear_bottom"] = df.iloc[i]["low"]
                found_ob = True
                break
            if i < df.index.get_loc(idx) - 10: break

    # 3. MITIGATION TRACKING (The Memory of Touch)
    # A Bullish OB is mitigated if price returns to its TOP
    # We'll flag current price mitigation
    last_bull_ob = df[df["ob_bull"]].tail(1)
    if not last_bull_ob.empty:
        ob_top = last_bull_ob["ob_bull_top"].values[0]
        # Check if any candle AFTER the OB touched it
        df["is_mitigated"] = df["low"] <= ob_top
        
    return df
