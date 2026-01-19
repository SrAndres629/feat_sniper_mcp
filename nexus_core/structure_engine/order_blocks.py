import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def detect_order_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    [v5.1 - VECTORIZED SMC] 
    Institutional Order Block (OB) Detection via Vectorized Logic.
    Definition: The last opposite candle before a validated BOS/Expansion.
    Includes Inducement (IDM) validation and Breaker transformation.
    """
    if df.empty: return df
    df = df.copy()
    
    # 0. Initialization
    df["ob_bull"] = False
    df["ob_bear"] = False
    df["ob_top"] = 0.0
    df["ob_bottom"] = 0.0
    df["is_mitigated"] = False
    
    # Needs BOS and FVG columns (Displacement validation)
    if "bos_bull" not in df.columns: return df

    # 1. VECTORIZED CANDLE SEARCH
    # We identify the candles where BOS occurs
    is_bos_bull = df["bos_bull"]
    is_bos_bear = df["bos_bear"]
    
    # For each BOS, we need the index of the 'last opposite candle'
    # Strategy: 
    # 1. Mark all bearish candles (for Bull OB)
    # 2. Use ffill to propagate the index of the last bearish candle
    # 3. Only keep those that preceded a Bullish BOS within N bars
    
    is_bearish = df["close"] < df["open"]
    is_bullish = df["close"] > df["open"]
    
    # Get indices of contrary candles
    df["last_bear_idx"] = np.where(is_bearish, np.arange(len(df)), np.nan)
    df["last_bear_idx"] = df["last_bear_idx"].ffill()
    
    df["last_bull_idx"] = np.where(is_bullish, np.arange(len(df)), np.nan)
    df["last_bull_idx"] = df["last_bull_idx"].ffill()

    # 2. DISPLACEMENT & IDM VALIDATION (Institutional Intent)
    # DISPLACEMENT: Does the expansion contain an FVG?
    has_fvg_bull = df["fvg_bull"].rolling(5).max() > 0 if "fvg_bull" in df.columns else False
    has_fvg_bear = df["fvg_bear"].rolling(5).max() > 0 if "fvg_bear" in df.columns else False
    
    # INDUCEMENT (IDM): Was minor liquidity swept before the BOS?
    # We already have 'major_h/l' which are only true if IDM was swept (from fractals.py)
    # So if BOS carries the signal, it's already validated.

    # 3. ASSIGN OBs
    # Bull OB: Last Bearish candle before Bullish BOS + Displacement
    bull_mask = is_bos_bull & (df["last_bear_idx"] > 0) & has_fvg_bull
    bear_mask = is_bos_bear & (df["last_bull_idx"] > 0) & has_fvg_bear
    
    # Apply to specific candles
    for idx in df.index[bull_mask]:
        candle_loc = int(df.at[idx, "last_bear_idx"])
        df.at[df.index[candle_loc], "ob_bull"] = True
        df.at[df.index[candle_loc], "ob_top"] = df.iloc[candle_loc]["high"]
        df.at[df.index[candle_loc], "ob_bottom"] = df.iloc[candle_loc]["low"]

    for idx in df.index[bear_mask]:
        candle_loc = int(df.at[idx, "last_bull_idx"])
        df.at[df.index[candle_loc], "ob_bear"] = True
        df.at[df.index[candle_loc], "ob_top"] = df.iloc[candle_loc]["high"]
        df.at[df.index[candle_loc], "ob_bottom"] = df.iloc[candle_loc]["low"]

    # 4. VECTORIZED MITIGATION & BREAKER LOGIC
    # Rule: Once a candle closes past an OB, it becomes a Breaker.
    # To keep it efficient, we track running levels
    
    df["active_bull_ob_top"] = df["ob_top"].where(df["ob_bull"]).ffill()
    df["active_bull_ob_bot"] = df["ob_bottom"].where(df["ob_bull"]).ffill()
    df["active_bear_ob_top"] = df["ob_top"].where(df["ob_bear"]).ffill()
    df["active_bear_ob_bot"] = df["ob_bottom"].where(df["ob_bear"]).ffill()
    
    # Mitigation: Current Low penetrates Bull OB Top
    df["is_mitigated"] = (df["low"] <= df["active_bull_ob_top"]) | (df["high"] >= df["active_bear_ob_bot"])
    
    # Breaker: Close violates the OB zone
    df["breaker_bear"] = (df["close"] < df["active_bull_ob_bot"]) & (df["close"].shift(1) >= df["active_bull_ob_bot"])
    df["breaker_bull"] = (df["close"] > df["active_bear_ob_top"]) & (df["close"].shift(1) <= df["active_bear_ob_top"])
    
    # Clean up Temp Columns
    df.drop(columns=["last_bear_idx", "last_bull_idx", "active_bull_ob_top", "active_bull_ob_bot", "active_bear_ob_top", "active_bear_ob_bot"], inplace=True)
    
    return df
