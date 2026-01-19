import pandas as pd
from .fractals import identify_fractals

def detect_structural_shifts(df: pd.DataFrame) -> pd.DataFrame:
    """
    BOS (Break of Structure): Trend continuation.
    CHOCH (Change of Character): Trend reversal (First break against trend).
    """
    df = identify_fractals(df)

    # Track last confirmed fractal levels
    df["last_h_fractal"] = df["high"].where(df["fractal_high"]).ffill()
    df["last_l_fractal"] = df["low"].where(df["fractal_low"]).ffill()

    # Calculate RVOL (Relative Volume) for confirmation
    vol_col = "tick_volume" if "tick_volume" in df.columns else "volume"
    if vol_col in df.columns:
        df["rvol"] = df[vol_col] / (df[vol_col].rolling(20).mean() + 1e-9)
    else:
        df["rvol"] = 1.0

    # BOS (Break of Structure): 
    # Valid IF: (Close > Level) OR (High/Low > Level AND RVOL > 1.5)
    # This filters out low-volume wicks (stop-hunts)
    bull_level = df["last_h_fractal"].shift(1)
    bear_level = df["last_l_fractal"].shift(1)

    df["bos_bull"] = (
        (df["close"] > bull_level) | 
        ((df["high"] > bull_level) & (df["rvol"] > 1.5))
    ) & (df["close"].shift(1) <= bull_level)

    df["bos_bear"] = (
        (df["close"] < bear_level) | 
        ((df["low"] < bear_level) & (df["rvol"] > 1.5))
    ) & (df["close"].shift(1) >= bear_level)

    # CHOCH (Change of Character): Reversal - 1st break against current trend
    df["choch_bull"] = df["bos_bull"] & (df["close"].shift(1) < df["last_l_fractal"].shift(1))
    df["choch_bear"] = df["bos_bear"] & (df["close"].shift(1) > df["last_h_fractal"].shift(1))
    
    return df
