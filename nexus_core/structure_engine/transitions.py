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

    # BOS: Close exceeds last fractal in trend direction
    df["bos_bull"] = (df["close"] > df["last_h_fractal"].shift(1)) & (
        df["close"].shift(1) <= df["last_h_fractal"].shift(1)
    )
    df["bos_bear"] = (df["close"] < df["last_l_fractal"].shift(1)) & (
        df["close"].shift(1) >= df["last_l_fractal"].shift(1)
    )

    # CHOCH (Change of Character): Reversal - 1st break against current trend
    df["choch_bull"] = df["bos_bull"] & (df["close"].shift(1) < df["last_l_fractal"].shift(1))
    df["choch_bear"] = df["bos_bear"] & (df["close"].shift(1) > df["last_h_fractal"].shift(1))
    
    return df
