import pandas as pd

def identify_fractals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bill Williams Fractals (5-candle pattern).
    A fractal is confirmed when the middle candle is the highest/lowest of 5.
    """
    # High Fractals
    df["fractal_high"] = (
        (df["high"].shift(2) > df["high"].shift(4))
        & (df["high"].shift(2) > df["high"].shift(3))
        & (df["high"].shift(2) > df["high"].shift(1))
        & (df["high"].shift(2) > df["high"])
    )

    # Low Fractals
    df["fractal_low"] = (
        (df["low"].shift(2) < df["low"].shift(4))
        & (df["low"].shift(2) < df["low"].shift(3))
        & (df["low"].shift(2) < df["low"].shift(1))
        & (df["low"].shift(2) < df["low"])
    )
    return df
