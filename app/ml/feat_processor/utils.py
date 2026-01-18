import pandas as pd
import numpy as np

def process_ticks_to_ohlcv(ticks_df: pd.DataFrame, timeframe: str = "15T") -> pd.DataFrame:
    """Aggregates raw ticks into OHLCV bars with high-fidelity resampling."""
    if ticks_df.empty: return pd.DataFrame()
    if "ts_ms" in ticks_df.columns: ticks_df["ts"] = pd.to_datetime(ticks_df["ts_ms"], unit="ms", utc=True)
    elif "time" in ticks_df.columns: ticks_df["ts"] = pd.to_datetime(ticks_df["time"], utc=True)
    ticks_df = ticks_df.set_index("ts").sort_index()
    ohlc = ticks_df["last_price"].resample(timeframe).ohlc()
    vol = ticks_df["last_size"].resample(timeframe).sum().rename("volume")
    tc = ticks_df["last_price"].resample(timeframe).count().rename("tick_volume")
    return pd.concat([ohlc, vol, tc], axis=1).dropna()
