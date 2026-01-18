import pandas as pd
import numpy as np
from nexus_core.structure_engine import structure_engine
from nexus_core.acceleration import acceleration_engine

def apply_feat_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Applies Structure, Acceleration, and Space logic (Level 66 Architecture)."""
    if df.empty: return df
    df = df.copy()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
    # Z-Score Normalization
    df["rolling_mean_50"] = df["close"].rolling(window=50).mean()
    df["rolling_std_50"] = df["close"].rolling(window=50).std()
    df["price_z_score"] = (df["close"] - df["rolling_mean_50"]) / (df["rolling_std_50"] + 1e-9)
    df["vol_mean_50"] = df["volume"].rolling(window=50).mean()
    df["vol_std_50"] = df["volume"].rolling(window=50).std()
    df["volume_z_score"] = (df["volume"] - df["vol_mean_50"]) / (df["vol_std_50"] + 1e-9)
    # External Engines
    df = structure_engine.detect_structural_shifts(df)
    df = structure_engine.detect_zones(df)
    accel = acceleration_engine.compute_acceleration_features(df)
    cols = ["disp_norm", "vol_z", "candle_momentum", "rvol", "cvd_proxy", "accel_score"]
    df = df.join(accel[[c for c in cols if c in accel.columns]])
    return df
