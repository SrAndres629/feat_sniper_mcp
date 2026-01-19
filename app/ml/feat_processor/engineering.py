import pandas as pd
import numpy as np
from nexus_core.structure_engine import structure_engine
from nexus_core.acceleration import acceleration_engine

def apply_feat_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Applies Structure, Acceleration, and Space logic (Level 66 Architecture)."""
    if df.empty: return df
    df = df.copy()
    
    # [LEVEL 50] STATIONARY LOG-DYNAMICS
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
    
    # [PHASE 13 - INSTITUTIONAL PROXY]
    # OFI (Order Flow Imbalance) Proxy based on Bar Microstructure
    # Measures the 'Aggression' of the candle vs its footprint
    df["bar_spread"] = df["high"] - df["low"]
    df["body_size"] = (df["close"] - df["open"]).abs()
    df["ofi_proxy"] = (df["close"] - df["open"]) * df["volume"] / (df["bar_spread"] + 1e-9)
    df["ofi_z"] = (df["ofi_proxy"] - df["ofi_proxy"].rolling(20).mean()) / (df["ofi_proxy"].rolling(20).std() + 1e-9)

    # Z-Score Normalization (Global Stationary Signals)
    df["price_z_score"] = (df["close"] - df["close"].rolling(50).mean()) / (df["close"].rolling(50).std() + 1e-9)
    df["volume_z_score"] = (df["volume"] - df["volume"].rolling(50).mean()) / (df["volume"].rolling(50).std() + 1e-9)
    
    # External Engines
    df = structure_engine.detect_structural_shifts(df)
    df = structure_engine.detect_zones(df)
    
    # Statistical Metrics for Inference
    roll = df["close"].rolling(window=20)
    df["skew"] = roll.skew().fillna(0)
    df["kurtosis"] = roll.kurt().fillna(0)
    
    # [v5.0] LOCAL ENTROPY (Market Chaos Tracking)
    df["entropy"] = df["log_ret"].rolling(20).apply(lambda x: -np.sum(np.abs(x)*np.log(np.abs(x)+1e-9))).fillna(0)
    
    # Physics Metrics
    df["energy_z"] = df["volume_z_score"]
    if "time" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            # Robust datetime conversion
            df["time"] = pd.to_datetime(df["time"], unit='s', errors='coerce')
        df["cycle_prog"] = (df["time"].dt.minute % 60) / 60.0
    else:
        df["cycle_prog"] = 0.5 

    accel = acceleration_engine.compute_acceleration_features(df)
    # Ensure all required acceleration columns are merged
    accel_cols = ["disp_norm", "vol_z", "candle_momentum", "rvol", "cvd_proxy", "accel_score"]
    df = df.join(accel[[c for c in accel_cols if c in accel.columns]], rsuffix='_accel')
    
    # Cleanup NaN to prevent Neural Crash
    return df.fillna(0)
