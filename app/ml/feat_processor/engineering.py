import pandas as pd
import numpy as np
from nexus_core.structure_engine import structure_engine
from nexus_core.acceleration import acceleration_engine

def apply_feat_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Applies Structure, Acceleration, and Space logic (Level 66 Architecture)."""
    if df.empty: return df
    df = df.copy()
    
    # [LEVEL 48] Data Normalization - Ensure 'volume' exists
    if "volume" not in df.columns and "tick_volume" in df.columns:
        df["volume"] = df["tick_volume"]
    elif "volume" not in df.columns:
        df["volume"] = 1.0 # Minimal fallback for volume-less data
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
    # Statistical Metrics for Inference
    roll = df["close"].rolling(window=20)
    df["skew"] = roll.skew().fillna(0)
    df["kurtosis"] = roll.kurt().fillna(0)
    # Entropy approx
    df["entropy"] = df["log_ret"].rolling(20).apply(lambda x: -np.sum(x*np.log(np.abs(x)+1e-9))).fillna(0)
    
    # Physics Metrics
    df["energy_z"] = df["volume_z_score"] # Alias
    if "time" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["time"]):
            first_val = df["time"].iloc[0]
            # Try to infer if it's a unix timestamp (int/float)
            is_numeric = isinstance(first_val, (int, float, np.integer, np.floating))
            if not is_numeric:
                try: 
                    first_val = float(first_val)
                    is_numeric = True
                except: pass
            
            unit = 's' if (is_numeric and float(first_val) > 1e9) else None
            df["time"] = pd.to_datetime(df["time"], unit=unit, errors='coerce')
        df["cycle_prog"] = (df["time"].dt.minute % 60) / 60.0
    else:
        df["cycle_prog"] = 0.5 # Default

    accel = acceleration_engine.compute_acceleration_features(df)
    cols = ["disp_norm", "vol_z", "candle_momentum", "rvol", "cvd_proxy", "accel_score"]
    df = df.join(accel[[c for c in cols if c in accel.columns]])
    return df
