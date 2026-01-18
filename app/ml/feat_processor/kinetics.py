import pandas as pd
import numpy as np
from app.core.config import settings

def calculate_multifractal_layers(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Cloud States (Micro, Structure, Macro) vectorized (Channel A)."""
    if df.empty: return df
    df = df.copy()
    periods = {"micro": settings.LAYER_MICRO_PERIODS, "structure": settings.LAYER_OPERATIVE_PERIODS, "macro": settings.LAYER_MACRO_PERIODS}
    atr = (df["high"] - df["low"]).rolling(14).mean().fillna(df["close"]*0.001)
    centroids = {}
    for name, p_list in periods.items():
        emas = [df["close"].ewm(span=p, adjust=False).mean() for p in p_list]
        centroid = pd.concat(emas, axis=1).mean(axis=1)
        centroids[name] = centroid
        df[f"{name}_compression"] = pd.concat(emas, axis=1).std(axis=1) / (atr + 1e-9)
        df[f"dist_{name}"] = (df["close"] - centroid) / (atr + 1e-9)
        df[f"{name}_slope"] = centroid.diff(1) / (atr + 1e-9)
        
    bias = df["close"].ewm(alpha=1.0/settings.LAYER_BIAS_PERIOD, adjust=False).mean()
    df["dist_bias"] = (df["close"] - bias) / (atr + 1e-9)
    df["bias_slope"] = bias.diff(1) / (atr + 1e-9)
    
    aligned_bull = (df["close"] > centroids["micro"]) & (centroids["micro"] > centroids["structure"]) & (centroids["structure"] > centroids["macro"])
    aligned_bear = (df["close"] < centroids["micro"]) & (centroids["micro"] < centroids["structure"]) & (centroids["structure"] < centroids["macro"])
    df["layer_alignment"] = 0.0
    df.loc[aligned_bull, "layer_alignment"], df.loc[aligned_bear, "layer_alignment"] = 1.0, -1.0
    
    df["kinetic_pattern_id"] = 0
    df.loc[df["layer_alignment"] != 0, "kinetic_pattern_id"] = 1
    df.loc[(df["micro_compression"] < 0.3) | (df["structure_compression"] < 0.3), "kinetic_pattern_id"] = 2
    return df
