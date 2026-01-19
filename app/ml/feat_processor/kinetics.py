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
    
    # [NEURAL HYGIENE] One-Hot Encoding for Tensors
    # 0 = Noise, 1 = Expansion, 2 = Compression
    # We replace strict IDs with scalar flags
    
    df["kinetic_is_expansion"] = 0.0
    df["kinetic_is_compression"] = 0.0
    
    # Expansion: Alignment nonzero
    df.loc[df["layer_alignment"] != 0, "kinetic_is_expansion"] = 1.0
    
    # Compression: Low Vol/Spread
    knots = (df["micro_compression"] < 0.3) | (df["structure_compression"] < 0.3)
    df.loc[knots, "kinetic_is_compression"] = 1.0
    
    # [MATH SENIOR FULLSTACK - Kinetic Physicist]
    # DOCTORAL FORMULA: Force = (Body / (ATR + ε)) * RVOL
    # - Body/ATR = Normalized Displacement (Dimensionless)
    # - RVOL = Relative Volume (Effort relative to market norm)
    
    body_size = (df["close"] - df["open"]).abs()
    range_size = (df["high"] - df["low"])
    
    # 1. RVOL FIRST (Needed for FEAT Force)
    if "volume" in df.columns:
        # [subskill_computational] Vectorized Rolling Mean (No Loops)
        vol_mean = df["volume"].rolling(20).mean()
        df["rvol"] = df["volume"] / (vol_mean + 1e-9)
        
        # 2. FEAT Force (Doctoral)
        # Force = (Body / ATR) * RVOL
        df["feat_force"] = (body_size / (atr + 1e-9)) * df["rvol"]
    else:
        df["feat_force"] = 0.0
        df["rvol"] = 0.0
        
    # 3. Wick Ratio (Rejection Power)
    # High Wick = Rejection. Low Wick = Full Body (Intent).
    df["wick_ratio"] = (range_size - body_size) / (range_size + 1e-9)

    # [DOCTORAL 3.0] NEURAL INTROSPECTION
    
    # ANTI-CRASH LOGIC (Context Integration)
    # If FEAT Force is High but RVOL is Low (< 0.8), it's a Fakeout (Liquidity Void).
    # We zero out the Force to prevent the Neural Net from seeing "Strength".
    mask_fakeout = (df["feat_force"] > 2.0) & (df["rvol"] < 0.8)
    df.loc[mask_fakeout, "feat_force"] = 0.0
    
    # [DOCTORAL 4.0] EXPLICIT ACCELERATION & EFFICIENCY
    # 1. Acceleration (Second Derivative)
    # We have slopes: {name}_slope. Need slope.diff()
    for name in ["micro", "structure", "macro"]:
        if f"{name}_slope" in df.columns:
            df[f"{name}_accel"] = df[f"{name}_slope"].diff(1).fillna(0.0)
            
    # 2. FEAT Efficiency (Anti-HFT)
    # Body Size / (Volume * ATR). 
    # High Eff = Institutional. Low Eff = Churn.
    if "volume" in df.columns:
         feat_eff_raw = body_size / ((df["volume"] * atr) + 1e-9)
         # Normalize (Optional: Z-Score or robust scaling)
         # For raw tensor input, robust scaling is preferred, but let's effectively rely on the ratio.
         df["feat_efficiency"] = feat_eff_raw
    else:
         df["feat_efficiency"] = 0.0

    # 4. FEAT Force Z-Score (Anomaly Detection)
    # Measures how "Unusual" this force is.
    df["feat_force_z"] = (df["feat_force"] - df["feat_force"].rolling(50).mean()) / (df["feat_force"].rolling(50).std() + 1e-9)
    
    # 5. Absorption State Learning (One-Hot Tensor)
    # We want to tell the Neural Net: "Was this Impulse Validated?"
    
    # Identify Impulses (Body > 1.5 ATR) & (Force > 2.0 - Optional Sync w/ Logic)
    # Let's use strict Body for visual impulse first.
    is_impulse_bull = (body_size > 1.5*atr) & (df["close"] > df["open"])
    is_impulse_bear = (body_size > 1.5*atr) & (df["close"] < df["open"])
    
    # Validation Levels
    limit_bull = df["low"] + 0.5 * body_size
    limit_bear = df["high"] - 0.5 * body_size
    
    # Look ahead 3 candles (using shift -1, -2, -3)
    # Check if ANY future candle violates the limit
    violation_bull = (df["close"].shift(-1) < limit_bull) | (df["close"].shift(-2) < limit_bull) | (df["close"].shift(-3) < limit_bull)
    violation_bear = (df["close"].shift(-1) > limit_bear) | (df["close"].shift(-2) > limit_bear) | (df["close"].shift(-3) > limit_bear)
    
    # One-Hot Encoding Initialization
    df["kinetic_state_neutral"] = 1.0
    df["kinetic_state_impulse"] = 0.0
    df["kinetic_state_confirmed"] = 0.0
    df["kinetic_state_failed"] = 0.0
    
    # CONFIRMED STATE (Impulse + No Violation)
    # We mark the IMPULSE candle as "CONFIRMED" retrospectively (for training labelling)
    mask_confirmed_bull = is_impulse_bull & ~violation_bull
    mask_confirmed_bear = is_impulse_bear & ~violation_bear
    
    df.loc[mask_confirmed_bull | mask_confirmed_bear, "kinetic_state_confirmed"] = 1.0
    df.loc[mask_confirmed_bull | mask_confirmed_bear, "kinetic_state_neutral"] = 0.0
    
    # FAILED STATE
    mask_failed_bull = is_impulse_bull & violation_bull
    mask_failed_bear = is_impulse_bear & violation_bear
    
    df.loc[mask_failed_bull | mask_failed_bear, "kinetic_state_failed"] = 1.0
    df.loc[mask_failed_bull | mask_failed_bear, "kinetic_state_neutral"] = 0.0
    
    # IMPULSE STATE (Simulated)
    # In live trading, an impulse starts as "IMPULSE" then moves to MONITORING/CONFIRMED.
    # For training, if we want to show "Fresh Impulse", we can check if T+1 hasn't happened yet? 
    # But batch processing sees all. 
    # Let's imply 'IMPULSE' is the trigger event. 
    # We set IMPULSE flag on the candle itself as well? 
    # User requested [NEUTRAL, IMPULSE, CONFIRMED, FAILED].
    # Effectively CONFIRMED and FAILED are subsets of IMPULSE in hindsight.
    # We'll mark the candle as IMPULSE as well if it qualifies as one.
    
    df.loc[is_impulse_bull | is_impulse_bear, "kinetic_state_impulse"] = 1.0
    
    # Note: A candle can be IMPULSE=1 AND MEANWHILE CONFIRMED=1 (Retroactive). 
    # Or IMPULSE=1 and FAILED=1.
    
    # Remove old scalar if prohibited, but keeping intermediates is fine if not outputted.
    
    return df
