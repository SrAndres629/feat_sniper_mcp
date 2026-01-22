import pandas as pd
import numpy as np
from app.core.config import settings
from nexus_core.physics_engine.engine import physics_engine

# FVG Node Types
FVG_GRAVITY = "GRAVITY_NODE"
FVG_PROPULSION = "PROPULSION_NODE"

def detect_imbalances(df: pd.DataFrame, timeframe_minutes: int = 1) -> pd.DataFrame:
    """
    [v6.1 - DOCTORAL GEOMETRY: FVG DUALITY]
    Fair Value Gap Detection with Gravity vs. Propulsion Classification.
    
    Logic:
    - GRAVITY_NODE: HTF gaps (>15m) or aged gaps that attract price.
    - PROPULSION_NODE: LTF gaps (M1/M5) occurring with high acceleration (runaway momentum).
    
    Formula: Gravity = (FVG_Size / ATR) * Newtonian_Force * Decay_Factor
    """
    if df.empty: return df
    df = df.copy()

    # 1. Basic Detection
    df["fvg_bull"] = (df["low"].shift(-1) > df["high"].shift(1))
    df["fvg_bull_top"] = df["low"].shift(-1)
    df["fvg_bull_bottom"] = df["high"].shift(1)
    
    df["fvg_bear"] = (df["high"].shift(-1) < df["low"].shift(1))
    df["fvg_bear_top"] = df["low"].shift(1)
    df["fvg_bear_bottom"] = df["high"].shift(-1)
    
    # 2. Physics Integration
    atr = (df["high"] - df["low"]).rolling(14, min_periods=1).mean().ffill().replace(0, 1e-9)
    physics_res = physics_engine.compute_vectorized_physics(df)
    n_force = physics_res["physics_force"]
    n_accel = physics_res.get("physics_accel", pd.Series(0.0, index=df.index))
    
    # Persist Newtonian columns for diagnostics
    df["physics_force"] = n_force
    df["physics_energy"] = physics_res.get("physics_energy", 0.0)
    df["physics_accel"] = n_accel
    
    # 3. FVG Magnitude (ATR-Relative)
    bull_size = (df["fvg_bull_top"] - df["fvg_bull_bottom"]).fillna(0).clip(lower=0)
    bear_size = (df["fvg_bear_top"] - df["fvg_bear_bottom"]).fillna(0).clip(lower=0)
    fvg_size = bull_size + bear_size
    
    # 4. [DOCTORAL DUALITY] FVG Classification: Gravity vs. Propulsion
    # Rule 1: HTF Gaps (>15m) are ALWAYS Gravity (Attraction)
    # Rule 2: LTF Gaps (<=5m) with HIGH ACCELERATION are Propulsion (Runaway)
    # Rule 3: LTF Gaps with LOW ACCELERATION are still Gravity (Inefficiency)
    
    is_htf = timeframe_minutes > 15
    accel_threshold = 0.5  # Configurable threshold for "high" acceleration
    
    is_high_accel = n_accel.abs() > accel_threshold
    is_bull_trend = df["close"] > df["close"].shift(1)
    is_bear_trend = df["close"] < df["close"].shift(1)
    
    # Determine FVG Type
    df["fvg_type"] = None
    
    # HTF or low-accel = Gravity
    gravity_mask = is_htf | (~is_high_accel)
    propulsion_mask = (~is_htf) & is_high_accel
    
    fvg_mask = df["fvg_bull"].fillna(False) | df["fvg_bear"].fillna(False)
    
    df.loc[fvg_mask & gravity_mask, "fvg_type"] = FVG_GRAVITY
    df.loc[fvg_mask & propulsion_mask, "fvg_type"] = FVG_PROPULSION
    
    # 5. [DOCTORAL DECAY] Gravitational Decay Factor (λ)
    # Rule: If price moves AWAY from FVG with force, gravity decays.
    # Decay = exp(-λ * bars_since_fvg) where λ increases with distance_force
    
    decay_lambda = 0.1  # Base decay rate
    bars_since_fvg = fvg_mask.groupby((~fvg_mask).cumsum()).cumcount()
    
    # Distance from FVG (using mid-point)
    fvg_mid = (df.get("fvg_bull_top", 0.0) + df.get("fvg_bull_bottom", 0.0)) / 2
    fvg_mid = fvg_mid.where(fvg_mask).ffill()
    
    distance_from_gap = (df["close"] - fvg_mid).abs() / atr
    decay_modifier = n_force.abs().clip(0, 2)  # Higher force = faster decay
    
    df["fvg_decay_factor"] = np.exp(-decay_lambda * bars_since_fvg * (1 + decay_modifier))
    
    # 6. [DOCTORAL FORMULA] Gravity & Propulsion Scores
    fvg_ratio = fvg_size / atr
    raw_gravity = np.log1p(fvg_ratio.clip(lower=0)) * n_force.fillna(0) * df["fvg_decay_factor"]
    
    # Apply Duality: Gravity attracts, Propulsion propels
    df["fvg_gravity"] = np.where(df["fvg_type"] == FVG_GRAVITY, raw_gravity, 0.0)
    df["fvg_propulsion"] = np.where(df["fvg_type"] == FVG_PROPULSION, raw_gravity, 0.0)
    
    # 7. Mitigation State Machine
    df["fvg_mitigated"] = False
    closes = df["close"].values
    
    bull_locs = np.where(df["fvg_bull"].fillna(False).values)[0]
    bear_locs = np.where(df["fvg_bear"].fillna(False).values)[0]
    
    if len(bull_locs) > 0:
        bull_bottoms = df["fvg_bull_bottom"].values
        for i in bull_locs:
            if i >= len(df) - 1: continue
            target = bull_bottoms[i]
            future = closes[i+1:]
            violated = future < target
            if violated.any():
                rel_idx = np.argmax(violated)
                df.iloc[i + 1 + rel_idx, df.columns.get_loc("fvg_mitigated")] = True

    if len(bear_locs) > 0:
        bear_tops = df["fvg_bear_top"].values
        for i in bear_locs:
            if i >= len(df) - 1: continue
            target = bear_tops[i]
            future = closes[i+1:]
            violated = future > target
            if violated.any():
                rel_idx = np.argmax(violated)
                df.iloc[i + 1 + rel_idx, df.columns.get_loc("fvg_mitigated")] = True

    # 8. Neural Payload (Structure Tensors)
    # Gravity = Attraction (signed towards gap)
    # Propulsion = Trend Boost (signed with trend)
    df["imbalance_score"] = 0.0
    
    # Gravity contributes towards the gap (reversal signal)
    df.loc[df["fvg_bull"].fillna(False) & (df["fvg_type"] == FVG_GRAVITY), "imbalance_score"] = df["fvg_gravity"]
    df.loc[df["fvg_bear"].fillna(False) & (df["fvg_type"] == FVG_GRAVITY), "imbalance_score"] = -df["fvg_gravity"]
    
    # Propulsion contributes WITH the trend (continuation signal)
    df.loc[df["fvg_bull"].fillna(False) & (df["fvg_type"] == FVG_PROPULSION), "imbalance_score"] = df["fvg_propulsion"]
    df.loc[df["fvg_bear"].fillna(False) & (df["fvg_type"] == FVG_PROPULSION), "imbalance_score"] = df["fvg_propulsion"]  # Positive for trend
    
    return df
