import pandas as pd
import numpy as np
from nexus_core.structure_engine import structure_engine
from nexus_core.acceleration import acceleration_engine
from app.ml.feat_processor.kinetics import calculate_multifractal_layers
from app.ml.feat_processor.space import TensorTopologist

tensor_topologist = TensorTopologist()

def apply_feat_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Applies Structure, Acceleration, and Space logic (Level 66 Architecture)."""
    if df.empty: return df
    df = df.copy()
    
    # [LEVEL 50] STATIONARY LOG-DYNAMICS
    # [FIX] Robust Volume Column Handing (MT5 Standard)
    if "volume" not in df.columns:
        if "tick_volume" in df.columns:
            df["volume"] = df["tick_volume"]
        elif "real_volume" in df.columns:
             df["volume"] = df["real_volume"]
        else:
             # Fallback if no volume data (Crypto/Forex gaps)
             df["volume"] = 1.0 

    df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
    
    # [PHASE 13 - INSTITUTIONAL PROXY]
    # OFI (Order Flow Imbalance) Proxy based on Bar Microstructure
    # Measures the 'Aggression' of the candle vs its footprint
    df["bar_spread"] = df["high"] - df["low"]
    df["body_size"] = (df["close"] - df["open"]).abs()
    df["ofi_proxy"] = (df["close"] - df["open"]) * df["volume"] / (df["bar_spread"] + 1e-9)
    df["ofi_z"] = (df["ofi_proxy"] - df["ofi_proxy"].rolling(20, min_periods=1).mean()) / (df["ofi_proxy"].rolling(20, min_periods=1).std() + 1e-9)

    # Z-Score Normalization (Global Stationary Signals)
    df["price_z_score"] = (df["close"] - df["close"].rolling(50, min_periods=1).mean()) / (df["close"].rolling(50, min_periods=1).std() + 1e-9)
    df["volume_z_score"] = (df["volume"] - df["volume"].rolling(50, min_periods=1).mean()) / (df["volume"].rolling(50, min_periods=1).std() + 1e-9)
    
    # External Engines
    df = structure_engine.detect_structural_shifts(df)
    # [DOCTORAL UPGRADE] Breakers & Inversion exist in DF after structural shifts/imbalances?
    # Wait, detect_zones (deleted) called structure functions internally in old logic.
    # We must ensure detect_imbalances and detect_order_blocks are called.
    df = structure_engine.detect_imbalances(df)
    df = structure_engine.detect_order_blocks(df)
    # Zones are handled by Space Tensor now
    
    # 1. KINETIC VECTOR RECRUITMENT (9-Dim One-Hot)
    # ---------------------------------------------
    df = calculate_multifractal_layers(df) 
    # This adds: kinetic_is_expansion, kinetic_is_compression, etc.

    # 2. STRUCTURAL GEOMETRY VALIDATION (Breakers & Inversion)
    # --------------------------------------------------------
    # Neural Networks need Float, not Bool
    cols_to_float = [
        "breaker_bull", "breaker_bear", 
        "inversion_bull", "inversion_bear",
        "ob_bull", "ob_bear",
        "fvg_bull", "fvg_bear"
    ]
    for c in cols_to_float:
        if c in df.columns:
            df[c] = df[c].astype(float)
        else:
            df[c] = 0.0

    # 3. SPACE TENSOR PROJECTION (Gaussian Fields)
    # --------------------------------------------
    # We need a vectorized approach for Proximity. 
    # Since space.py is scalar, we'll implement a fast proxy here or update space.py later.
    # Proxy: "Distance to nearest active structure"
    # We can use the 'fvg_bull_top', 'ob_bull_top' colums to find nearest levels.
    
    # [OPTIMIZATION] For now, we'll map the Confluence Score (which sums up the intensity)
    # structure_engine.calculate_confluence_score(df) handles this.
    df = structure_engine.calculate_confluence_score(df)
    
    # Normalize Confluence for Neural Input (-1 to 1 or 0 to 1)
    # Max score is roughly 5.0
    df["space_intensity"] = (df["confluence_score"] / 5.0).clip(upper=1.0)
    
    # Statistical Metrics for Inference
    roll = df["close"].rolling(window=20, min_periods=1)
    df["skew"] = roll.skew().fillna(0)
    df["kurtosis"] = roll.kurt().fillna(0)
    
    # [v5.0] LOCAL ENTROPY (Market Chaos Tracking)
    df["entropy"] = df["log_ret"].rolling(20, min_periods=1).apply(lambda x: -np.sum(np.abs(x)*np.log(np.abs(x)+1e-9))).fillna(0)
    
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
    accel_cols = ["disp_norm", "vol_z", "candle_momentum", "rvol", "cvd_proxy", "accel_score", "kinetic_context"]
    df = df.join(accel[[c for c in accel_cols if c in accel.columns]], rsuffix='_accel')
    
    # [PHASE 4 - MACRO SENTINEL INTEGRATION]
    # ======================================
    # Inject macro-economic awareness into the Neural Pipeline
    from nexus_core.fundamental_engine import fundamental_engine
    
    try:
        macro_tensor = fundamental_engine.get_macro_regime_tensor(currencies=["USD", "EUR", "GBP"])
        
        # Add as constant columns (same value for all rows in this batch)
        # In live trading, this would be updated per-candle fetch.
        df["macro_safe"] = macro_tensor["macro_safe"]
        df["macro_caution"] = macro_tensor["macro_caution"]
        df["macro_danger"] = macro_tensor["macro_danger"]
        df["position_multiplier"] = macro_tensor["position_multiplier"]
        df["minutes_to_event"] = macro_tensor["minutes_to_event"]
    except Exception:
        # Fallback if fundamental engine unavailable
        df["macro_safe"] = 1.0
        df["macro_caution"] = 0.0
        df["macro_danger"] = 0.0
        df["position_multiplier"] = 1.0
        df["minutes_to_event"] = 1.0
    
    # Cleanup NaN to prevent Neural Crash
    with pd.option_context('future.no_silent_downcasting', True):
        return df.fillna(0).infer_objects(copy=False)
