import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from app.core.config import settings
from .utils import process_ticks_to_ohlcv
from .utils import process_ticks_to_ohlcv
from .engineering import apply_feat_engineering
from .vision import generate_energy_map
from .tensor import tensorize_snapshot
from .io import export_parquet, export_jsonl_gz
from nexus_core.math_engine import calculate_kalman_filter
from nexus_core.features import feat_features

from tqdm import tqdm
logger = logging.getLogger("FeatProcessor.Engine")

from nexus_core.physics_engine.engine import physics_engine
from nexus_core.structure_engine.engine import structure_engine
from app.ml.feat_processor.alpha_tensor import AlphaTensorOrchestrator
# We use AlphaTensorOrchestrator logic to ensure training matches live logic exactly.

class FeatProcessor:
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, "ohlcv_parquet"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "ticks_jsonl"), exist_ok=True)
        self.alpha = AlphaTensorOrchestrator() # Use the Master Logic

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main pipeline: Engineering -> Kinetics -> Physics Fusion."""
        return self.process_dataframe(df)

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [PHASE 14 - GOD MODE 24-CHANNEL ARRAY]
        Orchestrates the full 24-channel institutional sensory mapping.
        """
        if df.empty: return pd.DataFrame()

        df = df.reset_index(drop=True)
        if not isinstance(df.index, pd.RangeIndex):
            df.index = pd.RangeIndex(start=0, stop=len(df), step=1)

        # 1. PRICE CORE (5 Ch)
        df["log_ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        df["vol_z"] = (df["volume"] - df["volume"].rolling(50).mean()) / (df["volume"].rolling(50).std() + 1e-9)
        # Using OHLC to estimate bid/ask spread if not provided
        df["spread_z"] = (df["high"] - df["low"]).rolling(20).mean() / (df["high"] - df["low"]).rolling(100).mean().fillna(1.0)
        df["hi_low_ratio"] = (df["high"] - df["low"]) / (df["close"].rolling(20).std() + 1e-9)
        df["price_sma_dist"] = (df["close"] - df["close"].rolling(50).mean()) / (df["close"].rolling(50).std() + 1e-9)

        # 2. PHYSICS (4 Ch)
        print("🧪 [Hydration] Phase 3: Physics Tensor...")
        physics_res = physics_engine.compute_vectorized_physics(df)
        for col in physics_res.columns:
            if col not in df.columns:
                df[col] = physics_res[col].values

        payload = self.alpha.process_dataframe(df)
        
        def safe_assign(col_name, payload_key, default_val=0.0):
            val = payload.get(payload_key, default_val)
            if np.isscalar(val): return np.full(len(df), float(val))
            arr = np.array(val).flatten()
            return np.resize(arr, len(df)) if len(arr) != len(df) else arr

        df["physics_force"] = safe_assign("physics_force", "physics_force")
        df["physics_energy"] = safe_assign("physics_energy", "physics_energy")
        df["physics_entropy"] = safe_assign("physics_entropy", "physics_entropy")
        df["physics_viscosity"] = safe_assign("physics_viscosity", "physics_viscosity")

        # 3. STRUCTURE (3 Ch)
        print("🧪 [Hydration] Phase 2: Structural SMC...")
        df = structure_engine.compute_feat_index(df)
        df["structural_feat_index"] = df["feat_index"] / 100.0
        df["confluence_tensor"] = df["confluence_score"] / 5.0
        # Proximity to nearest OB/Breaker (Simplified)
        df["proximity_to_structure"] = (df["close"] - df["close"].rolling(50).min()) / (df["close"].rolling(50).max() - df["close"].rolling(50).min() + 1e-9)

        # 4. MICROSTRUCTURE (4 Ch)
        # VPIN (Toxicity) - Simplified proxy based on volume/spread skew
        df["vpin_toxicity"] = (df["volume"] * df["log_ret"].abs()).rolling(20).mean() / (df["volume"].rolling(20).mean() + 1e-9)
        df["ofi_z"] = (df["log_ret"] * df["volume"]).rolling(20).mean() / (df["volume"].rolling(20).std() + 1e-9)
        df["spread_velocity"] = df["spread_z"].diff().fillna(0)
        df["tick_density"] = df["volume"] / (df["volume"].rolling(50).mean() + 1e-9)

        # 5. MTF ALIGNMENT (4 Ch) [REFACTORED v6.0 - Zero-Lag HMA]
        # Using Hull MA for reduced phase lag
        from nexus_core.resonance_engine.filters import hull_ma
        
        hma_fast = hull_ma(df["close"], settings.RESONANCE_HMA_FAST)
        hma_medium = hull_ma(df["close"], settings.RESONANCE_HMA_MEDIUM)
        hma_slow = hull_ma(df["close"], settings.RESONANCE_HMA_SLOW)
        
        df["align_m5"] = (hma_fast > hma_medium).astype(float)
        df["align_m15"] = (hma_fast > hma_slow).astype(float)
        df["align_h1"] = (hma_medium > hma_slow).astype(float)
        df["align_h4"] = (df["close"] > hma_slow).astype(float)

        # 6. AGGRESSION / DELTA (4 Ch)
        df["buy_aggression"] = np.where(df["close"] > df["open"], df["volume"], 0) / (df["volume"] + 1e-9)
        
        # 7. TEMPORAL FRACTAL ENGINE (Module 05) [PHASE 12 INTEGRATION]
        # =============================================================
        print("🧪 [Hydration] Phase 1: Temporal Fractal Sync...")
        from nexus_core.temporal_engine.engine import temporal_engine
        df = temporal_engine.compute_temporal_tensor(df)
        
        # 8. MACRO / FINAL CLEANUP
        # Ensure we don't have infinite values
        df["sell_aggression"] = np.where(df["close"] < df["open"], df["volume"], 0) / (df["volume"] + 1e-9)
        df["delta_vol_z"] = (df["buy_aggression"] - df["sell_aggression"]).rolling(20).mean()
        df["absorption_ratio"] = df["volume"] / (df["high"] - df["low"] + 1e-9)
        df["absorption_ratio"] = (df["absorption_ratio"] - df["absorption_ratio"].rolling(50).mean()) / (df["absorption_ratio"].rolling(50).std() + 1e-9)

        # 7. [GOD MODE] Volatility Context (Sidechain)
        # Used for FeatEncoder Physics Head, not TCN Body.
        # Logic: Relative Volatility (ATR / SMA_ATR)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().fillna(1.0)
        df["volatility_context"] = (atr / (atr.rolling(50).mean() + 1e-9)).fillna(1.0)

        # 8. [V6 LATENT INPUTS] Temporal & Killzone Features for FeatEncoder
        # These are NOT part of the 24 TCN channels, but required for inference.py
        import datetime
        if 'time' in df.columns and isinstance(df['time'].iloc[0], (datetime.datetime, pd.Timestamp)):
            df['time_hour'] = df['time'].dt.hour
        else:
            # Fallback if no time column
            df['time_hour'] = (df.index % 24)
        
        # Temporal Encoding (Sine/Cosine for cyclical nature)
        df["time_sin"] = np.sin(2 * np.pi * df['time_hour'] / 24.0)
        df["time_cos"] = np.cos(2 * np.pi * df['time_hour'] / 24.0)
        
        # Killzone Intensity (London: 8-12, NY: 13-17)
        df["killzone_intensity"] = 0.0
        df.loc[(df['time_hour'] >= 8) & (df['time_hour'] < 12), "killzone_intensity"] = 1.0   # London
        df.loc[(df['time_hour'] >= 13) & (df['time_hour'] < 17), "killzone_intensity"] = 0.8  # NY
        
        # Session Weight (Asia: 0.3, London: 1.0, NY: 0.9)
        df["session_weight"] = 0.3  # Default (Asia)
        df.loc[(df['time_hour'] >= 8) & (df['time_hour'] < 16), "session_weight"] = 1.0   # London
        df.loc[(df['time_hour'] >= 13) & (df['time_hour'] < 21), "session_weight"] = 0.9  # NY
        
        # Trap Score (Simplified: based on wick size vs body)
        body_size = (df["close"] - df["open"]).abs()
        upper_wick = df["high"] - df[["close", "open"]].max(axis=1)
        lower_wick = df[["close", "open"]].min(axis=1) - df["low"]
        total_wick = upper_wick + lower_wick
        df["trap_score"] = (total_wick / (body_size + total_wick + 1e-9)).fillna(0)

        return df.fillna(0)

    def compute_latent_vector(self, row: pd.Series) -> Dict[str, float]:
        """
        Extracts exactly the 24 Institutional dimensions specified in neural_config.py.
        PLUS additional latent fields for FeatEncoder (Physics Head).
        """
        def safe_get(key, default=0.0):
            val = row.get(key, default)
            return float(val) if pd.notnull(val) else default

        # Base 24-Channel Tensor features
        res = {name: safe_get(name) for name in settings.NEURAL_FEATURE_NAMES}
        
        # [GOD MODE PATCH] Additional sidechain fields for FeatEncoder
        # These are passed to inference.py but NOT to the TCN body.
        res["volatility_context"] = safe_get("volatility_context", default=1.0)
        res["temporal_sin"] = safe_get("time_sin", default=0.0)
        res["temporal_cos"] = safe_get("time_cos", default=1.0)
        res["killzone_intensity"] = safe_get("killzone_intensity", default=0.3)
        res["session_weight"] = safe_get("session_weight", default=0.5)
        res["trap_score"] = safe_get("trap_score", default=0.0)
        
        # [MODULE 05.9] MTF Fractal Position Encoding
        # Allows FeatEncoder to discover temporal patterns like "H4-3 = expansion"
        for tf in ["m5", "m15", "m30", "h1", "h4", "d1"]:
            res[f"{tf}_position"] = safe_get(f"{tf}_position", default=2)
            res[f"{tf}_phase"] = safe_get(f"{tf}_phase", default=0.5)
        
        # Weekly cycle
        res["dow_sin"] = safe_get("dow_sin", default=0.0)
        res["dow_cos"] = safe_get("dow_cos", default=1.0)
        res["weekly_phase"] = safe_get("weekly_phase", default=0.5)
        
        return res


    def tensorize_snapshot(self, snap: Dict, feature_names: List[str]) -> np.ndarray:
        return tensorize_snapshot(snap, feature_names)
