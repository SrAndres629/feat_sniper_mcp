import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .utils import process_ticks_to_ohlcv
from .engineering import apply_feat_engineering
from .kinetics import calculate_multifractal_layers
from .vision import generate_energy_map
from .tensor import tensorize_snapshot
from .io import export_parquet, export_jsonl_gz
from nexus_core.math_engine import calculate_kalman_filter
from nexus_core.features import feat_features

logger = logging.getLogger("FeatProcessor.Engine")

class FeatProcessor:
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, "ohlcv_parquet"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "ticks_jsonl"), exist_ok=True)

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main pipeline: Engineering -> Kinetics -> Physics Fusion."""
        return self.process_dataframe(df)

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alias for process_data to maintain compatibility."""
        # [PHASE 13 - DOCTORAL STRUCTURE INTEGRATION]
        # Full Institutional Stack (SMC/ICT Architecture)
        from nexus_core.structure_engine import (
            identify_fractals, detect_structural_shifts, 
            detect_imbalances, detect_liquidity_pools, detect_order_blocks
        )
        df = identify_fractals(df)
        df = detect_structural_shifts(df)
        df = detect_imbalances(df)
        df = detect_liquidity_pools(df)
        df = detect_order_blocks(df)
        
        df = apply_feat_engineering(df)
        df = calculate_multifractal_layers(df)
        
        # [LEVEL 35] ZERO-LAG TRACKING
        close_prices = df["close"].to_numpy()
        df["kalman_price"] = calculate_kalman_filter(close_prices)
        df["kalman_deviation"] = (df["close"] - df["kalman_price"]) / (df["kalman_price"] + 1e-9)
        df["kalman_score"] = np.abs(df["kalman_deviation"]) * 1000.0
        
        # [LEVEL 41] INSTITUTIONAL PVP
        pvp = feat_features._compute_pvp_metrics(df)
        df["poc_price"] = pvp.get("poc_price", df["close"].mean())
        df["vah"], df["val"] = pvp.get("vah", df["high"].max()), pvp.get("val", df["low"].min())
        df["energy_score"] = pvp.get("total_volume", 0.0)
        
        # [LEVEL 50 - DOCTORAL TOPOLOGY]
        atr = (df["high"] - df["low"]).rolling(14).mean().ffill() + 1e-9
        
        # Normalization of Hierarchical Distances (Distance from Price to Level)
        df["dist_swing_h"] = (df["close"] - df["swing_h"]) / atr
        df["dist_swing_l"] = (df["close"] - df["swing_l"]) / atr
        df["dist_internal_h"] = (df["close"] - df["internal_h"]) / atr
        df["dist_internal_l"] = (df["close"] - df["internal_l"]) / atr

        # [PHASE 13 - INTENT CLASSIFICATION]
        df["is_protected_h"] = (df["major_h"]) & (df["bos_bear"].rolling(50).any())
        df["is_protected_l"] = (df["major_l"]) & (df["bos_bull"].rolling(50).any())
        
        # Space & Form
        df["feat_form"] = (df["high"] - df["low"]) / atr
        vwap = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / (df["volume"].cumsum() + 1e-9)
        df["feat_space"] = (df["close"] - vwap) / atr
        
        return df

    def compute_latent_vector(self, row: pd.Series) -> Dict[str, float]:
        """
        Extracts exactly the 18 dimensions specified in neural_config.py for Inference.
        Ensures strict mapping alignment for the Physics Gating Unit.
        """
        def safe_get(key, default=0.0):
            val = row.get(key, default)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return float(val)

        atr = safe_get("atr14", 1.0) + 1e-9
        close = safe_get("close", 0.0)
        
        # Mapping dict according to settings.NEURAL_FEATURE_NAMES
        # [v5.0] SATURATED TOPOLOGIC VECTOR WITH TRAP AWARENESS
        return {
            "dist_micro": safe_get("dist_internal_h"), # Internal Structure
            "dist_struct": safe_get("dist_swing_h"),   # External Structure
            "dist_macro": safe_get("dist_swing_l"),    # Anchor Points
            "dist_bias": safe_get("dist_poc"),         # Gravity Bias
            "layer_alignment": safe_get("layer_alignment"),
            "kinetic_coherence": safe_get("kinetic_coherence"),
            "kinetic_pattern_id": int(safe_get("kinetic_pattern_id")),
            "dist_poc": safe_get("dist_poc"),
            "pos_in_va": 1.0 if (safe_get("val", -1) <= close <= safe_get("vah", -1)) else 0.0,
            "ofi_z": safe_get("ofi_z"),
            "energy": safe_get("energy_z"),
            "skew": safe_get("skew"),
            "entropy": safe_get("entropy"),
            "form": safe_get("feat_form", 0.5),
            "space": safe_get("feat_space", 0.5),
            "accel": safe_get("accel_score"),
            "range_pos": safe_get("range_pos"),
            "trap_score": safe_get("trap_score")  # [PREDATORY AWARENESS]
        }

    def tensorize_snapshot(self, snap: Dict, feature_names: List[str]) -> np.ndarray:
        return tensorize_snapshot(snap, feature_names)
