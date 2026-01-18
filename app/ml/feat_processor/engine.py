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
        
        # Physics & Form
        atr14 = (df["high"] - df["low"]).rolling(14).mean().fillna(df["close"]*0.001) + 1e-9
        df["feat_form"] = (df["high"] - df["low"]) / atr14
        vwap = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()
        df["feat_space"] = np.abs(df["close"] - vwap) / atr14
        
        return df

    def compute_latent_vector(self, row: pd.Series) -> Dict[str, float]:
        atr = row.get("atr14", 1.0) + 1e-9
        close = row["close"]
        return {
            "dist_micro": row.get("dist_micro", 0.0), "dist_struct": row.get("dist_structure", 0.0), "dist_macro": row.get("dist_macro", 0.0), "dist_bias": row.get("dist_bias", 0.0),
            "layer_alignment": row.get("layer_alignment", 0.0), "dist_poc": (close - row.get("poc_price", close)) / atr,
            "pos_in_va": 1.0 if (row.get("val", -1) <= close <= row.get("vah", -1)) else 0.0,
            "form": row.get("feat_form", 0.5), "space": row.get("feat_space", 0.5), "kalman_score": row.get("kalman_score", 0.0)
        }
