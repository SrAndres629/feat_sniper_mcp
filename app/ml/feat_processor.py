"""
FEAT NEXUS: DATA PROCESSOR & OPTIMIZER
======================================
Handles efficient data export (Parquet/JSONL) and FEAT feature engineering pipeline.
Bridges raw MT5/cTrader data with the Hybrid Intelligence AI.
"""

import pandas as pd
import numpy as np
import json
import gzip
import os
from typing import List, Dict, Any

# Internal Engines
from nexus_core.structure_engine import structure_engine
from nexus_core.acceleration import acceleration_engine


class FeatProcessor:
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "ohlcv_parquet"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "ticks_jsonl"), exist_ok=True)

    def process_ticks_to_ohlcv(
        self, ticks_df: pd.DataFrame, timeframe: str = "15T"
    ) -> pd.DataFrame:
        """
        Aggregates raw ticks into OHLCV bars with FEAT indicators.
        """
        if ticks_df.empty:
            return pd.DataFrame()

        # Ensure timestamp index
        if "ts_ms" in ticks_df.columns:
            ticks_df["ts"] = pd.to_datetime(ticks_df["ts_ms"], unit="ms", utc=True)
        elif "time" in ticks_df.columns:
            ticks_df["ts"] = pd.to_datetime(ticks_df["time"], utc=True)

        ticks_df = ticks_df.set_index("ts").sort_index()

        # Resample OHLCV
        ohlc = ticks_df["last_price"].resample(timeframe).ohlc()
        vol = (
            ticks_df["last_size"].resample(timeframe).sum().rename("volume")
        )  # Real volume proxy
        tick_count = (
            ticks_df["last_price"].resample(timeframe).count().rename("tick_volume")
        )

        df = pd.concat([ohlc, vol, tick_count], axis=1).dropna()

        # Calculate FEAT Core Indicators
        df = self.apply_feat_engineering(df)

        return df

    def apply_feat_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Structure, Acceleration, and Space logic to the DataFrame.
        """
        # 1. Structure Engine (Fractals, BOS, Zones)
        df = structure_engine.detect_structural_shifts(df)
        df = structure_engine.detect_zones(df)  # Supply/Demand zone detection

        # 2. Acceleration Engine (Momentum, RVOL, CVD)
        # Note: acceleration engine expects some columns like 'volume', 'high', 'low'
        accel_features = acceleration_engine.compute_acceleration_features(df)

        # Merge acceleration features back
        df = df.join(
            accel_features[
                [
                    "disp_norm",
                    "vol_z",
                    "candle_momentum",
                    "rvol",
                    "cvd_proxy",
                    "accel_score",
                    "accel_flag",
                ]
            ]
        )

        # 3. Standard Indicators (TA-Lib / Pandas implementation for speed)
        # EMA
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

        # ATR (Already in acceleration, but ensuring)
        df["atr14"] = (
            (df["high"] - df["low"]).rolling(14).mean()
        )  # Simple proxy if TA not avail

        return df

    def export_parquet(self, df: pd.DataFrame, filename: str):
        """
        Saves DataFrame to Parquet format (Snappy compression).
        """
        path = os.path.join(self.output_dir, "ohlcv_parquet", filename)
        # Requires pyarrow or fastparquet
        try:
            df.to_parquet(path, compression="snappy")
            print(f"✅ Exported Parquet: {path}")
        except Exception as e:
            print(f"❌ Failed export Parquet: {e}")

    def export_ticks_jsonl(self, ticks_data: List[Dict], filename: str):
        """
        Saves dictionary list to Compressed JSONL (.jsonl.gz).
        """
        path = os.path.join(self.output_dir, "ticks_jsonl", filename)
        try:
            with gzip.open(path, "wt", encoding="utf-8") as f:
                for record in ticks_data:
                    f.write(json.dumps(record) + "\n")
            print(f"✅ Exported JSONL.GZ: {path}")
        except Exception as e:
            print(f"❌ Failed export JSONL: {e}")



    def tensorize_snapshot(self, snapshot: Dict[str, Any], feature_names: List[str]) -> np.ndarray:
        """
        [LEVEL 25] THE ADAPTOR (Translator Universal).
        Converts a raw JSON/Dict snapshot into a Normalized Feature Vector for the Neural Network.
        
        Rules:
        - FEAT Scores (0-100) -> 0.0-1.0
        - Acceleration (0-10) -> 0.0-1.0
        - RSI (0-100) -> 0.0-1.0
        - Missing Data -> 0.0 (Safe Fill)
        """
        vector = []
        for feature in feature_names:
            # Safe Get with 0.0 default
            val = snapshot.get(feature, 0.0)
            
            # Ensure float
            try:
                val = float(val)
            except:
                val = 0.0
            
            # [NORMALIZATION LOGIC]
            # Identify feature type by name substring
            
        # [NORMALIZATION LOGIC]
            # 1. Structure/Regime/Trend STRINGS -> FLOAT Mapping
            if isinstance(val, str):
                val_upper = val.upper()
                if "BOS" in val_upper:
                    if "BULL" in val_upper: val = 1.0
                    elif "BEAR" in val_upper: val = -1.0
                    else: val = 0.0
                elif "TREND" in val_upper:
                    if "UP" in val_upper or "BULL" in val_upper: val = 1.0
                    elif "DOWN" in val_upper or "BEAR" in val_upper: val = -1.0
                    else: val = 0.0
                elif "RANGING" in val_upper or "NEUTRAL" in val_upper:
                    val = 0.0
                else:
                    # Generic String fallback (hash or 0)
                    val = 0.0
            
            # 2. FEAT/Structure Scores (Range 0-100)
            elif "score" in feature or "rsi" in feature:
                # heuristic: if value is > 1.0, assume it's raw 0-100. 
                # If it's already 0-1, don't divide.
                # But some scores might be 0.5 raw. Context needed.
                # For Level 25, we assume inputs like 'feat_structure_score' are 0-100.
                if "structure_score" in feature:
                    val = val / 100.0 if val > 1.0 else val
                elif "rsi" in feature:
                    val = val / 100.0 if val > 1.0 else val
                elif "fuzzy" in feature:
                    # Fuzzy is -10 to +10 usually
                    val = val / 10.0 # Map -1.0 to 1.0 roughly
                    
            # 3. Acceleration (Range 0-10 or 0-1)
            elif "accel" in feature:
                if "accel_score" in feature:
                    val = val / 10.0 if val > 1.0 else val
            
            vector.append(float(val))
            
        # [SAFETY NET] Remove NaNs/Infs that crash CUDA
        vec_np = np.array(vector, dtype=np.float32)
        vec_np = np.nan_to_num(vec_np, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return vec_np

feat_processor = FeatProcessor()
