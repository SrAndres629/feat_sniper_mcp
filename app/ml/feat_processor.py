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


feat_processor = FeatProcessor()
