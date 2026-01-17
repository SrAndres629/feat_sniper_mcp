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

        # 3. Standard Indicators (Speed Optimized)
        # [SENSOR FUSION CHANNEL A: MULTIFRACTAL KINETIC LAYER]
        # Calculate Cloud States (Micro, Structure, Macro, Bias)
        from nexus_core.kinetic_engine import kinetic_engine
        
        # Optimized: Pass strictly what's needed or the full DF. 
        # KineticEngine handles full DF well.
        # This replaces the manual EMA calc 8, 21, 50, 200
        kinetic_state = kinetic_engine.compute_kinetic_state(df)
        
        # [LEVEL 49] COGNITIVE PATTERN RECOGNITION
        kinetic_patterns = kinetic_engine.detect_kinetic_patterns(kinetic_state)
        
        # Broadcast Cloud Metrics & Patterns to DF Columns
        for k, v in kinetic_state.items():
            df[k] = v
        for k, v in kinetic_patterns.items():
            df[k] = v
        
        # ATR (Average True Range) for Normalization
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())

        low_close = np.abs(df["low"] - df["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df["atr14"] = true_range.rolling(14).mean()
        
        # [Safety] Fill NaN ATR with mean or small epsilon to prevent division by zero
        df["atr14"] = df["atr14"].fillna(df["atr14"].mean()).fillna(0.0001)

        # [LEVEL 35] ZERO-LAG TRACKING (Kalman Filter)
        from nexus_core.math_engine import calculate_kalman_filter
        
        # Calculate Kalman Estimate (Fast Numba)
        close_prices = df["close"].to_numpy()
        kalman_est = calculate_kalman_filter(close_prices)
        df["kalman_price"] = kalman_est
        
        # Kalman Deviation (Anomalies in acceleration)
        # High deviation = Price moving faster than physics model = Impulsive Move
        df["kalman_deviation"] = (df["close"] - df["kalman_price"]) / (df["kalman_price"] + 1e-9)
        df["kalman_score"] = np.abs(df["kalman_deviation"]) * 1000.0 # Scaling for ML

        # [LEVEL 41 & 45] INSTITUTIONAL PVP-FEAT METRICS
        # Computes Entropy, Skew, Kurtosis from Volume Profile
        # We leverage the powerful FEATFeatures engine to get accurate PVP data
        from nexus_core.features import feat_features
        
        # We need to compute PVP metrics for the window if possible, or per row.
        # For efficiency in backtesting, we compute simplified PVP on the whole DF if small,
        # or rolling if large. Here we use the scalar extractor which uses the whole DF window passed.
        
        # Note: In a real streaming scenario, this would be incremental.
        # Here we approximate by calculating the Profile of the current dataframe window.
        pvp_stats = feat_features._compute_pvp_metrics(df)
        
        # Broadcast these global/window stats to columns (Contextual embedding)
        df["poc_price"] = pvp_stats.get("poc_price", df["close"].mean())
        df["vah"] = pvp_stats.get("vah", df["high"].max())
        df["val"] = pvp_stats.get("val", df["low"].min())
        df["energy_score"] = pvp_stats.get("total_volume", 0.0)
        
        # Density Zone: Volume of current bar relative to Mean Profile Volume
        # Estimated as: Current Vol / (Total Vol / Bins)
        avg_vol_per_level = pvp_stats.get("total_volume", 1.0) / 50.0 # 50 bins
        df["density_zone"] = df["volume"] / (avg_vol_per_level + 1e-9)

        # 1. Energy Z-Score (Acceleracion)
        # Is current volume statistically significant?
        vol_mean = df["volume"].rolling(window=20).mean()
        vol_std = df["volume"].rolling(window=20).std()
        df["energy_z_score"] = (df["volume"] - vol_mean) / (vol_std + 1e-9)
        
        # 2. POC Velocity (Aceleracion)
        # Using VWAP as POC proxy for speed
        df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()
        df["poc_velocity"] = df["vwap"].diff() / df["atr14"]
        
        # 3. Form Metrics (Entropy/Skew) require full profile construction. 
        # We will assume these are computed by a dedicated loop or simplified here.
        # Simplification: Price Range Entropy
        # Rolling Log Returns Entropy
        returns = np.log(df["close"] / df["close"].shift(1))
        # Rolling standard deviation as proxy for entropy of distribution width
        df["entropy_proxy"] = returns.rolling(20).std() * 100
        
        return df

    def compute_latent_vector(self, row: pd.Series) -> Dict[str, float]:
        """
        [LEVEL 47] SENSOR FUSION PROTOCOL
        Extracts Kinetic (Channel A), Structural (Channel B), and Temporal (Channel C) features.
        """
        atr = row.get("atr14", 1.0) + 1e-9
        close = row["close"]
        
        return {
            # --- CHANNEL A: KINETIC LAYERS (Multifractal) ---
            # 1. Micro Layer (Intent)
            "micro_compression": row.get("micro_compression", 1.0),
            "dist_micro": row.get("dist_micro", 0.0),
            
            # 2. Structure Layer (Reality)
            "struct_compression": row.get("structure_compression", 1.0),
            "dist_struct": row.get("dist_structure", 0.0),
            
            # 3. Macro Layer (Memory)
            "macro_compression": row.get("macro_compression", 1.0),
            "dist_macro": row.get("dist_macro", 0.0),
            
            # 4. Bias (Absolute)
            "dist_bias": row.get("dist_bias", 0.0),
            
            # 5. Relationships
            "delta_micro_struct": row.get("delta_micro_struct", 0.0),
            "delta_struct_macro": row.get("delta_struct_macro", 0.0),
            "layer_alignment": row.get("layer_alignment", 0.0),
            
            # 6. Cognitive Patterns (Level 49)
            "kinetic_pattern_id": float(row.get("pattern_id", 0)),
            "kinetic_coherence": row.get("kinetic_coherence", 0.0),

            # --- CHANNEL B: STRUCTURAL (Terrain / PVP) ---
            "dist_poc_norm": (close - row.get("poc_price", close)) / atr,
            "pos_in_va": 1.0 if (row.get("val", -1) <= close <= row.get("vah", -1)) else 0.0,
            "density_zone": row.get("density_zone", 0.0),
            "energy_score": row.get("energy_score", 0.0),
            "skew": 0.0, # Placeholder
            
            # --- CHANNEL C: TEMPORAL / SPACE ---
            "cycle_prog": (row.name.minute / 60.0) if hasattr(row.name, 'minute') else 0.5,
            
            # Absolute Levels (Context for Decoder)
            "poc_price": row.get("poc_price", 0.0),
            "vah_price": row.get("vah", 0.0),
            "val_price": row.get("val", 0.0),
            "centroid_micro": row.get("centroid_micro", 0.0),
            "centroid_struct": row.get("centroid_structure", 0.0), 
            "bias_val": row.get("bias_val", 0.0)
        }
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
