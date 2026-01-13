"""
FEAT NEXUS: FEATURE ENGINEERING (Energy Map)
============================================
Converts raw market physics into institutional intention maps.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from nexus_core.math_engine import fast_bin_indices, bin_volume_fast, calculate_weighted_kde
from nexus_core.structure_engine import structure_engine
from nexus_core.acceleration import acceleration_engine

class FEATFeatures:
    def __init__(self):
        self.version = "2.5.0"

    def generate_energy_map(self, df: pd.DataFrame, bins_n: int = 50, tick_cvd: dict = None) -> Dict[str, Any]:
        """
        Synthesizes the FEAT Energy Map: Composite of Density, Volatility, and Flow.
        Returns tensors ready for CNN/LSTM consumption.
        
        Args:
            df: DataFrame with OHLCV data
            bins_n: Number of bins for price discretization
            tick_cvd: Optional dict from compute_real_cvd() for real CVD flow field
        """
        if df.empty or len(df) < 20:
            return {"energy_tensor": np.array([]), "hotspots": []}

        prices = df['close'].to_numpy()
        volumes = df['volume'].to_numpy()
        
        # 1. Density Field (PVP FEAT)
        bin_size = (np.max(prices) - np.min(prices)) / bins_n
        centers, vol_profile = bin_volume_fast(prices, volumes, bin_size)
        density_field = vol_profile / (vol_profile.sum() + 1e-9)

        # 2. Kinetic Field (Volatility)
        # Higher range = higher kinetic energy
        highs = df['high'].to_numpy()
        lows = df['low'].to_numpy()
        candle_range = highs - lows
        # Project range back to bins
        kinetic_field = np.zeros(len(centers))
        indices = fast_bin_indices(prices, bin_size, np.min(prices))
        for i in range(len(indices)):
            if indices[i] < len(kinetic_field):
                kinetic_field[indices[i]] += candle_range[i]
        
        kinetic_field /= (kinetic_field.sum() + 1e-9)

        # 3. Flow Field (CVD Intensity)
        # Use REAL CVD if tick data available, otherwise tick-rule approximation
        if tick_cvd and 'cvd_series' in tick_cvd and len(tick_cvd['cvd_series']) > 0:
            # Real CVD from MT5 tick flags - resample to match bins
            cvd_series = np.array(tick_cvd['cvd_series'])
            # Normalize and project to price bins (simplified: use last N values)
            cvd_resampled = np.interp(
                np.linspace(0, len(cvd_series)-1, len(centers)),
                np.arange(len(cvd_series)),
                cvd_series
            )
            flow_field = cvd_resampled
        else:
            # Fallback: Tick-rule approximation for the heat map
            deltas = np.diff(prices, prepend=prices[0])
            flow_delta = deltas * volumes
            flow_field = np.zeros(len(centers))
            for i in range(len(indices)):
                if indices[i] < len(flow_field):
                    flow_field[indices[i]] += flow_delta[i]

        # 4. Composite Energy Tensor
        # E = Density * Kinetic * tanh(Flow)
        energy_tensor = density_field * kinetic_field * np.tanh(flow_field / (np.std(flow_field) + 1e-9))
        
        # Normalize with Z-Score
        mean_e = np.mean(energy_tensor)
        std_e = np.std(energy_tensor) + 1e-9
        energy_tensor_norm = (energy_tensor - mean_e) / std_e

        return {
            "energy_tensor": energy_tensor_norm.tolist(),
            "poc_idx": int(np.argmax(density_field)),
            "max_energy_idx": int(np.argmax(np.abs(energy_tensor_norm))),
            "cvd_source": "real_mt5_ticks" if tick_cvd else "tick_rule_approximation",
            "metadata": {
                "bins": bins_n,
                "bin_size": bin_size,
                "range_min": float(np.min(prices)),
                "range_max": float(np.max(prices))
            }
        }

    def extract_scalar_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Legacy scalar features for LightGBM baseline.
        """
        # Integration with market_physics is expected here
        from app.skills.market_physics import market_physics
        pvp = market_physics.calculate_pvp_feat(df)
        cvd = market_physics.calculate_cvd_metrics(df)
        energy = market_physics.calculate_energy_map(df)
        
        # Structure Engine (FourJarvis Protocol)
        struct_results = structure_engine.compute_feat_index(df).iloc[-1].to_dict()
        
        # Acceleration Engine (A)
        accel_results = acceleration_engine.compute_acceleration_features(df).iloc[-1].to_dict()
        
        return {
            "z_score_poc": pvp.get("z_score", 0),
            "cvd_imbalance": cvd.get("imbalance_ratio", 0),
            "energy_score": energy.get("energy_score", 0),
            "absorption_tension": energy.get("absorption_tension", 0),
            "feat_form_score": struct_results.get("feat_form", 0),
            "feat_space_score": struct_results.get("feat_space", 0),
            "feat_acceleration_score": struct_results.get("feat_acceleration", 0),
            "feat_time_score": struct_results.get("feat_time", 0),
            "feat_index": struct_results.get("feat_index", 0),
            "accel_trigger": accel_results.get("accel_flag", 0),
            "accel_score": accel_results.get("accel_score", 0),
            "accel_type": accel_results.get("accel_type", "normal")
        }

    def apply_feat_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrates Structure and Acceleration engines to produce the full FEAT feature set.
        Returns the DataFrame enriched with all institutional metrics.
        """
        if df.empty: return df
        
        # 1. Structure Engine
        df = structure_engine.detect_structure(df)
        df = structure_engine.detect_zones(df)
        
        # 2. Acceleration Engine
        # Returns a dataframe with columns like 'accel_score', 'disp_norm', etc.
        accel_df = acceleration_engine.compute_acceleration_features(df)
        
        # Join acceleration features
        # Ensure we don't duplicate columns if they already exist
        cols_to_use = accel_df.columns.difference(df.columns)
        df = df.join(accel_df[cols_to_use])
        
        # 3. Compute Final Index
        # This adds feat_form, feat_space, feat_acceleration, feat_time, feat_index
        index_res = structure_engine.compute_feat_index(df)
        cols_to_use_idx = index_res.columns.difference(df.columns)
        df = df.join(index_res[cols_to_use_idx])
        
        return df

feat_features = FEATFeatures()
