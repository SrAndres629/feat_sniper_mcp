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

        # 3. Flow Field (OFI - Order Flow Imbalance)
        # Delta = Buy Vol - Sell Vol
        flow_field = np.zeros(len(centers))
        ofi_signal = 0.0 # Normalized -1.0 to 1.0
        
        if tick_cvd and 'cvd_series' in tick_cvd and len(tick_cvd['cvd_series']) > 0:
            # Real CVD from MT5 tick flags
            cvd_series = np.array(tick_cvd['cvd_series'])
            cvd_resampled = np.interp(
                np.linspace(0, len(cvd_series)-1, len(centers)),
                np.arange(len(cvd_series)),
                cvd_series
            )
            flow_field = cvd_resampled
            # Global OFI Signal from real delta
            net_delta = tick_cvd.get("delta", 0.0)
            total_vol = tick_cvd.get("volume", 1.0)
            ofi_signal = np.clip(net_delta / (total_vol + 1e-9), -1.0, 1.0)
        else:
            # [APPROXIMATION] "Within-Candle" Delta
            # Delta ≈ Vol * (2*(Close-Low)/(High-Low) - 1)
            # This captures buying pressure better than Close > Close[1]
            h = df['high'].values
            l = df['low'].values
            c = df['close'].values
            v = df['volume'].values
            
            # Avoid division by zero
            rng = h - l
            rng[rng == 0] = 1e-9
            
            # Buying Pressure Ratio (-1 to 1)
            pressure = (2 * (c - l) / rng) - 1.0
            deltas = v * pressure
            
            # Project to bins
            for i in range(len(indices)):
                if indices[i] < len(flow_field):
                    flow_field[indices[i]] += deltas[i]
            
            # Global OFI Signal (Sum of weighted deltas)
            ofi_signal = np.clip(np.sum(deltas) / (np.sum(v) + 1e-9), -1.0, 1.0)

        # 4. Composite Energy Tensor
        # E = Density * Kinetic * tanh(Flow)
        energy_tensor = density_field * kinetic_field * np.tanh(flow_field / (np.std(flow_field) + 1e-9))
        
        # Normalize with Z-Score
        mean_e = np.mean(energy_tensor)
        std_e = np.std(energy_tensor) + 1e-9
        energy_tensor_norm = (energy_tensor - mean_e) / std_e

        # [SCALAR METRICS FOR HYBRID MODEL]
        # Skew: Distribution vs Price (Absorption Tension)
        poc_idx = int(np.argmax(density_field))
        avg_idx = np.sum(np.arange(len(density_field)) * density_field) # Center of Mass
        skew = (avg_idx - poc_idx) / (len(density_field) * 0.5) # Normalized -1 to 1

        # Energy Score: Total system activation
        energy_score = float(np.mean(np.abs(energy_tensor_norm)))

        return {
            "energy_tensor": energy_tensor_norm.tolist(),
            "poc_idx": poc_idx,
            "max_energy_idx": int(np.argmax(np.abs(energy_tensor_norm))),
            "skew": float(skew),
            "energy_score": energy_score,
            "ofi_signal": float(ofi_signal),
            "cvd_source": "real_mt5_ticks" if tick_cvd else "tick_rule_approximation",
            "metadata": {
                "bins": bins_n,
                "bin_size": bin_size,
                "range_min": float(np.min(prices)),
                "range_max": float(np.max(prices))
            }
        }

    def generate_2d_energy_map(self, df: pd.DataFrame, bins: int = 50) -> np.ndarray:
        """
        [LEVEL 55] NEURO-MATHEMATICAL SINGULARITY
        Generates a 50x50 Spatio-Temporal Energy Tensor.
        Y-Axis: Price Bins (Spatial)
        X-Axis: Time Steps (Temporal Evolution)
        """
        if len(df) < bins:
            return np.zeros((bins, bins))
            
        # Use last N bars
        df_target = df.iloc[-bins:]
        prices = df_target["close"].values
        volumes = df_target["volume"].values
        
        min_p = prices.min()
        max_p = prices.max()
        range_p = max_p - min_p if max_p > min_p else 1.0
        
        # Grid initialization
        grid = np.zeros((bins, bins))
        
        # Physics Mapping: 
        # Each candle projects its energy onto the 2D manifold
        for i in range(bins):
            # Price position in bins
            price_bin = int(((prices[i] - min_p) / range_p) * (bins - 1))
            price_bin = max(0, min(bins - 1, price_bin))
            
            # Volume Intensity (Energy)
            vol_intensity = volumes[i] / (df["volume"].rolling(20).mean().iloc[-bins+i] + 1e-9)
            
            # Spatial Decay (Kernel): Spread energy to nearby bins to simulate price influence
            # Simplified Gaussian spread
            grid[price_bin, i] = vol_intensity
            if price_bin > 0: grid[price_bin-1, i] = vol_intensity * 0.5
            if price_bin < bins-1: grid[price_bin+1, i] = vol_intensity * 0.5
                
        # Z-Score Normalization for Neural Consistency
        grid = (grid - np.mean(grid)) / (np.std(grid) + 1e-9)
        return grid

    def extract_scalar_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Legacy scalar features for LightGBM baseline.
        [FIX] Now computes PVP metrics internally using MathEngine.
        """
        # 1. Physics & Structure
        # ----------------------
        df = structure_engine.detect_structural_shifts(df)
        health = structure_engine.get_structural_health(df)
        accel_results = acceleration_engine.compute_acceleration_features(df).iloc[-1].to_dict()
        
        # 2. PVP / Volume Profile (Computed Locally)
        # ------------------------------------------
        pvp_metrics = self._compute_pvp_metrics(df)
        
        # 3. Energy Map Summary
        # ---------------------
        energy_map = self.generate_energy_map(df, bins_n=50)
        energy_tensor = np.array(energy_map.get("energy_tensor", []))
        energy_score = np.mean(np.abs(energy_tensor)) if len(energy_tensor) > 0 else 0.0
        
        return {
            "z_score_poc": pvp_metrics.get("z_score_poc", 0),
            "cvd_imbalance": energy_map.get("ofi_signal", 0.0), # Real OFI Flow
            "energy_score": energy_score,
            "absorption_tension": pvp_metrics.get("skew", 0),
            
            "feat_form_score": health.get("overall_form_score", 0),
            "feat_space_score": health.get("zone_confidence", 0),
            "feat_acceleration_score": health.get("mae_confidence", 0),
            "feat_time_score": health.get("layer_alignment", 0),
            "feat_index": health.get("health_score", 0),
            
            "accel_trigger": accel_results.get("accel_flag", 0),
            "accel_score": accel_results.get("accel_score", 0),
            "accel_type": accel_results.get("accel_type", "normal")
        }

    def _compute_pvp_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Helper: Computes simplified Volume Profile metrics for scalar features.
        """
        if len(df) < 20:
            return {"z_score_poc": 0.0, "skew": 0.0}
            
        prices = df['close'].to_numpy()
        volumes = df['volume'].to_numpy()
        
        # Use MathEngine for fast binning
        bin_size = (np.max(prices) - np.min(prices)) / 50
        if bin_size == 0: bin_size = 0.0001
            
        centers, vol_profile = bin_volume_fast(prices, volumes, bin_size)
        
        # Find POC (Point of Control)
        poc_idx = np.argmax(vol_profile)
        poc_price = centers[poc_idx]
        current_price = prices[-1]

        # Calculate Value Area (70%)
        from nexus_core.math_engine import calculate_value_area_fast
        total_vol = np.sum(vol_profile)
        vah, val = calculate_value_area_fast(centers, vol_profile, total_vol, 0.70)
        
        # Z-Score of Price vs POC Distribution
        # Use simple standard deviation of prices weighted by volume
        avg_price = np.average(prices, weights=volumes)
        variance = np.average((prices - avg_price)**2, weights=volumes)
        std_price = np.sqrt(variance)
        
        z_score_poc = (current_price - poc_price) / (std_price + 1e-9)
        
        # Skew (Absorption Tension): Is volume concentrated above or below mean?
        skew = (avg_price - poc_price) / (std_price + 1e-9)
        
        return {
            "z_score_poc": z_score_poc,
            "skew": skew,
            "poc_price": poc_price,
            "vah": vah,
            "val": val,
            "total_volume": total_vol
        }

    def apply_feat_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Orchestrates Structure and Acceleration engines to produce the full FEAT feature set.
        Returns the DataFrame enriched with all institutional metrics.
        """
        if df.empty: return df
        
        # 1. Structure Engine
        df = structure_engine.detect_structural_shifts(df)
        
        # 2. Acceleration Engine
        accel_df = acceleration_engine.compute_acceleration_features(df)
        cols_to_use = accel_df.columns.difference(df.columns)
        df = df.join(accel_df[cols_to_use])
        
        # 3. Compute Health metrics as features
        # We broadcast the health score to the dataframe
        health = structure_engine.get_structural_health(df)
        df["feat_index"] = health.get("health_score", 0.0)
        df["feat_form"] = health.get("overall_form_score", 0.0)
        
        return df

feat_features = FEATFeatures()
