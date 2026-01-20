"""
FEAT NEXUS: ADAPTATION ENGINE (The Meta-Controller)
==================================================
Dynamically adjusts math windows and spectral levels based on Volatility Regimes.
Ensures high sensitivity in fast markets and high noise-filtering in dead markets.
"""

import numpy as np
from typing import Dict, Any

class AdaptationEngine:
    """
    State-aware controller for dynamic hyperparameter scaling.
    Motto: 'Speed for Crashes, Depth for Calms.'
    """

    def __init__(self):
        # Baseline configurations
        self.base_wavelet_level = 2
        self.base_regress_period = 5
        self.base_resolution = 64

    def get_volatility_scalar(self, volatility_metric: float, p20: float, p90: float) -> float:
        """
        Maps raw volatility (e.g. ATR or Energy Burst) to a scalar [0.5, 2.0].
        Standardizes the regime into a 'Sensitivity Factor'.
        """
        if volatility_metric >= p90:
            # High Volatility Regime (Fast Market)
            # We want LOW periods (fast reaction) -> Scalar < 1.0
            return 0.6 
        elif volatility_metric <= p20:
            # Low Volatility Regime (Dead Market)
            # We want HIGH periods (deep filtering) -> Scalar > 1.0
            return 1.8
        else:
            # Neutral Regime
            return 1.0

    def calculate_adaptive_params(self, vol_scalar: float) -> Dict[str, Any]:
        """
        Returns a dictionary of scaled parameters for the Physics/Neural stacks.
        """
        # Dynamic Wavelet Level: 
        # Fast market (scalar < 1) -> Level 1 or 2 (Lower lag)
        # Slow market (scalar > 1) -> Level 3 (Higher lag, more smoothing)
        wavelet_level = 2
        if vol_scalar < 0.8: wavelet_level = 1
        if vol_scalar > 1.5: wavelet_level = 3

        # Dynamic Regression Period:
        # Scale the base period by the scalar
        regress_period = int(max(3, min(12, self.base_regress_period * vol_scalar)))

        # Dynamic Volume Resolution:
        # Slow markets need more price precision (bins) to find value
        resolution = int(self.base_resolution * (1.5 if vol_scalar > 1.2 else 1.0))

        return {
            "wavelet_level": wavelet_level,
            "regress_period": regress_period,
            "volume_resolution": resolution,
            "vol_scalar": vol_scalar
        }

    def get_retracement_limit(self, vol_scalar: float) -> float:
        """
        Dynamically calculates the retracement limit based on volatility.
        - High Vol (Fast/Crash): Exigimos un retroceso profundo (ej. 61.8% -> 0.618) por seguridad.
        - Low Vol (Trend fuerte): Aceptamos un retroceso corto (ej. 23.6% -> 0.236) para entrar rápido.
        """
        # Linear interpolation between 0.236 (at scalar 0.6) and 0.618 (at scalar 1.8)
        # Simplified: base 0.5 and nudge by scalar
        if vol_scalar < 0.8: return 0.236
        if vol_scalar > 1.5: return 0.618
        return 0.5

    def get_absorption_window(self, vol_scalar: float) -> int:
        """
        Dynamically calculates how many candles to monitor for absorption.
        """
        if vol_scalar < 0.8: return 2 # Market moves fast, confirm fast
        if vol_scalar > 1.5: return 5 # Market is slow, allow more time
        return 3

    def get_config_for_regime(self, current_vol: float, history_vol: np.ndarray) -> Dict[str, Any]:
        """
        End-to-end regime detection and parameter generation.
        """
        if len(history_vol) < 20:
            scalar = 1.0
        else:
            p20 = np.percentile(history_vol, 20)
            p90 = np.percentile(history_vol, 90)
            scalar = self.get_volatility_scalar(current_vol, p20, p90)
            
        params = self.calculate_adaptive_params(scalar)
        params["retracement_limit"] = self.get_retracement_limit(scalar)
        params["absorption_window"] = self.get_absorption_window(scalar)
        return params

# Singleton Export
adaptation_engine = AdaptationEngine()
