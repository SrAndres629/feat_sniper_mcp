import logging
import pandas as pd
from typing import Dict, Any
from .vectors import MomentumVector
from .features import compute_acceleration_features

logger = logging.getLogger("feat.acceleration")

class AccelerationEngine:
    def __init__(self, config: Dict[str, Any] = None):
        print("[Physics] Sigma Monitor ON (Velocity/Acceleration vectors)")
        self.config = config or {
            "atr_w": 14,
            "vol_w": 20,
            "score_th": 0.70,
            "sigma_th": 3.0, # Alert threshold for sigma
            "accel_th": 1.5, # Newtonian threshold
            "weights": {
                "w1": 0.4, # Displacement
                "w2": 0.3, # Volume Z-Score
                "w3": 0.2, # FVG Presence
                "w4": 0.1  # Velocity
            }
        }
        self.momentum_v = MomentumVector(threshold=self.config["accel_th"])

    def calculate_momentum_vector(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gate A: Velocity/Acceleration Vector.
        Calculates Newtonian physics of price movement.
        """
        return self.momentum_v.analyze(df)

    def compute_acceleration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main API for context-aware acceleration analysis.
        Incorporates FEAT acceleration logic: Momentum, RVOL, CVD Proxy.
        """
        return compute_acceleration_features(df, self.config)

acceleration_engine = AccelerationEngine()
