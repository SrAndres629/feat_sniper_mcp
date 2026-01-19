import logging
import pandas as pd
from typing import Dict, Any
from .vectors import MomentumVector
from .features import compute_acceleration_features
from app.core.config import settings

logger = logging.getLogger("feat.acceleration")

class AccelerationEngine:
    def __init__(self, config: Dict[str, Any] = None):
        print("[Physics] Sigma Monitor ON (Velocity/Acceleration vectors)")
        self.config = config or {
            "atr_w": settings.ACCEL_ATR_WINDOW,
            "vol_w": settings.ACCEL_VOL_WINDOW,
            "score_th": settings.ACCEL_SCORE_THRESHOLD,
            "sigma_th": settings.ACCEL_SIGMA_THRESHOLD,
            "accel_th": settings.ACCEL_NEWTON_THRESHOLD,
            "weights": settings.ACCEL_WEIGHTS
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
