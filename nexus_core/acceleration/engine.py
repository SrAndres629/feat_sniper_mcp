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
        [UPDATED] Adds Contextual Kinetics (Breakout vs Exhaustion).
        """
        accel_df = compute_acceleration_features(df, self.config)
        
        # [CONTEXTUAL KINETICS]
        # We need to know if we are interacting with a zone.
        # We check the INPUT df for 'confluence_score'
        
        # Initialize context column
        accel_df["kinetic_context"] = 0 
        
        if "confluence_score" in df.columns:
            # Preserve score in output for debugging/downstream
            accel_df["confluence_score"] = df["confluence_score"]
            
            # Touching a zone (high score)
            # Use .values to ignore index alignment issues if any, assuming same index
            in_zone = df["confluence_score"] > 2.0 
            
            # High Accel (using calculation result)
            is_fast = accel_df["accel_score"] > self.config["score_th"]
            
            # Exhaustion: Fast INTO a zone
            mask_exhaustion = is_fast & in_zone
            accel_df.loc[mask_exhaustion, "kinetic_context"] = -1 # Exhaustion / Warning
            
            # Breakout: Extreme Acceleration
            mask_breakout = (accel_df["accel_score"] > self.config["score_th"] * 2.0)
            
            # Refinement: If Breakout Force is high, we override Exhaustion
            # But normally Breakout requires CLEAN space. If Confluence is high, it's a fight.
            # We stick to User Rule: Accelerating INTO Zone = Exhaustion Risk unless Force is massive.
            # Let's say Extreme Force > 2.0 overrides.
            
            accel_df.loc[mask_breakout, "kinetic_context"] = 1 # Breakout
            
        return accel_df

acceleration_engine = AccelerationEngine()
