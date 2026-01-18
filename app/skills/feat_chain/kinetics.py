import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("feat.neuro.kinetics")

class KineticsChannel:
    """
    🧬 Neuro-Channel: Market Physics & Kinetics
    Processes velocity, acceleration, and momentum vectors for scalping.
    """
    def __init__(self):
        try:
            from nexus_core.acceleration import acceleration_engine
            self.accel_engine = acceleration_engine
        except ImportError:
            self.accel_engine = None

    def compute_kinetic_vectors(self, df: pd.DataFrame) -> Dict[str, float]:
        if self.accel_engine:
            vec = self.accel_engine.calculate_momentum_vector(df)
            return {
                "velocity": vec.get("velocity", 0.0),
                "acceleration": vec.get("acceleration", 0.0),
                "momentum_bias": 1.0 if vec.get("status") == "ACCELERATING_UP" else -1.0 if vec.get("status") == "ACCELERATING_DOWN" else 0.0,
                "is_initiative": 1.0 if vec.get("is_initiative", False) else 0.0
            }
        return {"velocity": 0.0, "acceleration": 0.0, "momentum_bias": 0.0, "is_initiative": 0.0}

    def get_relative_strength(self, df: pd.DataFrame, period: int = 14) -> float:
        """Normalized RSI or relative momentum for the neural network."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1] / 100.0) # Normalized to 0-1
