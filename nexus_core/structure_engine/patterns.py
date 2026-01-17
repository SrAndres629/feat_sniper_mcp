import logging
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger("feat.structure.patterns")

class MAE_Pattern_Recognizer:
    """
    Gate F: Pattern Recognition Cortex.
    Implements the MAE Axiom: Momentum -> Accumulation -> Expansion.
    """
    def __init__(self):
        logger.debug("[Form] Pattern Engine Online")

    def detect_mae_pattern(self, df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < 15:
            return {"phase": "WARMUP", "status": "RANGING"}

        atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
        body = df["close"].iloc[-1] - df["open"].iloc[-1]
        is_momentum = abs(body) > (atr * 1.5)

        recent_range = df["high"].iloc[-5:-1].max() - df["low"].iloc[-5:-1].min()
        is_accumulation = recent_range < (atr * 1.2)

        upper_bound = df["high"].iloc[-5:-1].max()
        lower_bound = df["low"].iloc[-5:-1].min()

        is_expansion_up = df["close"].iloc[-1] > upper_bound and body > 0
        is_expansion_down = df["close"].iloc[-1] < lower_bound and body < 0

        status = "RANGING"
        phase = "NORMAL"

        if is_momentum:
            phase = "MOMENTUM"
            status = "IMPULSE"
        elif is_accumulation:
            phase = "ACCUMULATION"
            status = "COMPRESSION"
        elif is_expansion_up or is_expansion_down:
            phase = "EXPANSION"
            status = "BREAKOUT"

        return {
            "phase": phase,
            "status": status,
            "is_expansion": is_expansion_up or is_expansion_down,
            "direction": 1 if is_expansion_up else (-1 if is_expansion_down else 0),
        }
