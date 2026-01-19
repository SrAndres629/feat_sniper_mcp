import pandas as pd
import numpy as np

class AbsorptionValidator:
    """
    [DOCTORAL SKILL]
    Validates the 'Strength Test' of an Impulse Candle.
    Rule: A true institutional move is not immediately reversed > 50% by subsequent candles.
    """
    
    def validate_impulse_persistence(self, df: pd.DataFrame, idx: int, lookahead: int = 3) -> dict:
        """
        Checks if the impulse at 'idx' held its ground over the next 'lookahead' candles.
        Returns:
            - status: CONTINUATION, FAILURE, or NEUTRAL
            - retained_pct: Percentage of impulse body retained
        """
        if idx + lookahead >= len(df):
            return {"status": "PENDING", "retained_pct": 1.0}
            
        impulse = df.iloc[idx]
        body_size = abs(impulse["close"] - impulse["open"])
        if body_size == 0: return {"status": "NEUTRAL", "retained_pct": 0.0}
        
        is_bull = impulse["close"] > impulse["open"]
        impulse_mid = impulse["low"] + (body_size * 0.5)
        impulse_open = impulse["open"]
        
        # Check subsequent candles
        failures = 0
        min_retained = 1.0
        
        for i in range(1, lookahead + 1):
            future = df.iloc[idx + i]
            
            if is_bull:
                # Failure: Closing below 50% of impulse body
                if future["close"] < impulse_mid:
                    return {"status": "FAILURE", "retained_pct": 0.0}
            else:
                # Bearish Impulse
                impulse_mid = impulse["high"] - (body_size * 0.5)
                if future["close"] > impulse_mid:
                    return {"status": "FAILURE", "retained_pct": 0.0}
                    
        return {"status": "CONTINUATION", "retained_pct": 1.0}

    def check_absorption_state(self, df: pd.DataFrame) -> float:
        """
        Rolling check: Are we currently 'inside' a valid impulse structure?
        Returns: 1.0 (Safe/Continued), 0.0 (Neutral), -1.0 (Failed/Reversed)
        """
        # Optimized for last 5 candles
        # Find last "Big Candle" (Body > 1.5x ATR)
        # Check if we are still holding its 50% level
        
        # Simplified for now: just return a placeholder or scan last 3
        # Ideally this requires tracking 'active_impulse_idx' state
        return 0.0
