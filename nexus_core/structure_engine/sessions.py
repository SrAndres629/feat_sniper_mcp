import pandas as pd
from datetime import time

def identify_trading_session(dt: pd.Timestamp) -> str:
    """
    Identifies the active trading session based on New York Time (EST/EDT).
    Simplified for structural weighting.
    """
    # NYC Time
    current_time = dt.time()
    
    # London: 03:00 - 12:00 (Overlap with NY)
    # NY: 08:00 - 17:00 
    # Asia: 19:00 - 03:00
    
    if current_time >= time(8, 0) and current_time <= time(12, 0):
        return "NY_OPEN" # High Importance
    elif current_time > time(12, 0) and current_time <= time(17, 0):
        return "NY_LATE"
    elif current_time >= time(3, 0) and current_time < time(8, 0):
        return "LONDON"
    elif current_time >= time(19, 0) or current_time < time(3, 0):
        return "ASIA"
    
    return "NONE"

def get_session_weight(session: str) -> float:
    weights = {
        "NY_OPEN": 1.0,
        "LONDON": 0.8,
        "NY_LATE": 0.5,
        "ASIA": 0.3,
        "NONE": 0.1
    }
    return weights.get(session, 0.1)
