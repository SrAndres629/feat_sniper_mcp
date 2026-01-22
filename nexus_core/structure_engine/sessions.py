"""
[MODULE 05 - LEGACY REFACTOR]
Session identification using config-driven parameters.
Eliminates hardcoded weights.
"""
import pandas as pd
from datetime import time
from app.core.config import settings


def identify_trading_session(dt: pd.Timestamp) -> str:
    """
    Identifies the active trading session based on UTC time.
    Uses config-driven Killzone definitions.
    """
    if not hasattr(dt, "time"):
        return "NY_OPEN"  # Default for synthetic data

    current_time = dt.time()
    h = dt.hour if hasattr(dt, 'hour') else current_time.hour
    
    # Use config-driven ranges
    london_start, london_end = settings.TEMPORAL_LONDON_OPEN
    ny_start, ny_end = settings.TEMPORAL_NY_OPEN
    lc_start, lc_end = settings.TEMPORAL_LONDON_CLOSE
    
    # NY/London Overlap
    if lc_start <= h < lc_end:
        return "LONDON_CLOSE"
    
    # NY Open
    if ny_start <= h < ny_end:
        return "NY_OPEN"
    
    # London
    if london_start <= h < london_end:
        return "LONDON"
    
    # NY Late
    if ny_end <= h < 21:
        return "NY_LATE"
    
    # Asia (night session)
    asia_start, asia_end = settings.TEMPORAL_ASIA
    if h >= asia_start or h < asia_end:
        return "ASIA"
    
    return "NONE"


def get_session_weight(session: str) -> float:
    """
    Returns session weight from config.
    No hardcoded values.
    """
    return settings.TEMPORAL_SESSION_WEIGHTS.get(session, 0.1)
