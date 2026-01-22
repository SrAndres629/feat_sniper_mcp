"""
[MODULE 05 - DOCTORAL CHRONOS]
Temporal Encoding Functions.
Transforms linear time into harmonic circular embeddings.
"""
import numpy as np
import pandas as pd
from typing import Tuple
from app.core.config import settings


def sinusoidal_encode(value: float, period: float) -> Tuple[float, float]:
    """
    [HARMONIC ENCODING]
    Maps a linear value to a 2D circular embedding.
    
    Properties:
    - Eliminates discontinuity (23:59 → 00:00 is smooth)
    - Preserves cyclical distance relationships
    
    Returns: (sin, cos) tuple
    """
    radians = 2 * np.pi * (value / period)
    return np.sin(radians), np.cos(radians)


def encode_hour(hour: float) -> Tuple[float, float]:
    """24-hour cycle encoding."""
    return sinusoidal_encode(hour, 24.0)


def encode_minute(minute: float) -> Tuple[float, float]:
    """60-minute cycle encoding."""
    return sinusoidal_encode(minute, 60.0)


def encode_day_of_week(dow: int) -> Tuple[float, float]:
    """
    [WEEKLY CYCLE ENCODING]
    Encodes day of week as sinusoidal for 5-day trading week.
    
    Trading week mapping (0-4 for sin/cos calculation):
    - Monday = 0
    - Tuesday = 1
    - Wednesday = 2
    - Thursday = 3
    - Friday = 4
    - Saturday/Sunday = Treated as Friday (no trading, weekend gap)
    
    This ensures Friday→Monday continuity in circular space.
    """
    # Map to 5-day trading week (0-4)
    if dow >= 5:  # Weekend
        trading_dow = 4  # Treat as Friday
    else:
        trading_dow = dow
    
    # Sinusoidal encoding for 5-day cycle
    return sinusoidal_encode(trading_dow, 5.0)


def encode_weekly_phase(dow: int, hour: float) -> float:
    """
    [WEEKLY PHASE]
    Returns position in the weekly cycle (0.0 to 1.0).
    
    FEAT Weekly Cycle:
    - Monday early: 0.0-0.2 (Weekly Accumulation)
    - Tuesday-Wednesday: 0.2-0.6 (Weekly Manipulation/Expansion)
    - Thursday: 0.6-0.8 (Weekly Expansion Peak)
    - Friday: 0.8-1.0 (Weekly Distribution)
    """
    if dow >= 5:  # Weekend
        return 1.0  # End of cycle
    
    # Calculate position: (day * 24 + hour) / (5 days * 24 hours)
    total_week_hours = 5 * 24
    current_position = (dow * 24) + hour
    
    return min(current_position / total_week_hours, 1.0)


def get_weekly_feat_phase(dow: int) -> str:
    """
    Returns the FEAT phase name for the current day of week.
    """
    phases = {
        0: "ACCUMULATION",    # Monday - weekly range forms
        1: "MANIPULATION",    # Tuesday - false breakouts
        2: "EXPANSION",       # Wednesday - directional move
        3: "EXPANSION",       # Thursday - continuation
        4: "DISTRIBUTION",    # Friday - profit taking
        5: "CLOSED",          # Weekend
        6: "CLOSED"           # Weekend
    }
    return phases.get(dow, "UNKNOWN")


def encode_fractal_block(hour: int) -> Tuple[float, float]:
    """
    4-hour IPDA block encoding.
    Blocks: 00-04, 04-08, 08-12, 12-16, 16-20, 20-24
    """
    block_size = settings.TEMPORAL_FRACTAL_BLOCK_SIZE
    position_in_block = hour % block_size
    return sinusoidal_encode(position_in_block, float(block_size))


# =============================================
# MULTI-TIMEFRAME FRACTAL POSITION ENCODING
# =============================================
# For each timeframe, calculates:
# 1. candle_position: Which sub-candle are we in (1, 2, 3, or 4)
# 2. candle_phase: Position within the candle (0.0 to 1.0)

def get_mtf_positions(hour: int, minute: int) -> dict:
    """
    [MULTI-TIMEFRAME FRACTAL ENCODING]
    Returns position within each timeframe's candle.
    
    Allows the neural network to DISCOVER patterns like:
    - "H4-3 = expansion" (without hardcoding it)
    - "M15-3 = confirmation window"
    
    Returns dict with:
    - {tf}_position: Which sub-candle (1-4)
    - {tf}_phase: Position 0.0-1.0 within the candle
    """
    total_minutes = hour * 60 + minute
    
    result = {}
    
    # M1 position within M5 (which M1 are we in the current M5?)
    # M5 has 5 M1 candles, but we use 4-phase model
    m1_in_m5 = (minute % 5) + 1  # 1-5
    m1_phase = (minute % 5) / 5.0
    result["m1_position"] = min(m1_in_m5, 4)  # Cap at 4 for FEAT model
    result["m1_phase"] = m1_phase
    
    # M5 position within H1 (which M5 are we in the current hour?)
    # H1 has 12 M5 candles, mapped to 4 phases
    m5_in_h1 = (minute // 5) + 1  # 1-12
    m5_phase = (minute / 60.0)
    # Map 1-12 to 1-4 (3 candles per phase)
    result["m5_position"] = min((m5_in_h1 - 1) // 3 + 1, 4)
    result["m5_phase"] = m5_phase
    
    # M15 position within H1 (which M15 are we in the current hour?)
    # H1 has 4 M15 candles - perfect for FEAT model
    m15_in_h1 = (minute // 15) + 1  # 1-4
    m15_phase = (minute / 60.0)
    result["m15_position"] = m15_in_h1
    result["m15_phase"] = m15_phase
    
    # M30 position within H4 (which M30 are we in the current H4 block?)
    # H4 has 8 M30 candles, mapped to 4 phases
    h4_block_start = (hour // 4) * 4  # 0, 4, 8, 12, 16, 20
    minutes_into_h4 = (hour - h4_block_start) * 60 + minute
    m30_in_h4 = (minutes_into_h4 // 30) + 1  # 1-8
    m30_phase = minutes_into_h4 / 240.0  # 240 min = 4h
    result["m30_position"] = min((m30_in_h4 - 1) // 2 + 1, 4)
    result["m30_phase"] = m30_phase
    
    # H1 position within H4 (which H1 are we in the current H4 block?)
    # H4 has 4 H1 candles - perfect for FEAT model
    h1_in_h4 = (hour % 4) + 1  # 1-4
    h1_phase = ((hour % 4) * 60 + minute) / 240.0
    result["h1_position"] = h1_in_h4
    result["h1_phase"] = h1_phase
    
    # H4 position within D1 (which H4 are we in the current day?)
    # D1 has 6 H4 candles, mapped to 4 phases
    h4_in_d1 = (hour // 4) + 1  # 1-6
    h4_phase = (hour * 60 + minute) / 1440.0  # 1440 min = 24h
    # Map 1-6 to 1-4 (approximate)
    h4_position_map = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4}
    result["h4_position"] = h4_position_map.get(h4_in_d1, 2)
    result["h4_phase"] = h4_phase
    
    # D1 position within W1 (which day are we in the week?) - already have dow
    # Use day of week directly (0-4 for trading week)
    # This is handled by weekly_phase, but we add d1_position for consistency
    
    # W1 position within Month (approximately which week?)
    # Simplified: assume 4 weeks per month
    
    return result


def get_mtf_feat_names(positions: dict) -> dict:
    """
    Maps position numbers to FEAT phase names.
    """
    feat_map = {
        1: "ACCUMULATION",
        2: "MANIPULATION",
        3: "EXPANSION",
        4: "DISTRIBUTION"
    }
    
    result = {}
    for tf in ["m1", "m5", "m15", "m30", "h1", "h4"]:
        pos_key = f"{tf}_position"
        if pos_key in positions:
            result[f"{tf}_feat"] = feat_map.get(positions[pos_key], "UNKNOWN")
    
    return result

def killzone_intensity(utc_hour: float, utc_minute: float = 0) -> float:
    """
    [INSTITUTIONAL PULSE]
    Calculates Killzone Intensity using Gaussian distribution.
    
    Returns: 0.0 to 1.0 (peak = 1.0 at session open, decay = Gaussian)
    
    Logic:
    - NY Open peak: 14:00 UTC (9:00 AM EST)
    - London Open peak: 08:00 UTC
    - Asia: Low intensity baseline
    """
    # Convert to decimal hour
    h = utc_hour + utc_minute / 60.0
    
    # Killzone peaks (UTC)
    ny_peak = 14.0       # 9:00 AM EST = 14:00 UTC
    london_peak = 8.0    # London Open
    overlap_peak = 15.5  # NY/London overlap
    
    # Gaussian spread (hours)
    sigma = settings.TEMPORAL_KZ_PEAK_SPREAD
    
    def gaussian(x, mu, s):
        return np.exp(-0.5 * ((x - mu) / s) ** 2)
    
    # Calculate intensity for each session
    ny_intensity = gaussian(h, ny_peak, sigma)
    london_intensity = gaussian(h, london_peak, sigma)
    overlap_intensity = gaussian(h, overlap_peak, sigma * 0.8)  # Tighter peak
    
    # Combine (max of all sessions)
    combined = max(ny_intensity, london_intensity, overlap_intensity)
    
    # Apply floor for dead zones
    asia_start, asia_end = settings.TEMPORAL_ASIA
    if asia_start <= h or h < asia_end:
        # In Asia session, cap intensity
        combined = min(combined, 0.3)
    
    return float(np.clip(combined, 0.05, 1.0))


def get_session_name(utc_hour: int) -> str:
    """
    Returns the current session name based on UTC hour.
    Uses config-driven Killzone definitions.
    """
    london_start, london_end = settings.TEMPORAL_LONDON_OPEN
    ny_start, ny_end = settings.TEMPORAL_NY_OPEN
    lc_start, lc_end = settings.TEMPORAL_LONDON_CLOSE
    asia_start, asia_end = settings.TEMPORAL_ASIA
    
    if ny_start <= utc_hour < ny_end:
        if lc_start <= utc_hour < lc_end:
            return "LONDON_CLOSE"  # Overlap
        return "NY_OPEN"
    elif london_start <= utc_hour < london_end:
        return "LONDON"
    elif asia_start <= utc_hour or utc_hour < asia_end:
        return "ASIA"
    elif ny_end <= utc_hour < 21:
        return "NY_LATE"
    
    return "NONE"


def get_session_weight(utc_hour: int) -> float:
    """Returns session weight from config."""
    session = get_session_name(utc_hour)
    return settings.TEMPORAL_SESSION_WEIGHTS.get(session, 0.1)
