import datetime
import pytz
from enum import Enum
from dataclasses import dataclass

class IPDAGlobalState(Enum):
    ACCUMULATION = "ACCUMULATION"   # Range / Build
    MANIPULATION = "MANIPULATION"   # Inducement / False Move
    EXPANSION = "EXPANSION"         # True Move / Trend
    DISTRIBUTION = "DISTRIBUTION"   # Reversal / Exit

class KillZone(Enum):
    ASIA = "ASIA"
    LONDON_OPEN = "LONDON_OPEN"
    NY_OPEN = "NY_OPEN"
    LONDON_CLOSE = "LONDON_CLOSE"
    NONE = "NONE"

@dataclass
class ChronosState:
    timestamp_ny: datetime.datetime
    ipda_phase: IPDAGlobalState
    fractal_hour: int           # 1, 2, 3, or 4 (Position in 4H block)
    kill_zone: KillZone
    is_kill_zone_active: bool
    liquidity_quality_score: float # 0.0 to 1.0 (Time Prob)

class FractalTimeTracker:
    """
    [CHRONOS ENGINE: LAYER 1]
    Engineering-grade time tracking for Institutional Liquidity Cycles.
    """
    def __init__(self):
        self.ny_tz = pytz.timezone('America/New_York')

    def get_market_state(self, utc_time: datetime.datetime = None) -> ChronosState:
        """
        Calculates the full Temporal State of the market.
        Everything is derived from Wall Street Time (NY).
        """
        if utc_time is None:
            utc_time = datetime.datetime.now(datetime.timezone.utc)
        
        # 1. Convert to Wall St Time
        ny_time = utc_time.astimezone(self.ny_tz)
        
        # 2. Derive Sub-Components
        fractal_hour = self._calculate_fractal_phase(ny_time)
        ipda_phase = self._calculate_ipda_phase(ny_time, fractal_hour)
        kill_zone, is_active = self._check_kill_zone(ny_time)
        quality = self._calculate_quality_score(ipda_phase, is_active)

        return ChronosState(
            timestamp_ny=ny_time,
            ipda_phase=ipda_phase,
            fractal_hour=fractal_hour,
            kill_zone=kill_zone,
            is_kill_zone_active=is_active,
            liquidity_quality_score=quality
        )

    def _calculate_fractal_phase(self, t: datetime.datetime) -> int:
        """
        Returns position in the 4H Institutional Block (1, 2, 3, or 4).
        Standard Blocks start at: 00, 04, 08, 12, 16, 20 NY Time.
        """
        hour = t.hour
        # Modulo 4 gives 0, 1, 2, 3. We want 1-based index.
        # Example: 08:00 -> 8 % 4 = 0 -> 1st hour
        # Example: 09:30 -> 9 % 4 = 1 -> 2nd hour
        # Example: 10:45 -> 10 % 4 = 2 -> 3rd hour
        # Example: 11:59 -> 11 % 4 = 3 -> 4th hour
        phase = (hour % 4) + 1
        return phase

    def _calculate_ipda_phase(self, t: datetime.datetime, fractal_h: int) -> IPDAGlobalState:
        """
        Determines market state based on Time of Day AND Fractal Phase.
        """
        # MACRO CYCLE (Time of Day)
        # 17:00 - 00:00 NY -> ASIA -> ACCUMULATION
        # 02:00 - 05:00 NY -> LONDON -> MANIPULATION/EXPANSION
        # 07:00 - 11:00 NY -> NEW YORK -> EXPANSION
        # 12:00 - 17:00 NY -> PM SESSION -> DISTRIBUTION/ACCUMULATION
        
        # Simplified Logic for Robustness:
        h = t.hour
        
        if 18 <= h <= 23 or 0 <= h < 2:
            return IPDAGlobalState.ACCUMULATION
        
        if 2 <= h < 5:
            # London often manipulates early, expands later
            return IPDAGlobalState.MANIPULATION if h < 3 else IPDAGlobalState.EXPANSION
            
        if 5 <= h < 7:
            # Lunch Gap (Pre-NY)
            return IPDAGlobalState.ACCUMULATION
            
        if 7 <= h < 11:
            # NY Session (Prime Time)
            return IPDAGlobalState.EXPANSION
            
        return IPDAGlobalState.DISTRIBUTION # PM Session

    def _check_kill_zone(self, t: datetime.datetime):
        """
        Checks if we are in a high-probability Kill Window.
        """
        # Convert to time object for comparison
        ct = t.time()
        
        # Kill Zones (NY Time)
        # LONDON: 02:00 - 05:00
        # NY OPEN: 07:00 - 10:00
        # LONDON CLOSE: 10:00 - 12:00 (Extension)
        
        if datetime.time(2,0) <= ct < datetime.time(5,0):
            return KillZone.LONDON_OPEN, True
            
        if datetime.time(7,0) <= ct < datetime.time(10,0):
            return KillZone.NY_OPEN, True
            
        if datetime.time(10,0) <= ct < datetime.time(12,0):
            return KillZone.LONDON_CLOSE, True
            
        if datetime.time(19,0) <= ct or ct < datetime.time(0,0):
            return KillZone.ASIA, False # Active range but NOT a Kill Zone for entry
            
        return KillZone.NONE, False

    def _calculate_quality_score(self, ipda: IPDAGlobalState, kz_active: bool) -> float:
        """
        Bayesian Probability Score for Timing.
        """
        base_score = 0.1
        
        if ipda == IPDAGlobalState.ACCUMULATION:
            base_score = 0.1 # Don't trade rangess
        elif ipda == IPDAGlobalState.MANIPULATION:
            base_score = 0.4 # Careful, can trade the inducement fade
        elif ipda == IPDAGlobalState.EXPANSION:
            base_score = 0.7 # Go time
        elif ipda == IPDAGlobalState.DISTRIBUTION:
            base_score = 0.3 # Fade moves only
            
        # Kill Zone Multiplier
        if kz_active:
            base_score = min(0.99, base_score * 1.5) # Boost probability
            
        return round(base_score, 4)
