import datetime
import pytz
from enum import Enum
from dataclasses import dataclass

class WeekDay(Enum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6

class H4CandleRole(Enum):
    CANDLE_1_ASIA_RANGE = "Candle 1 (Asia Range)"
    CANDLE_2_PRE_LONDON = "Candle 2 (Pre-London)"
    CANDLE_3_LONDON_EXPANSION = "Candle 3 (London Expansion)"
    CANDLE_4_NY_EARLY = "Candle 4 (NY Early)"
    CANDLE_5_NY_EXPANSION = "Candle 5 (NY Expansion)" # Specifically 11:00-15:00 but effectively the follow through
    CANDLE_6_CLOSE = "Candle 6 (Close)"

@dataclass
class MacroState:
    weekday: WeekDay
    cycle_weight: float # 1.0 (Tue/Wed) vs 0.6 (Mon/Fri)
    h4_candle_index: int # 1 to 6
    h4_candle_role: H4CandleRole
    is_expansion_window: bool # True if Tuesday/Wednesday AND in Expansion Candle

class MacroCycleTracker:
    """
    [EXPANSION HORIZON - MACRO LAYER]
    Tracks the 'Big Picture' Time Cycles:
    1. Weekly (Tue-Wed Expansion Bias)
    2. Intraday Fractal (The 6 H4 Candles)
    """
    
    def __init__(self):
        self.ny_tz = pytz.timezone('America/New_York')

    def get_macro_state(self, utc_time: datetime.datetime = None) -> MacroState:
        if utc_time is None:
            utc_time = datetime.datetime.now(datetime.timezone.utc)
            
        ny_time = utc_time.astimezone(self.ny_tz)
        
        # 1. Weekly Cycle
        weekday = WeekDay(ny_time.weekday())
        
        # Weighting: Tue (1) and Wed (2) are PRIME EXPANSION days
        cycle_weight = 0.6
        if weekday in [WeekDay.TUESDAY, WeekDay.WEDNESDAY]:
            cycle_weight = 1.0
        elif weekday == WeekDay.THURSDAY:
            cycle_weight = 0.8
            
        # 2. H4 Fractal (Intraday 6 Candles)
        # Block 1: 17:00 (Prev) - 21:00 (Index 1? No, 17 is start of day)
        # Standard FX Day starts 17:00 NY? Or 00:00 NY?
        # User defined: "Vela 4H #1 (19:00–23:00) Asia define rango inicial." (Bolivia Time)
        # Bolivia is UTC-4. NY is UTC-5 (Winter). Difference is 1 hour.
        # So 19:00 Bolivia = 18:00 NY.
        
        # Let's align with User's Table logic based on NY Time (Standard H4 Blocks)
        # Standard MT4 Server (GMT+2/3) usually aligns 00:00.
        # Let's stick to the User's defined 19:00 Bolivia start logic.
        # Gap: 17:00 - 19:00 NY is Rollover.
        # User Table: Candle #1 (19:00 - 23:00 Bolivia) -> 18:00 - 22:00 NY.
        
        hour_bolivia = (ny_time.hour + 1) % 24 # Crude timezone shift for logic alignment
        # Or better, just map strictly to the User's table hours which are Bolivia.
        # Let's convert to Bolivia for this specific calculation to be safe.
        bolivia_tz = pytz.timezone('America/La_Paz')
        bol_time = utc_time.astimezone(bolivia_tz)
        h = bol_time.hour
        
        # Cycle starts at 19:00 Bolivia.
        # We need an offset from 19:00.
        # If h < 19, add 24 to handle crossing midnight effectively for 'day' calculation relative to session start?
        # Simpler: Map ranges.
        
        # Candle 1: 19:00 - 23:00
        # Candle 2: 23:00 - 03:00
        # Candle 3: 03:00 - 07:00 (London)
        # Candle 4: 07:00 - 11:00 (NY Early)
        # Candle 5: 11:00 - 15:00 (NY Real)
        # Candle 6: 15:00 - 19:00 (Close)
        
        candle_idx = 0
        role = H4CandleRole.CANDLE_6_CLOSE # Default
        
        # Adjust time for check (treat 00:00-19:00 as part of "current cycle starting prev day 19:00", 
        # or easier: just strict hour check)
        
        if 19 <= h < 23:
            candle_idx = 1
            role = H4CandleRole.CANDLE_1_ASIA_RANGE
        elif (23 <= h) or (h < 3):
            candle_idx = 2
            role = H4CandleRole.CANDLE_2_PRE_LONDON
        elif 3 <= h < 7:
            candle_idx = 3
            role = H4CandleRole.CANDLE_3_LONDON_EXPANSION
        elif 7 <= h < 11:
            candle_idx = 4
            role = H4CandleRole.CANDLE_4_NY_EARLY
        elif 11 <= h < 15:
            candle_idx = 5
            role = H4CandleRole.CANDLE_5_NY_EXPANSION
        elif 15 <= h < 19:
            candle_idx = 6
            role = H4CandleRole.CANDLE_6_CLOSE
            
        # Expansion Window Logic
        is_expansion = False
        if cycle_weight >= 0.8: # Tue/Wed/Thu
            if candle_idx in [3, 5]: # London Exp or NY Exp
                is_expansion = True

        return MacroState(weekday, cycle_weight, candle_idx, role, is_expansion)
