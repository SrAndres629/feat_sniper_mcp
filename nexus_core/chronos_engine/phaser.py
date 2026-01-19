import datetime
import pytz
from enum import Enum, auto
from dataclasses import dataclass

class MicroPhase(Enum):
    # --- ASIA (19:00 - 04:00) ---
    ASIA_OPEN_ROLLOVER = "ASIA_OPEN_ROLLOVER" # 19:00 - 19:30
    ASIA_EARLY_CONSOLIDATION = "ASIA_EARLY_CONSOLIDATION" # 19:30 - 22:00
    ASIA_SETUP = "ASIA_SETUP"               # 22:00 - 00:00 (Algos define bias)
    ASIA_PRE_LONDON_INDUCEMENT = "ASIA_PRE_LONDON_INDUCEMENT"     # 00:00 - 01:30
    
    # --- LONDON (01:30 - 06:00) ---
    LONDON_PRE_OPEN = "LONDON_PRE_OPEN"             # 01:30 - 02:00
    LONDON_RAID = "LONDON_RAID"             # 02:00 - 03:00 (The Killzone Cleanse)
    LONDON_REACTION = "LONDON_REACTION"     # 03:00 - 03:30 (Retest)
    LONDON_EXPANSION = "LONDON_EXPANSION"   # 03:30 - 04:30 (Real Move)
    LONDON_TRANSITION = "LONDON_TRANSITION" # 04:30 - 06:00
    
    # --- EURO BREAK (06:00 - 07:00) ---
    EURO_LUNCH_BREAK = "EURO_LUNCH_BREAK"   # 06:00 - 07:00
    
    # --- NY (07:00 - 13:30) ---
    NY_PRE_INDUCTION = "NY_PRE_INDUCTION"     # 07:00 - 08:00
    NY_SOFT_OPEN = "NY_SOFT_OPEN"             # 08:00 - 08:30 (Algo Positioning)
    NY_OPEN_COMEX_START = "NY_OPEN_COMEX_START"   # 08:30 - 09:00 (High Volatility)
    NY_CONFIRMATION = "NY_CONFIRMATION"       # 09:00 - 09:30 (The Real Trend)
    NY_EXPANSION_MOMENTUM = "NY_EXPANSION_MOMENTUM" # 09:30 - 10:30
    NY_DISTRIBUTION = "NY_DISTRIBUTION"       # 10:30 - 11:15
    NY_MIDDAY_CONSOLIDATION = "NY_MIDDAY_CONSOLIDATION" # 11:15 - 12:30
    NY_PRE_CLOSE = "NY_PRE_CLOSE"             # 12:30 - 13:30
    
    # --- PM & CLOSES ---
    NY_PM_CLOSE = "NY_PM_CLOSE"           # 13:30 - 15:00
    POST_NY_LOW_LIQUIDITY = "POST_NY_LOW_LIQUIDITY" # 15:00 - 17:00
    SYSTEM_RESET = "SYSTEM_RESET"         # 17:00 - 19:00

    OFF_HOURS = "OFF_HOURS"

class Intent(Enum):
    ACCUMULATION = "ACCUMULATION"
    MANIPULATION = "MANIPULATION" # TRAP / RAID
    EXPANSION = "EXPANSION"       # REAL MOVE
    DISTRIBUTION = "DISTRIBUTION"

class VolatilityRegime(Enum):
    LOW = "LOW"           # Mean Reversion / Scalping (Asia)
    HIGH = "HIGH"         # Trend / Breakout (London/NY)
    EXTREME = "EXTREME"   # KillZones / News (High Risk/Reward)
    BLACKOUT = "BLACKOUT" # SPREAD DANGER (Close/Open)

from nexus_core.chronos_engine.profiles import ProfileLibrary, SessionProfile

@dataclass
class SessionState:
    micro_phase: MicroPhase
    intent: Intent
    risk_allowed: float 
    action_guard: str   
    warning_label: str
    vol_regime: VolatilityRegime
    profile: SessionProfile # The Dynamic Instructions

class GoldCyclePhaser:
    """
    [GOLD CYCLE SUPREMACY]
    The 'Doctrine Supreme' Minute-by-Minute Map for XAU/USD.
    Reference Timezone: Bolivia (UTC-4).
    """
    
    def __init__(self):
        self.bolivia_tz = pytz.timezone('America/La_Paz') 
        self.profiles = ProfileLibrary()

    def get_current_state(self, utc_time: datetime.datetime = None) -> SessionState:
        if utc_time is None:
            utc_time = datetime.datetime.now(datetime.timezone.utc)
            
        local_time = utc_time.astimezone(self.bolivia_tz)
        t = local_time.time()
        
        # --- BLACKOUT (17:00 - 20:00) ---
        # 17:00 - 18:00 EST (Close) -> 18:00 - 19:00 Bolivia
        # 18:00 - 19:00 EST (Open) -> 19:00 - 20:00 Bolivia
        
        # Pre-Close (Low Liq)
        if t >= datetime.time(17, 0) and t < datetime.time(18, 0):
             return SessionState(MicroPhase.POST_NY_LOW_LIQUIDITY, Intent.ACCUMULATION, 0.0, "NO_TRADE", "Dead Time", VolatilityRegime.LOW, self.profiles.BLACKOUT_PROFILE)

        # CME CLOSE (Actual Blackout)
        if t >= datetime.time(18, 0) and t < datetime.time(19, 0):
             return SessionState(MicroPhase.SYSTEM_RESET, Intent.ACCUMULATION, 0.0, "HARD_STOP", "CME CLOSE", VolatilityRegime.BLACKOUT, self.profiles.BLACKOUT_PROFILE)

        # CME OPEN (Actual Blackout)
        if t >= datetime.time(19, 0) and t < datetime.time(20, 0):
             return SessionState(MicroPhase.ASIA_OPEN_ROLLOVER, Intent.ACCUMULATION, 0.0, "HARD_STOP", "CME OPEN", VolatilityRegime.BLACKOUT, self.profiles.BLACKOUT_PROFILE)
        
        # --- ASIA SESSION (20:00 - 01:30) ---
        if t >= datetime.time(20, 0) and t < datetime.time(22, 0):
             return SessionState(MicroPhase.ASIA_EARLY_CONSOLIDATION, Intent.ACCUMULATION, 0.5, "MEAN_REVERSION", "Asia Range", VolatilityRegime.LOW, self.profiles.ASIA_PROFILE)
             
        if t >= datetime.time(22, 0) or t < datetime.time(0, 0): 
             if t >= datetime.time(22, 0):
                 return SessionState(MicroPhase.ASIA_SETUP, Intent.ACCUMULATION, 0.5, "MEAN_REVERSION", "Asia Bias", VolatilityRegime.LOW, self.profiles.ASIA_PROFILE)
        
        if t >= datetime.time(0, 0) and t < datetime.time(1, 30):
             return SessionState(MicroPhase.ASIA_PRE_LONDON_INDUCEMENT, Intent.MANIPULATION, 0.3, "TRAP_DANGER", "Pre-London Trap", VolatilityRegime.HIGH, self.profiles.ASIA_PROFILE)

        # --- LONDON (01:30 - 06:00) ---
        if t >= datetime.time(1, 30) and t < datetime.time(2, 0):
             return SessionState(MicroPhase.LONDON_PRE_OPEN, Intent.ACCUMULATION, 0.4, "WAIT", "London Prep", VolatilityRegime.LOW, self.profiles.LULL_PROFILE)

        if t >= datetime.time(2, 0) and t < datetime.time(3, 0):
             return SessionState(MicroPhase.LONDON_RAID, Intent.MANIPULATION, 0.6, "REVERSAL_ONLY", "LONDON TRAP", VolatilityRegime.EXTREME, self.profiles.LONDON_NY_PROFILE)

        if t >= datetime.time(3, 0) and t < datetime.time(3, 30):
             return SessionState(MicroPhase.LONDON_REACTION, Intent.MANIPULATION, 0.8, "CONFIRM_RETEST", "Reaction Check", VolatilityRegime.HIGH, self.profiles.LONDON_NY_PROFILE)

        if t >= datetime.time(3, 30) and t < datetime.time(4, 30):
             return SessionState(MicroPhase.LONDON_EXPANSION, Intent.EXPANSION, 1.0, "FULL_EXECUTION", "London Drive", VolatilityRegime.HIGH, self.profiles.LONDON_NY_PROFILE)

        if t >= datetime.time(4, 30) and t < datetime.time(6, 0):
             return SessionState(MicroPhase.LONDON_TRANSITION, Intent.DISTRIBUTION, 0.5, "TIGHT_SL", "Pre-NY Pause", VolatilityRegime.LOW, self.profiles.LULL_PROFILE)

        # --- EURO LUNCH (06:00 - 07:00) ---
        if t >= datetime.time(6, 0) and t < datetime.time(7, 0):
             return SessionState(MicroPhase.EURO_LUNCH_BREAK, Intent.ACCUMULATION, 0.2, "SCALP_ONLY", "Low Liquidity", VolatilityRegime.LOW, self.profiles.LULL_PROFILE)

        # --- NEW YORK (07:00 - 17:00) ---
        if t >= datetime.time(7, 0) and t < datetime.time(8, 0):
             return SessionState(MicroPhase.NY_PRE_INDUCTION, Intent.MANIPULATION, 0.4, "TRAP_DANGER", "Pre-NY Trap", VolatilityRegime.HIGH, self.profiles.LONDON_NY_PROFILE)

        if t >= datetime.time(8, 0) and t < datetime.time(8, 30):
             return SessionState(MicroPhase.NY_SOFT_OPEN, Intent.MANIPULATION, 0.5, "WAIT_VOL", "Algo Positioning", VolatilityRegime.HIGH, self.profiles.LONDON_NY_PROFILE)

        if t >= datetime.time(8, 30) and t < datetime.time(9, 0):
             return SessionState(MicroPhase.NY_OPEN_COMEX_START, Intent.EXPANSION, 0.8, "VOL_BREAKOUT", "COMEX OPEN", VolatilityRegime.EXTREME, self.profiles.LONDON_NY_PROFILE)

        if t >= datetime.time(9, 0) and t < datetime.time(9, 30):
             return SessionState(MicroPhase.NY_CONFIRMATION, Intent.EXPANSION, 1.0, "REQUIRE_OFI", "Confirmation Window", VolatilityRegime.HIGH, self.profiles.LONDON_NY_PROFILE)

        if t >= datetime.time(9, 30) and t < datetime.time(10, 30):
             return SessionState(MicroPhase.NY_EXPANSION_MOMENTUM, Intent.EXPANSION, 1.0, "FULL_EXECUTION", "Stock Flow Momentum", VolatilityRegime.HIGH, self.profiles.LONDON_NY_PROFILE)

        if t >= datetime.time(10, 30) and t < datetime.time(11, 15):
             return SessionState(MicroPhase.NY_DISTRIBUTION, Intent.DISTRIBUTION, 0.6, "TAKE_PROFIT", "Distribution", VolatilityRegime.HIGH, self.profiles.LONDON_NY_PROFILE)

        if t >= datetime.time(11, 15) and t < datetime.time(12, 30):
             return SessionState(MicroPhase.NY_MIDDAY_CONSOLIDATION, Intent.ACCUMULATION, 0.4, "SCALP_ONLY", "Midday Lull", VolatilityRegime.LOW, self.profiles.LULL_PROFILE)
             
        if t >= datetime.time(12, 30) and t < datetime.time(13, 30):
             return SessionState(MicroPhase.NY_PRE_CLOSE, Intent.DISTRIBUTION, 0.5, "TAKE_PROFIT", "Pre-Close Moves", VolatilityRegime.HIGH, self.profiles.LONDON_NY_PROFILE)

        if t >= datetime.time(13, 30) and t < datetime.time(15, 0):
             return SessionState(MicroPhase.NY_PM_CLOSE, Intent.DISTRIBUTION, 0.3, "CLOSE_ALL", "CME Close", VolatilityRegime.EXTREME, self.profiles.LONDON_NY_PROFILE)
             
        if t >= datetime.time(15, 0) and t < datetime.time(17, 0):
             return SessionState(MicroPhase.POST_NY_LOW_LIQUIDITY, Intent.ACCUMULATION, 0.1, "NO_TRADE", "Dead Time", VolatilityRegime.LOW, self.profiles.LULL_PROFILE)
             
        # Catch-all
        return SessionState(MicroPhase.OFF_HOURS, Intent.ACCUMULATION, 0.0, "NO_TRADE", "Off Hours", VolatilityRegime.LOW, self.profiles.BLACKOUT_PROFILE)
