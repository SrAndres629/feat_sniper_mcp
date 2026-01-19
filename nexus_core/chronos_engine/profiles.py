from dataclasses import dataclass
from enum import Enum

class StrategyMode(Enum):
    RANGE_BOUND = "RANGE_BOUND"         # Mean Reversion (Asia)
    TREND_FOLLOWING = "TREND_FOLLOWING" # Breakout/Expansion (London/NY)
    SCALPING = "SCALPING"               # Quick hits (Lunch/Lull)
    NO_TRADE = "NO_TRADE"               # Blackout

@dataclass
class SessionProfile:
    name: str
    target_multiplier: float # 1.0 = Conservative, 2.0 = Aggressive
    stop_loss_type: str      # "TIGHT", "STRUCTURAL"
    strategy_mode: StrategyMode
    expected_volatility: float # 0.0 to 1.0 (Information for Neural Net)

class ProfileLibrary:
    """
    [DYNAMIC PROFILES]
    Defines the 'Rules of Engagement' for each specific time window.
    The AI allows trading in ALL non-blackout windows, but adapts parameters.
    """
    
    ASIA_PROFILE = SessionProfile(
        name="ASIA_STANDARD",
        target_multiplier=1.0, # Don't aim for home runs
        stop_loss_type="TIGHT",
        strategy_mode=StrategyMode.RANGE_BOUND,
        expected_volatility=0.3
    )
    
    LONDON_NY_PROFILE = SessionProfile(
        name="KILLZONE_STANDARD",
        target_multiplier=2.0, # Aim for expansion
        stop_loss_type="STRUCTURAL",
        strategy_mode=StrategyMode.TREND_FOLLOWING,
        expected_volatility=0.9
    )
    
    LULL_PROFILE = SessionProfile(
        name="MIDDAY_LULL",
        target_multiplier=0.8, # Scalp only
        stop_loss_type="TIGHT",
        strategy_mode=StrategyMode.SCALPING,
        expected_volatility=0.4
    )
    
    BLACKOUT_PROFILE = SessionProfile(
        name="HARD_BLACKOUT",
        target_multiplier=0.0,
        stop_loss_type="NONE",
        strategy_mode=StrategyMode.NO_TRADE,
        expected_volatility=0.0 # Or infinite risk
    )
