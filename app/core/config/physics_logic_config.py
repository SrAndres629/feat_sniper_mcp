from pydantic import BaseModel
from typing import Dict

class PhysicsLogicSettings(BaseModel):
    """
    [DOCTORAL AUDIT] Centralized Mathematical Constants and Scoring Weights.
    Eliminates hardcoded "magic numbers" from the logic engines.
    """
    
    # Structure Engine Scoring
    STRUCT_WEIGHTS: Dict[str, float] = {"F": 0.35, "E": 0.25, "A": 0.20, "T": 0.20}
    STRUCT_EXPANSION_SCORE: float = 0.4
    STRUCT_BOS_SCORE: float = 0.3
    STRUCT_CHOCH_SCORE: float = 0.3
    STRUCT_FVG_SCORE_HIGH: float = 0.5
    STRUCT_FVG_SCORE_LOW: float = 0.2
    
    # MTF Engine Constants
    MTF_THRESHOLD_SNIPER_TRIGGER: float = 0.75
    MTF_THRESHOLD_SNIPER: float = 0.85
    MTF_THRESHOLD_SIGNAL: float = 0.65
    MTF_WEIGHTS: Dict[str, float] = {
        "W1": 0.05, "D1": 0.10, "H4": 0.20, "H1": 0.20,
        "M30": 0.10, "M15": 0.15, "M5": 0.10, "M1": 0.10
    }
    MTF_ATR_PROXY_MULTIPLIER: float = 0.001
    MTF_SL_ATR_MULTIPLIER: float = 10.0
    MTF_TP_ATR_MULTIPLIER: float = 20.0
    
    # Acceleration Engine
    ACCEL_ATR_WINDOW: int = 14
    ACCEL_VOL_WINDOW: int = 20
    ACCEL_SCORE_THRESHOLD: float = 0.70
    ACCEL_SIGMA_THRESHOLD: float = 3.0
    ACCEL_NEWTON_THRESHOLD: float = 1.5
    ACCEL_WEIGHTS: Dict[str, float] = {
        "w1": 0.4, # Displacement
        "w2": 0.3, # Volume Z-Score
        "w3": 0.2, # FVG Presence
        "w4": 0.1  # Velocity
    }
    
    # Zone Projector Engine
    ZONE_VOL_EXTREME_TH: float = 3.0
    ZONE_VOL_HIGH_TH: float = 1.5
    ZONE_VOL_LOW_TH: float = 0.8
    ZONE_PROB_BASE_HIGH_VOL: float = 0.75
    ZONE_PROB_BASE_LOW_VOL: float = 0.7
    ZONE_PROB_BASE_NORMAL: float = 0.65
    ZONE_DIST_MULTIPLIER: float = 10.0
    ZONE_CONF_KZ_BOOST: float = 0.15
    ZONE_CONF_VOL_BOOST: float = 0.1
    ZONE_FVG_PROB: float = 0.65
    
    # Risk Logic Parameters
    RISK_BASE_RR_RATIO: float = 1.5
    RISK_DAMPING_FACTOR: float = 0.5
    CONVERGENCE_MAX_UNCERTAINTY: float = 0.08
    CONVERGENCE_MIN_SCORE: float = 0.65
    NEURAL_SNIPER_TH: float = 0.85
    NEURAL_ASSERTIVE_TH: float = 0.70
    NEURAL_DEFENSIVE_TH: float = 0.60
    EXHAUSTION_ATR_THRESHOLD: float = 2.5
    
    # Volatility Guard Parameters
    VOL_HALT_REL_THRESHOLD: float = 0.05
    VOL_DEAD_MARKET_THRESHOLD: float = 0.0005
    VOL_RATIO_MULTIPLIER: float = 3.0
    VOL_TURBULENT_RATIO: float = 1.5
    VOL_LAMINAR_RATIO: float = 0.8
    VOL_MEMORY_SAMPLES: int = 50
    
    # Dashboard & Operational Configuration
    DASHBOARD_LIVE_STATE_PATH: str = "data/live_state.json"
    DASHBOARD_COMMAND_PATH: str = "data/app_commands.json"
    DASHBOARD_REFRESH_JS_MS: int = 2000
    DASHBOARD_REFRESH_SLEEP_SEC: int = 1
    ZONE_PROXIMITY_FACTOR: float = 0.0005
    # Circuit Breaker & SRE Parameters
    CB_RATE_LIMIT_CAPACITY: int = 20
    CB_RATE_LIMIT_FILL_RATE: float = 0.5
    CB_MAX_LATENCY_SEC: float = 600.0
    CB_LEVEL_2_PAUSE_SEC: int = 3600
    CB_LEVEL_1_LOT_MULT: float = 0.75
    CB_LEVEL_2_LOT_MULT: float = 0.50
    CB_CONSECUTIVE_FAILURES_LIMIT: int = 5
    CB_MONITOR_INTERVAL_SEC: float = 5.0
    CB_RATE_LIMIT_FILE: str = "data/rate_limit.json"
    
    # Analytics & Journaling
    ANALYTICS_TRADING_DAYS: int = 252
    JOURNAL_PATH: str = "data/trade_journal.json"
    JOURNAL_MONTE_CARLO_ITERATIONS: int = 1000
