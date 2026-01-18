import os
from typing import List

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DATA_DIR = os.getenv("DATA_DIR", "data")
DB_PATH = os.path.join(DATA_DIR, "market_data.db")
N_LOOKAHEAD = int(os.getenv("N_LOOKAHEAD", "10"))
PROFIT_THRESHOLD = float(os.getenv("PROFIT_THRESHOLD", "0.002"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
FLUSH_INTERVAL = float(os.getenv("FLUSH_INTERVAL", "1.0"))

FEATURE_NAMES: List[str] = [
    "close", "open", "high", "low", "volume",
    "rsi", "atr", "ema_fast", "ema_slow",
    "feat_score", "fsm_state", "liquidity_ratio", "volatility_zscore",
    "momentum_kinetic_micro", "entropy_coefficient", "cycle_harmonic_phase", 
    "institutional_mass_flow", "volatility_regime_norm", "acceptance_ratio", 
    "wick_stress", "poc_z_score", "cvd_acceleration",
    "micro_comp", "micro_slope", "oper_slope", "macro_slope", "bias_slope", "fan_bullish"
]

TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]
TIMEFRAME_MAP = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440, "W1": 10080
}

class SystemState:
    BOOTING = "BOOTING"
    HYDRATING = "HYDRATING"
    READY = "READY"
    ERROR = "ERROR"
