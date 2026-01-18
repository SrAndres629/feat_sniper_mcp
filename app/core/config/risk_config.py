from pydantic import BaseModel

class RiskSettings(BaseModel):
    MAX_DAILY_DRAWDOWN_PERCENT: float = 6.0
    CB_LEVEL_1_DD: float = 2.0
    CB_LEVEL_2_DD: float = 4.0
    CB_LEVEL_3_DD: float = 6.0
    RISK_PER_TRADE_PERCENT: float = 15.0
    MAX_OPEN_POSITIONS: int = 20
    MAX_CORRELATION_LIMIT: float = 0.65
    SPREAD_MAX_PIPS: float = 50.0
    ATR_SL_MULTIPLIER: float = 2.0
    ATR_TP_MULTIPLIER: float = 4.0
    ATR_TRAILING_MULTIPLIER: float = 1.5
    VOLATILITY_ADAPTIVE_LOTS: bool = True
    ENABLE_CIRCUIT_BREAKER: bool = True
    CB_FAILURE_THRESHOLD: int = 5
    MAX_UNCERTAINTY_THRESHOLD: float = 0.08
