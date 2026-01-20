from __future__ import annotations
from pydantic import BaseModel
from typing import Any

class TradingSettings(BaseModel):
    SYMBOL: str = "XAUUSD"
    MT5_LOGIN: int | None = None
    MT5_PASSWORD: str | None = None
    MT5_SERVER: str | None = None
    MT5_PATH: str | None = None
    MT5_MAGIC_NUMBER: int = 123456
    MT5_ORDER_COMMENT: str = "FEAT_AI_Sniper_V2"
    TRADING_MODE: str = "SHADOW"
    EXECUTION_ENABLED: bool = False
    SHADOW_MODE: bool = True
    DRY_RUN: bool = False
    HEADLESS_MODE: bool = False
    MAGIC_SCALP: int = 234001
    MAGIC_SWING: int = 234002
    INITIAL_CAPITAL: float = 20.0
    SCALP_TARGET_USD: float = 2.0
    SWING_TARGET_USD: float = 10.0
    EQUITY_UNLOCK_THRESHOLD: float = 50.0
    SCALP_TIME_LIMIT_SECONDS: int = 300
