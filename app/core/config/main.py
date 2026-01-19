from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field, Field
from typing import Tuple, Dict, Optional, Literal
from .constants import ExecutionMode
from .trading_config import TradingSettings
from .neural_config import NeuralSettings
from .risk_config import RiskSettings
from .system_config import SystemSettings
from .physics_logic_config import PhysicsLogicSettings

class Settings(BaseSettings, TradingSettings, NeuralSettings, RiskSettings, SystemSettings, PhysicsLogicSettings):
    """
    Master Configuration (Hybrid Architecture).
    Aggregates all specialized config modules while maintaining a flat API for backward compatibility.
    """
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @computed_field
    @property
    def execution_mode(self) -> ExecutionMode:
        if self.TRADING_MODE == "LIVE" and self.EXECUTION_ENABLED and not self.SHADOW_MODE:
            return ExecutionMode.LIVE
        elif self.TRADING_MODE == "PAPER" or (self.SHADOW_MODE and self.EXECUTION_ENABLED):
            return ExecutionMode.PAPER
        elif self.TRADING_MODE == "BACKTEST":
            return ExecutionMode.BACKTEST
        return ExecutionMode.SHADOW

    @computed_field
    @property
    def is_live_trading(self) -> bool:
        return self.execution_mode == ExecutionMode.LIVE

    @computed_field
    @property
    def effective_risk_cap(self) -> float:
        if self.is_live_trading: return min(self.RISK_PER_TRADE_PERCENT, 30.0)
        return self.RISK_PER_TRADE_PERCENT
