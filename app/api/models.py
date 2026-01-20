"""
FEAT SNIPER: API Data Models
============================
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class SimulationMode(str, Enum):
    STANDARD = "standard"
    ADVERSARIAL = "adversarial"
    IMITATION = "imitation"


class SimulationRequest(BaseModel):
    """Request to start a simulation."""
    episodes: int = Field(default=5, ge=1, le=10000, description="Number of training episodes")
    mode: SimulationMode = Field(default=SimulationMode.ADVERSARIAL, description="Simulation mode")
    save_weights: bool = Field(default=True, description="Save neural weights after training")


class SimulationStatus(BaseModel):
    """Current status of a running simulation."""
    running: bool = False
    current_episode: int = 0
    total_episodes: int = 0
    current_balance: float = 20.0
    start_time: Optional[datetime] = None
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0
    last_update: Optional[datetime] = None


class SystemStatus(BaseModel):
    """Overall system health status."""
    daemon_running: bool = True
    engine_state: str = "IDLE"
    api_version: str = "1.0.0"
    uptime_seconds: float = 0.0
    simulation_active: bool = False
    mt5_connected: bool = False
    positions_count: int = 0
    account_balance: float = 0.0
    account_equity: float = 0.0


class PerformanceReport(BaseModel):
    """Historical performance analytics."""
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_pnl_pips: float = 0.0
    avg_trade_duration_minutes: float = 0.0
    exit_reasons: Dict[str, int] = {}
    equity_curve: List[float] = []


class LogEntry(BaseModel):
    """A single log entry for streaming."""
    timestamp: datetime
    level: str
    message: str
    source: str = "system"


class CommandResponse(BaseModel):
    """Standard API response."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class RiskProfileRequest(BaseModel):
    """Request to update risk profile."""
    risk_factor: float = Field(default=1.0, ge=0.1, le=5.0, description="Risk aggression multiplier")
