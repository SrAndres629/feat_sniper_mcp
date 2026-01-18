from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

class VolatilityRegime(Enum):
    """Market volatility states for adaptive risk management."""
    COMPRESSED = "COMPRESSED"
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

@dataclass
class VolatilityState:
    """Current volatility regime with full context."""
    regime: VolatilityRegime
    atr_current: float
    atr_baseline: float
    atr_ratio: float
    zscore: float
    lot_multiplier: float
    can_trade: bool
    cooldown_until: Optional[datetime] = None
    message: str = ""

@dataclass
class SpreadState:
    """Current spread analysis with trading permission."""
    spread_current: float
    spread_baseline: float
    spread_ratio: float
    spread_atr_ratio: float
    is_normal: bool
    lot_multiplier: float
    message: str

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"
    WARNING = "WARNING"
    REDUCED = "REDUCED"
    OPEN = "OPEN"

@dataclass
class CircuitBreakerState:
    """Current circuit breaker status."""
    state: CircuitState
    daily_drawdown_pct: float
    total_drawdown_pct: float
    lot_multiplier: float
    can_trade: bool
    requires_manual_reset: bool
    message: str

@dataclass
class BlackSwanDecision:
    """Unified decision from all guards."""
    can_trade: bool
    lot_multiplier: float
    volatility: VolatilityState
    spread: Optional[SpreadState]
    circuit: CircuitBreakerState
    rejection_reasons: list
    timestamp: str
