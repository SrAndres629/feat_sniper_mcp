from .models import (
    VolatilityRegime, VolatilityState, SpreadState, 
    CircuitState, CircuitBreakerState, BlackSwanDecision
)
from .volatility import VolatilityGuard
from .spread import SpreadGuard
from .circuit import MultiLevelCircuitBreaker
from .unified import BlackSwanGuard

# Singleton Instances for backward compatibility
black_swan_guard = BlackSwanGuard(initial_balance=20.0)

def get_black_swan_guard(initial_balance: float = None) -> BlackSwanGuard:
    """Get or reinitialize the Black Swan Guard."""
    global black_swan_guard
    if initial_balance is not None:
        black_swan_guard = BlackSwanGuard(initial_balance=initial_balance)
    return black_swan_guard
