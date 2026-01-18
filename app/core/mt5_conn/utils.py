import logging
from typing import TypeVar, Callable

logger = logging.getLogger("MT5_Bridge.Utils")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

T = TypeVar("T")

def resilient(*args, **kwargs):
    """Fallback decorator for observability-less environments."""
    def decorator(func):
        return func
    return decorator

# Attempt to load advanced observability
try:
    from app.core.observability import obs_engine, tracer
except ImportError:
    obs_engine = None
    tracer = None
