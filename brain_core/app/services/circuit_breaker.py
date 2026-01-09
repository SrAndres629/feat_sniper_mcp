import logging
import time
from typing import Dict, Any
from app.core.config import settings

logger = logging.getLogger("MT5_Bridge.Services.CircuitBreaker")

class CircuitBreaker:
    """
    Institutional Circuit Breaker System.
    Prevents execution cascade failures during market anomalies or API downtime.
    """
    _state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    _failures: int = 0
    _last_failure_time: float = 0
    _recovery_timeout: int = 60 # Seconds to wait before HALF_OPEN

    @classmethod
    def record_failure(cls):
        """Increments failure count and trips if threshold reached."""
        if not settings.ENABLE_CIRCUIT_BREAKER:
            return

        cls._failures += 1
        cls._last_failure_time = time.time()
        
        if cls._failures >= settings.CB_FAILURE_THRESHOLD:
            cls._state = "OPEN"
            logger.critical(f"CIRCUIT BREAKER: TRIPPED! State: {cls._state}. Failures: {cls._failures}")

    @classmethod
    def record_success(cls):
        """Resets failures and closes the circuit."""
        if cls._state != "CLOSED":
            logger.info(f"CIRCUIT BREAKER: RECOVERY DETECTED. Closing circuit.")
        
        cls._failures = 0
        cls._state = "CLOSED"

    @classmethod
    def can_execute(cls) -> bool:
        """Determines if the system is allowed to perform operations."""
        if not settings.ENABLE_CIRCUIT_BREAKER or cls._state == "CLOSED":
            return True

        # Check for recovery timeout
        if cls._state == "OPEN":
            if (time.time() - cls._last_failure_time) > cls._recovery_timeout:
                cls._state = "HALF_OPEN"
                logger.warning("CIRCUIT BREAKER: Transitioning to HALF_OPEN. Probing system...")
                return True
            return False
        
        return True

    @classmethod
    def get_status(cls) -> Dict[str, Any]:
        return {
            "state": cls._state,
            "failures": cls._failures,
            "last_failure": cls._last_failure_time,
            "healthy": cls._state != "OPEN"
        }

circuit_breaker = CircuitBreaker()
