class SystemGuardError(Exception):
    """Base exception for all system guard errors."""
    ...

class ArtifactIntegrityError(SystemGuardError):
    def __init__(self, path: str, error_type: str, details: str):
        self.path = path
        self.error_type = error_type
        self.details = details
        super().__init__(f"[{error_type}] {path}: {details}")

class RiskViolationError(SystemGuardError):
    """Raised when an operation violates a hard risk rule."""
    ...

class CircuitBreakerTrip(SystemGuardError):
    """Raised when the system detects a systemic anomaly."""
    ...

class ResourceExhaustionError(SystemGuardError):
    """Raised when the system predicts resource exhaustion."""
    ...
