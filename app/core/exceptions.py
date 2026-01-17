class RiskViolationError(Exception):
    """
    Se lanza cuando una operación viola una regla dura de 'The Vault' (Módulo 5).
    Debe abortar inmediatamente la secuencia de trading.
    """
    def __init__(self, message="Risk Policy Violation"):
        super().__init__(message)

class CircuitBreakerTrip(Exception):
    """
    Se lanza cuando el Módulo 9 detecta una anomalía sistémica (Latencia/Desconexión).
    """
    def __init__(self, message="Circuit Breaker Tripped"):
        super().__init__(message)
