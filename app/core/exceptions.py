class RiskViolationError(Exception):
    """
    Se lanza cuando una operación viola una regla dura de 'The Vault' (Módulo 5).
    Debe abortar inmediatamente la secuencia de trading.
    """
    pass

class CircuitBreakerTrip(Exception):
    """
    Se lanza cuando el Módulo 9 detecta una anomalía sistémica (Latencia/Desconexión).
    """
    pass
