import logging
from typing import Dict, Any, Tuple
from app.core.config import settings

logger = logging.getLogger("feat.services.spread_filter")

class SpreadFilter:
    """
    Filtro de Ejecución para Protección de Puntos (Institutional Grade).
    Bloquea si el spread actual es > 3x la media histórica del símbolo.
    """
    def __init__(self):
        self.max_spread_multiplier = 3.0 # Visionary Standard

    def is_spread_toxic(self, symbol: str, current_spread: float, avg_spread: float) -> bool:
        """
        Valida si el spread actual es aceptable para la ejecución HFT.
        """
        if avg_spread > 0:
            ratio = current_spread / avg_spread
            if ratio > self.max_spread_multiplier:
                logger.warning(f"🛡️ SPREAD FILTER: Toxic Liquidity on {symbol} ({ratio:.1f}x avg)")
                return True
        return False

spread_filter = SpreadFilter()
