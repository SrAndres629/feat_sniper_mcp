import logging
from typing import Dict, Any, Tuple, Optional, Deque
from collections import deque
from app.core.config import settings

logger = logging.getLogger("feat.services.spread_filter")

class SpreadFilter:
    """
    Filtro de Ejecución para Protección de Puntos (Institutional Grade).
    Bloquea si el spread actual es > 3x la media histórica del símbolo.
    """
    def __init__(self):
        self.max_spread_multiplier = 1.5 # Definitive Institutional Grade
        self._history: Dict[str, deque] = {}

    def _get_history(self, symbol: str) -> deque:
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=100)
        return self._history[symbol]

    def is_spread_toxic(self, symbol: str, current_spread: float, avg_spread: Optional[float] = None) -> bool:
        """
        Valida si el spread actual es aceptable para la ejecución HFT.
        [SENIOR] Autonomous Memory fallback if avg_spread is missing.
        """
        history = self._get_history(symbol)
        
        # Self-Learning Fallback
        if avg_spread is None or avg_spread <= 0:
            if len(history) < 10:
                history.append(current_spread)
                return False # Not enough data to judge
            avg_spread = sum(history) / len(history)
        
        history.append(current_spread)
        
        if avg_spread > 0:
            ratio = current_spread / avg_spread
            if ratio > self.max_spread_multiplier:
                logger.warning(f"🛡️ SPREAD FILTER: Toxic Liquidity on {symbol} ({ratio:.1f}x avg)")
                return True
        return False

spread_filter = SpreadFilter()
