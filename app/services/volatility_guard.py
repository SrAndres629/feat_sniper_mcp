import logging
import time
from typing import Dict, Any, Optional, Tuple
from app.core.config import settings

logger = logging.getLogger("feat.services.volatility_guard")

class VolatilityGuard:
    """
    Módulo de Protección de Liquidez (Institutional Grade).
    Bloquea la ejecución si la volatilidad instantánea (ATR) supera el 300% de la media.
    """
    def __init__(self):
        self.last_check_time = 0
        self.is_halted = False
        self.halt_reason = ""
        self.atr_threshold_multiplier = 3.0 # Visionary Standard

    def check_market_toxicity(self, market_data: Dict[str, Any]) -> bool:
        """
        Veredicto de toxicidad. Retorna True si el mercado es tóxico (HALT).
        """
        atr = market_data.get("atr")
        avg_atr = market_data.get("avg_atr")
        
        if atr is not None and avg_atr is not None and avg_atr > 0:
            vol_ratio = atr / avg_atr
            if vol_ratio > self.atr_threshold_multiplier:
                self.is_halted = True
                self.halt_reason = f"Flash Crash Detected: Volatility Ratio {vol_ratio:.1f}x > {self.atr_threshold_multiplier}x"
                logger.critical(f"🛡️ VOLATILITY GUARD TRIGGERED: {self.halt_reason}")
                return True
        
        self.is_halted = False
        self.halt_reason = ""
        return False

    def can_trade(self, market_data: Dict[str, Any]) -> Tuple[bool, str]:
        if self.check_market_toxicity(market_data):
            return False, self.halt_reason
        return True, "Safe Volatility"

volatility_guard = VolatilityGuard()
