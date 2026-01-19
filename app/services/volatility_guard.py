import logging
import time
from typing import Dict, Any, Optional, Tuple, Deque
from collections import deque
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
        self.current_regime = "LAMINAR"
        
        # [SENIOR ARCHITECTURE] Relative Thresholds (Now from Settings)
        self.rel_threshold = settings.VOL_HALT_REL_THRESHOLD 
        self.dead_market_threshold = settings.VOL_DEAD_MARKET_THRESHOLD
        self.multiplier = settings.VOL_RATIO_MULTIPLIER
        self._atr_memories: Dict[str, deque] = {}

    def _get_memory(self, symbol: str) -> deque:
        if symbol not in self._atr_memories:
            self._atr_memories[symbol] = deque(maxlen=settings.VOL_MEMORY_SAMPLES)
        return self._atr_memories[symbol]
        
    def check_market_toxicity(self, market_data: Dict[str, Any]) -> bool:
        """
        Veredicto de toxicidad. Retorna True si el mercado es tóxico (HALT).
        [SENIOR] Relative Volatility Guard (ATR/Price) + Memory.
        """
        symbol = market_data.get("symbol", "UNKNOWN")
        price = float(market_data.get("bid", 0) or market_data.get("close", 0))
        atr = market_data.get("atr")
        avg_atr = market_data.get("avg_atr")
        
        memory = self._get_memory(symbol)
        
        if atr is not None:
            memory.append(atr)
            
            # Rule 1: Relative Volatility (Scale proof)
            if price > 0:
                rel_vol = atr / price
                if rel_vol > self.rel_threshold:
                    self.is_halted = True
                    self.halt_reason = f"Toxic Volatility: Relative ATR ({rel_vol:.2%}) > {self.rel_threshold:.2%}"
                    logger.critical(f"🛡️ VOLATILITY GUARD [RELATIVE_HIGH]: {self.halt_reason}")
                    return True
                
                # Rule 1b: Dead Market Check
                if rel_vol < self.dead_market_threshold:
                    self.is_halted = True
                    self.halt_reason = f"Dead Market: Relative ATR ({rel_vol:.4%}) < {self.dead_market_threshold:.4%}"
                    logger.warning(f"🛡️ VOLATILITY GUARD [RELATIVE_LOW]: {self.halt_reason}")
                    return True
            
            # Rule 2: Multiplier (Autonomous Memory Fallback)
            effective_avg = avg_atr if (avg_atr and avg_atr > 0) else (sum(memory)/len(memory) if len(memory) > 10 else None)
            
            if effective_avg and atr / effective_avg > self.multiplier:
                self.is_halted = True
                self.halt_reason = f"Flash Crash: ATR Ratio {atr/effective_avg:.1f}x > {self.multiplier}x"
                logger.critical(f"🛡️ VOLATILITY GUARD [RATIO]: {self.halt_reason}")
                return True
        
        self.is_halted = False
        self.halt_reason = ""
        return False

    def can_trade(self, market_data: Dict[str, Any]) -> Tuple[bool, str]:
        if self.check_market_toxicity(market_data):
            return False, self.halt_reason
        return True, "Safe Volatility"

    def get_regime(self, market_data: Dict[str, Any]) -> str:
        """Determines market flow regime based on volatility and speed."""
        if self.is_halted:
            return "TOXIC"
        
        atr = market_data.get("atr", 0)
        avg_atr = market_data.get("avg_atr", 1)
        
        if avg_atr == 0: return "NORMAL"
        
        ratio = atr / avg_atr
        if ratio > settings.VOL_TURBULENT_RATIO: return "TURBULENT"
        if ratio < settings.VOL_LAMINAR_RATIO: return "LAMINAR"
        return "NORMAL"

volatility_guard = VolatilityGuard()
