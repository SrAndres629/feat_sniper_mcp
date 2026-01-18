import logging
from typing import Deque
from collections import deque
from .models import SpreadState

logger = logging.getLogger("feat.black_swan.spread")

class SpreadGuard:
    """Institutional Spread Anomaly Detector."""
    SPREAD_ATR_NORMAL = 0.10
    SPREAD_ATR_WARNING = 0.20
    SPREAD_ATR_DANGER = 0.30
    SPREAD_ATR_BLOCKED = 0.50
    SPREAD_MULT_WARNING = 2.0
    SPREAD_MULT_BLOCKED = 3.0
    
    def __init__(self, baseline_window: int = 100):
        self.spread_history: Deque[float] = deque(maxlen=baseline_window)
        self.spread_ema: float = 0.0
        self.ema_alpha: float = 0.05
        logger.info("[BLACK_SWAN] SpreadGuard initialized")

    def update_spread(self, current_spread: float):
        self.spread_history.append(current_spread)
        if len(self.spread_history) < 5:
            self.spread_ema = current_spread
        else:
            self.spread_ema = (self.ema_alpha * current_spread) + ((1 - self.ema_alpha) * self.spread_ema)

    def evaluate(self, current_spread: float, current_atr: float) -> SpreadState:
        self.update_spread(current_spread)
        ratio = current_spread / self.spread_ema if self.spread_ema > 0 else 1.0
        atr_ratio = current_spread / current_atr if current_atr > 0 else 0
        
        if atr_ratio >= self.SPREAD_ATR_BLOCKED or ratio >= self.SPREAD_MULT_BLOCKED:
            return SpreadState(current_spread, self.spread_ema, ratio, atr_ratio, False, 0.0, f"🚫 SPREAD BLOQUEADO: {ratio:.1f}x normal")
        if atr_ratio >= self.SPREAD_ATR_DANGER or ratio >= self.SPREAD_MULT_WARNING:
            return SpreadState(current_spread, self.spread_ema, ratio, atr_ratio, False, 0.25, f"⚠️ SPREAD ALTO: {ratio:.1f}x normal")
        if atr_ratio >= self.SPREAD_ATR_WARNING:
            return SpreadState(current_spread, self.spread_ema, ratio, atr_ratio, True, 0.5, f"⚡ SPREAD ELEVADO: {atr_ratio*100:.0f}% ATR")
        
        return SpreadState(current_spread, self.spread_ema, ratio, atr_ratio, True, 1.0, f"✅ Spread normal: {atr_ratio*100:.1f}% ATR")
