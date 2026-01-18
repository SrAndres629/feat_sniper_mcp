import logging
import numpy as np
from typing import Optional, Deque, Dict, Any
from collections import deque
from datetime import datetime, timezone, timedelta
from .models import VolatilityRegime, VolatilityState

logger = logging.getLogger("feat.black_swan.volatility")

class VolatilityGuard:
    """Institutional Volatility Regime Detector."""
    COMPRESSED_THRESHOLD = 0.6
    ELEVATED_THRESHOLD = 1.5
    HIGH_THRESHOLD = 2.0
    EXTREME_THRESHOLD = 3.0
    ZSCORE_HIGH = 2.0
    ZSCORE_EXTREME = 3.0
    LOT_MULTIPLIERS = {
        VolatilityRegime.COMPRESSED: 0.5,
        VolatilityRegime.NORMAL: 1.0,
        VolatilityRegime.ELEVATED: 0.5,
        VolatilityRegime.HIGH: 0.25,
        VolatilityRegime.EXTREME: 0.0,
    }
    EXTREME_COOLDOWN_MINUTES = 30
    HIGH_COOLDOWN_MINUTES = 10
    
    def __init__(self, baseline_window: int = 50, atr_smoothing: float = 0.1):
        self.baseline_window = baseline_window
        self.atr_smoothing = atr_smoothing
        self.atr_history: Deque[float] = deque(maxlen=baseline_window)
        self.atr_ema: float = 0.0
        self.atr_std: float = 0.0
        self._last_regime: VolatilityRegime = VolatilityRegime.NORMAL
        self._cooldown_until: Optional[datetime] = None
        logger.info("[BLACK_SWAN] VolatilityGuard initialized")

    def update_atr(self, current_atr: float):
        self.atr_history.append(current_atr)
        if len(self.atr_history) < 10:
            self.atr_ema = float(np.mean(self.atr_history))
            self.atr_std = float(np.std(self.atr_history)) if len(self.atr_history) > 1 else 0.0
        else:
            self.atr_ema = (self.atr_smoothing * current_atr) + ((1 - self.atr_smoothing) * self.atr_ema)
            self.atr_std = float(np.std(self.atr_history))

    def evaluate(self, current_atr: float) -> VolatilityState:
        self.update_atr(current_atr)
        now = datetime.now(timezone.utc)
        if self._cooldown_until and now < self._cooldown_until:
            rem = (self._cooldown_until - now).total_seconds() / 60
            return VolatilityState(VolatilityRegime.EXTREME, current_atr, self.atr_ema, current_atr/self.atr_ema if self.atr_ema > 0 else 1.0, self._calculate_zscore(current_atr), 0.0, False, self._cooldown_until, f"🧊 COOLDOWN: {rem:.1f} min")
        
        if len(self.atr_history) < 10:
            return VolatilityState(VolatilityRegime.NORMAL, current_atr, self.atr_ema, 1.0, 0.0, 0.5, True, message="⏳ Warmup")
        
        ratio = current_atr / self.atr_ema if self.atr_ema > 0 else 1.0
        z = self._calculate_zscore(current_atr)
        regime = self._classify_regime(ratio, z)
        
        if regime == VolatilityRegime.EXTREME:
            self._cooldown_until = now + timedelta(minutes=self.EXTREME_COOLDOWN_MINUTES)
            logger.critical(f"🚨 EXTREME VOLATILITY DETECTED! Halt until {self._cooldown_until}")
        elif regime == VolatilityRegime.HIGH:
            self._cooldown_until = now + timedelta(minutes=self.HIGH_COOLDOWN_MINUTES)
            
        if regime != self._last_regime:
            logger.warning(f"[BLACK_SWAN] Regime Transition: {self._last_regime.value} -> {regime.value}")
            self._last_regime = regime
            
        return VolatilityState(regime, current_atr, self.atr_ema, ratio, z, self.LOT_MULTIPLIERS[regime], regime != VolatilityRegime.EXTREME, self._cooldown_until, self._get_msg(regime, ratio, z))

    def _calculate_zscore(self, current_atr: float) -> float:
        return (current_atr - self.atr_ema) / self.atr_std if self.atr_std > 0 else 0.0

    def _classify_regime(self, ratio: float, z: float) -> VolatilityRegime:
        if ratio >= self.EXTREME_THRESHOLD or z >= self.ZSCORE_EXTREME: return VolatilityRegime.EXTREME
        if ratio >= self.HIGH_THRESHOLD or z >= self.ZSCORE_HIGH: return VolatilityRegime.HIGH
        if ratio >= self.ELEVATED_THRESHOLD: return VolatilityRegime.ELEVATED
        if ratio <= self.COMPRESSED_THRESHOLD: return VolatilityRegime.COMPRESSED
        return VolatilityRegime.NORMAL

    def _get_msg(self, r: VolatilityRegime, ratio: float, z: float) -> str:
        msgs = {VolatilityRegime.COMPRESSED: f"⚡ COMPRESIÓN ({ratio:.1f}x)", VolatilityRegime.NORMAL: f"✅ Normal ({ratio:.1f}x, z={z:.1f})",
                VolatilityRegime.ELEVATED: f"⚠️ ELEVADA ({ratio:.1f}x)", VolatilityRegime.HIGH: f"🔴 ALTA ({ratio:.1f}x, z={z:.1f})", VolatilityRegime.EXTREME: "🚨 EXTREMA: HALT"}
        return msgs.get(r, "")

    def get_status(self) -> Dict[str, Any]:
        return {"atr_ema": round(self.atr_ema, 5), "last_regime": self._last_regime.value, "cooldown_active": self._cooldown_until is not None}
