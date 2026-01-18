import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from .models import BlackSwanDecision
from .volatility import VolatilityGuard
from .spread import SpreadGuard
from .circuit import MultiLevelCircuitBreaker

logger = logging.getLogger("feat.black_swan.unified")

class BlackSwanGuard:
    """Unified Black Swan Protection System."""
    def __init__(self, initial_balance: float = 20.0):
        self.volatility_guard = VolatilityGuard()
        self.spread_guard = SpreadGuard()
        self.circuit_breaker = MultiLevelCircuitBreaker(initial_balance)
        logger.info("[BLACK_SWAN] Unified Black Swan Guard initialized")

    def evaluate(self, current_atr: float, current_equity: float, current_spread: Optional[float] = None) -> BlackSwanDecision:
        rejection_reasons = []
        vol_state = self.volatility_guard.evaluate(current_atr)
        if not vol_state.can_trade: rejection_reasons.append(f"Volatility: {vol_state.message}")
        
        spread_state = None
        if current_spread is not None:
            spread_state = self.spread_guard.evaluate(current_spread, current_atr)
            if not spread_state.is_normal and spread_state.lot_multiplier == 0:
                rejection_reasons.append(f"Spread: {spread_state.message}")
        
        circuit_state = self.circuit_breaker.check(current_equity)
        if not circuit_state.can_trade: rejection_reasons.append(f"Circuit: {circuit_state.message}")
        
        mults = [vol_state.lot_multiplier, circuit_state.lot_multiplier]
        if spread_state: mults.append(spread_state.lot_multiplier)
        
        min_mult = min(mults)
        can_trade = len(rejection_reasons) == 0 and min_mult > 0
        
        return BlackSwanDecision(can_trade, min_mult, vol_state, spread_state, circuit_state, rejection_reasons, datetime.now(timezone.utc).isoformat())

    def get_status(self) -> Dict[str, Any]:
        return {"volatility": self.volatility_guard.get_status(), "spread": {"spread_ema": self.spread_guard.spread_ema}, "circuit_breaker": self.circuit_breaker.get_status()}
