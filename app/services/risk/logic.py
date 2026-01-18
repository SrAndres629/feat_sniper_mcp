import logging
from typing import Dict, Any, Tuple, Optional, Literal
from datetime import datetime
from app.core.config import settings
from app.core.mt5_conn import mt5_conn, mt5
from app.services.volatility_guard import volatility_guard
from app.services.spread_filter import spread_filter

logger = logging.getLogger("Risk.Logic")

# =============================================================================
# KELLY CRITERION (SIZING)
# =============================================================================
def calculate_damped_kelly(win_prob: float, uncertainty: float, rr_ratio: float = 1.5) -> float:
    """[LEVEL 41] Bayesian Damped Kelly (PhD Protocol)."""
    max_unc = getattr(settings, "MAX_UNCERTAINTY_THRESHOLD", 0.08)
    if uncertainty > max_unc: return 0.0
    k_f = (win_prob * (rr_ratio + 1) - 1) / rr_ratio
    if k_f <= 0: return 0.0
    damping = 0.5 * (1.0 - (uncertainty / max_unc))
    final_f = k_f * max(0.0, damping)
    max_risk = (settings.RISK_PER_TRADE_PERCENT / 100.0) if hasattr(settings, "RISK_PER_TRADE_PERCENT") else 0.02
    return min(final_f, max_risk)

def get_neural_allocation(conf: float) -> str:
    if conf > 0.85: return "SNIPER"
    if conf > 0.70: return "ASSERTIVE"
    if conf < 0.60: return "DEFENSIVE"
    return "TEPID"

# =============================================================================
# VETOES (VALIDATION)
# =============================================================================
async def check_trading_veto(symbol: str, market_data: Dict[str, Any], cb_multiplier: float) -> Tuple[bool, str]:
    """Evaluates toxic conditions (Spread, Volatility, Circuit Breaker)."""
    regime = volatility_guard.get_regime(market_data)
    can_trade, reason = volatility_guard.can_trade(market_data)
    tick = await mt5_conn.execute(mt5.symbol_info_tick, symbol)
    spread = (tick.ask - tick.bid) if tick else 0
    is_toxic = spread_filter.is_spread_toxic(symbol, spread, market_data.get("avg_spread", 0))
    if not can_trade: return False, f"VolVeto: {reason}"
    if is_toxic: return False, "ToxicSpread"
    if cb_multiplier <= 0: return False, "CircuitBreakerTrip"
    return True, regime

# =============================================================================
# DRAWDOWN (SAFETY)
# =============================================================================
class DrawdownMonitor:
    """Manages daily balance reset and drawdown vetoes."""
    def __init__(self):
        self.opening_balance: Optional[float] = None
        self.last_reset: Optional[str] = None

    async def ensure_daily_reset(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if self.last_reset != today:
            acc = await mt5_conn.execute(mt5.account_info)
            if acc:
                self.opening_balance = acc.balance
                self.last_reset = today
                logger.info(f"Daily balance reset: ${self.opening_balance:.2f}")

    async def check_limit(self) -> Tuple[bool, float]:
        await self.ensure_daily_reset()
        acc = await mt5_conn.execute(mt5.account_info)
        if not acc or not self.opening_balance: return True, 0.0
        dd = ((self.opening_balance - acc.equity) / self.opening_balance) * 100
        if dd > settings.MAX_DAILY_DRAWDOWN_PERCENT:
            logger.warning(f"PHANTOM MODE: Daily DD {dd:.2f}% Hit (Limit: {settings.MAX_DAILY_DRAWDOWN_PERCENT}%)")
            return False, dd
        return True, dd
