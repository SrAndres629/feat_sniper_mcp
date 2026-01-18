import logging
from datetime import datetime, timezone
from typing import Dict, Any
from .models import CircuitState, CircuitBreakerState

logger = logging.getLogger("feat.black_swan.circuit")

class MultiLevelCircuitBreaker:
    """Institutional Multi-Level Circuit Breaker."""
    LEVEL_1_WARNING = 0.02
    LEVEL_2_REDUCED = 0.04
    LEVEL_3_OPEN = 0.06
    TOTAL_DD_WARNING = 0.10
    TOTAL_DD_HALT = 0.20
    LOT_MULTIPLIERS = {CircuitState.CLOSED: 1.0, CircuitState.WARNING: 0.75, CircuitState.REDUCED: 0.25, CircuitState.OPEN: 0.0}
    
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.peak_equity = initial_balance
        self.daily_start_balance = initial_balance
        self.last_reset_date = datetime.now(timezone.utc).date()
        self._current_state = CircuitState.CLOSED
        self._requires_reset = False
        self._state_history = []
        logger.info(f"[BLACK_SWAN] CircuitBreaker initialized. Balance: ${initial_balance:.2f}")

    def check(self, current_equity: float) -> CircuitBreakerState:
        self._check_daily_reset(current_equity)
        if current_equity > self.peak_equity: self.peak_equity = current_equity
        daily_dd = max(0, (self.daily_start_balance - current_equity) / self.daily_start_balance) if self.daily_start_balance > 0 else 0.0
        total_dd = max(0, (self.peak_equity - current_equity) / self.peak_equity) if self.peak_equity > 0 else 0.0
        
        new_state = self._evaluate_state(daily_dd, total_dd)
        if new_state != self._current_state:
            self._log_change(new_state, daily_dd, total_dd)
            self._current_state = new_state
            if new_state == CircuitState.OPEN: self._requires_reset = True
            
        return CircuitBreakerState(self._current_state, daily_dd*100, total_dd*100, self.LOT_MULTIPLIERS[self._current_state], 
                                    self._current_state != CircuitState.OPEN, self._requires_reset, self._get_msg(daily_dd, total_dd))

    def _evaluate_state(self, d_dd: float, t_dd: float) -> CircuitState:
        if t_dd >= self.TOTAL_DD_HALT or self._requires_reset or d_dd >= self.LEVEL_3_OPEN: return CircuitState.OPEN
        if d_dd >= self.LEVEL_2_REDUCED: return CircuitState.REDUCED
        if d_dd >= self.LEVEL_1_WARNING or t_dd >= self.TOTAL_DD_WARNING: return CircuitState.WARNING
        return CircuitState.CLOSED

    def _check_daily_reset(self, eq: float):
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset_date:
            self.daily_start_balance = eq
            self.last_reset_date = today
            if self._current_state in [CircuitState.WARNING, CircuitState.REDUCED]: self._current_state = CircuitState.CLOSED

    def _log_change(self, new: CircuitState, d: float, t: float):
        logger.warning(f"[BLACK_SWAN] Circuit Breaker: {self._current_state.value} -> {new.value} (DD: {d*100:.1f}%)")
        if new == CircuitState.OPEN: logger.critical("🚨 CIRCUIT BREAKER OPEN! MANUAL RESET REQUIRED.")

    def _get_msg(self, d: float, t: float) -> str:
        msgs = {CircuitState.CLOSED: f"✅ Normal (DD: {d*100:.1f}%)", CircuitState.WARNING: f"⚠️ ADVERTENCIA: DD {d*100:.1f}%",
                CircuitState.REDUCED: f"🔴 REDUCIDO: DD {d*100:.1f}%", CircuitState.OPEN: f"🚨 DETENIDO: DD {d*100:.1f}% - Reinicio manual"}
        return msgs.get(self._current_state, "")

    def manual_reset(self, new_eq: float, code: str) -> bool:
        if code != f"RESET_{datetime.now(timezone.utc).strftime('%Y%m%d')}": return False
        self._requires_reset, self._current_state = False, CircuitState.CLOSED
        self.daily_start_balance, self.peak_equity = new_eq, new_eq
        return True

    def get_status(self) -> Dict[str, Any]:
        return {"state": self._current_state.value, "requires_reset": self._requires_reset, "peak": self.peak_equity}
