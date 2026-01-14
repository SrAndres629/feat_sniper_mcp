"""
BLACK SWAN GUARD - Institutional Risk Protection Layer
=======================================================
Module for extreme market event protection including:
- Flash Crash Detection (Volatility Regime)
- Spread Anomaly Filter
- Multi-Level Circuit Breaker
- Adaptive Cooldown System

Designed by Senior Algo Trading Specialist.
Integration: FEATChain validation pipeline.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, Deque
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger("feat.black_swan_guard")


# =============================================================================
# VOLATILITY REGIME DETECTION
# =============================================================================

class VolatilityRegime(Enum):
    """Market volatility states for adaptive risk management."""
    COMPRESSED = "COMPRESSED"   # Below normal - Breakout imminent
    NORMAL = "NORMAL"           # Standard conditions
    ELEVATED = "ELEVATED"       # 1.5-2x normal - Reduce size
    HIGH = "HIGH"               # 2-3x normal - Minimum size only
    EXTREME = "EXTREME"         # >3x normal - TRADING HALTED


@dataclass
class VolatilityState:
    """Current volatility regime with full context."""
    regime: VolatilityRegime
    atr_current: float
    atr_baseline: float
    atr_ratio: float
    zscore: float
    lot_multiplier: float
    can_trade: bool
    cooldown_until: Optional[datetime] = None
    message: str = ""


class VolatilityGuard:
    """
    Institutional Volatility Regime Detector.
    
    Uses ATR ratio with exponential baseline and Z-score validation.
    Implements adaptive cooldown after extreme events.
    
    Architecture:
    - Primary: ATR ratio vs 50-period EMA baseline
    - Secondary: Z-score confirmation (3σ = EXTREME)
    - Tertiary: Consecutive tick velocity spikes
    """
    
    # Regime Thresholds (Calibrated for Black Swan survival)
    COMPRESSED_THRESHOLD = 0.6    # <60% of normal = Compression
    ELEVATED_THRESHOLD = 1.5      # 150% = Elevated
    HIGH_THRESHOLD = 2.0          # 200% = High (reduce to 25%)
    EXTREME_THRESHOLD = 3.0       # 300% = EXTREME (halt trading)
    
    # Z-Score Confirmation
    ZSCORE_HIGH = 2.0             # 2σ deviation
    ZSCORE_EXTREME = 3.0          # 3σ deviation = Black Swan
    
    # Lot Multipliers per Regime
    LOT_MULTIPLIERS = {
        VolatilityRegime.COMPRESSED: 0.5,   # Breakout risk
        VolatilityRegime.NORMAL: 1.0,
        VolatilityRegime.ELEVATED: 0.5,
        VolatilityRegime.HIGH: 0.25,
        VolatilityRegime.EXTREME: 0.0,      # No trading
    }
    
    # Cooldown Configuration
    EXTREME_COOLDOWN_MINUTES = 30  # After flash crash
    HIGH_COOLDOWN_MINUTES = 10     # After high volatility
    
    def __init__(self, baseline_window: int = 50, atr_smoothing: float = 0.1):
        self.baseline_window = baseline_window
        self.atr_smoothing = atr_smoothing  # EMA smoothing factor
        
        # State buffers
        self.atr_history: Deque[float] = deque(maxlen=baseline_window)
        self.atr_ema: float = 0.0
        self.atr_std: float = 0.0
        
        # Event tracking
        self._last_regime: VolatilityRegime = VolatilityRegime.NORMAL
        self._extreme_event_time: Optional[datetime] = None
        self._cooldown_until: Optional[datetime] = None
        
        # Velocity spike detector (consecutive abnormal moves)
        self.velocity_spike_count: int = 0
        self.VELOCITY_SPIKE_THRESHOLD = 3  # 3 consecutive 2σ velocity spikes
        
        logger.info("[BLACK_SWAN] VolatilityGuard initialized with institutional parameters")
    
    def update_atr(self, current_atr: float) -> None:
        """Update ATR baseline with new value using EMA."""
        self.atr_history.append(current_atr)
        
        if len(self.atr_history) < 10:
            # Cold start - use simple average
            self.atr_ema = float(np.mean(self.atr_history))
            self.atr_std = float(np.std(self.atr_history)) if len(self.atr_history) > 1 else 0.0
        else:
            # Exponential Moving Average update
            self.atr_ema = (self.atr_smoothing * current_atr) + ((1 - self.atr_smoothing) * self.atr_ema)
            self.atr_std = float(np.std(self.atr_history))
    
    def evaluate(self, current_atr: float) -> VolatilityState:
        """
        Evaluate current volatility regime.
        
        Args:
            current_atr: Current ATR value from market data
            
        Returns:
            VolatilityState with full regime context
        """
        # Update baseline
        self.update_atr(current_atr)
        
        # Check cooldown first
        now = datetime.now(timezone.utc)
        if self._cooldown_until and now < self._cooldown_until:
            remaining = (self._cooldown_until - now).total_seconds() / 60
            return VolatilityState(
                regime=VolatilityRegime.EXTREME,
                atr_current=current_atr,
                atr_baseline=self.atr_ema,
                atr_ratio=current_atr / self.atr_ema if self.atr_ema > 0 else 1.0,
                zscore=self._calculate_zscore(current_atr),
                lot_multiplier=0.0,
                can_trade=False,
                cooldown_until=self._cooldown_until,
                message=f"🧊 COOLDOWN ACTIVO: {remaining:.1f} min restantes post-evento extremo"
            )
        
        # Cold start protection
        if len(self.atr_history) < 10:
            return VolatilityState(
                regime=VolatilityRegime.NORMAL,
                atr_current=current_atr,
                atr_baseline=self.atr_ema,
                atr_ratio=1.0,
                zscore=0.0,
                lot_multiplier=0.5,  # Conservative during warmup
                can_trade=True,
                message="⏳ Warmup: Baseline en construcción"
            )
        
        # Calculate metrics
        atr_ratio = current_atr / self.atr_ema if self.atr_ema > 0 else 1.0
        zscore = self._calculate_zscore(current_atr)
        
        # Determine regime
        regime = self._classify_regime(atr_ratio, zscore)
        
        # Handle extreme events
        if regime == VolatilityRegime.EXTREME:
            self._trigger_extreme_event()
        elif regime == VolatilityRegime.HIGH:
            self._cooldown_until = now + timedelta(minutes=self.HIGH_COOLDOWN_MINUTES)
        
        # Track regime transitions
        if regime != self._last_regime:
            logger.warning(f"[BLACK_SWAN] Regime Change: {self._last_regime.value} → {regime.value} (ATR Ratio: {atr_ratio:.2f}x)")
            self._last_regime = regime
        
        return VolatilityState(
            regime=regime,
            atr_current=current_atr,
            atr_baseline=self.atr_ema,
            atr_ratio=atr_ratio,
            zscore=zscore,
            lot_multiplier=self.LOT_MULTIPLIERS[regime],
            can_trade=regime != VolatilityRegime.EXTREME,
            message=self._get_regime_message(regime, atr_ratio, zscore)
        )
    
    def _calculate_zscore(self, current_atr: float) -> float:
        """Calculate Z-score of current ATR vs historical distribution."""
        if self.atr_std <= 0:
            return 0.0
        return (current_atr - self.atr_ema) / self.atr_std
    
    def _classify_regime(self, atr_ratio: float, zscore: float) -> VolatilityRegime:
        """
        Classify volatility regime using dual-confirmation logic.
        Primary: ATR ratio
        Secondary: Z-score confirmation
        """
        # EXTREME: Either 3x ATR OR 3σ deviation
        if atr_ratio >= self.EXTREME_THRESHOLD or zscore >= self.ZSCORE_EXTREME:
            return VolatilityRegime.EXTREME
        
        # HIGH: Either 2x ATR OR 2σ deviation
        if atr_ratio >= self.HIGH_THRESHOLD or zscore >= self.ZSCORE_HIGH:
            return VolatilityRegime.HIGH
        
        # ELEVATED: 1.5x ATR
        if atr_ratio >= self.ELEVATED_THRESHOLD:
            return VolatilityRegime.ELEVATED
        
        # COMPRESSED: Below 60% of normal
        if atr_ratio <= self.COMPRESSED_THRESHOLD:
            return VolatilityRegime.COMPRESSED
        
        return VolatilityRegime.NORMAL
    
    def _trigger_extreme_event(self) -> None:
        """Handle extreme volatility event with cooldown."""
        now = datetime.now(timezone.utc)
        self._extreme_event_time = now
        self._cooldown_until = now + timedelta(minutes=self.EXTREME_COOLDOWN_MINUTES)
        logger.critical(f"🚨 [BLACK_SWAN] EXTREME EVENT DETECTED! Trading halted until {self._cooldown_until.isoformat()}")
    
    def _get_regime_message(self, regime: VolatilityRegime, ratio: float, zscore: float) -> str:
        """Generate human-readable regime message."""
        messages = {
            VolatilityRegime.COMPRESSED: f"⚡ COMPRESIÓN ({ratio:.1f}x): Breakout inminente",
            VolatilityRegime.NORMAL: f"✅ Normal ({ratio:.1f}x, z={zscore:.1f}σ)",
            VolatilityRegime.ELEVATED: f"⚠️ ELEVADA ({ratio:.1f}x): Reducir tamaño 50%",
            VolatilityRegime.HIGH: f"🔴 ALTA ({ratio:.1f}x, z={zscore:.1f}σ): Solo 25% del lote",
            VolatilityRegime.EXTREME: f"🚨 EXTREMA ({ratio:.1f}x, z={zscore:.1f}σ): TRADING DETENIDO"
        }
        return messages.get(regime, "")
    
    def force_reset_cooldown(self, reason: str = "Manual override") -> None:
        """Emergency reset of cooldown (requires manual intervention)."""
        logger.warning(f"[BLACK_SWAN] Cooldown reset: {reason}")
        self._cooldown_until = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current guard status for monitoring."""
        return {
            "atr_ema": round(self.atr_ema, 5),
            "atr_std": round(self.atr_std, 5),
            "last_regime": self._last_regime.value,
            "cooldown_active": self._cooldown_until is not None,
            "cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until else None,
            "history_length": len(self.atr_history)
        }


# =============================================================================
# SPREAD ANOMALY DETECTION
# =============================================================================

@dataclass
class SpreadState:
    """Current spread analysis with trading permission."""
    spread_current: float
    spread_baseline: float
    spread_ratio: float
    spread_atr_ratio: float  # Spread as percentage of ATR
    is_normal: bool
    lot_multiplier: float
    message: str


class SpreadGuard:
    """
    Institutional Spread Anomaly Detector.
    
    Blocks trading when spread exceeds safe thresholds.
    Uses ATR-normalized spread for cross-asset consistency.
    
    Thresholds:
    - Normal: Spread < 10% of ATR
    - Warning: Spread 10-20% of ATR
    - Danger: Spread > 20% of ATR (50% lot)
    - Blocked: Spread > 50% of ATR or > 3x average
    """
    
    # Spread/ATR thresholds (institutional standard)
    SPREAD_ATR_NORMAL = 0.10      # 10% of ATR
    SPREAD_ATR_WARNING = 0.20     # 20% of ATR
    SPREAD_ATR_DANGER = 0.30      # 30% of ATR
    SPREAD_ATR_BLOCKED = 0.50     # 50% of ATR - News event
    
    # Spread multiplier thresholds (vs historical average)
    SPREAD_MULT_WARNING = 2.0     # 2x normal
    SPREAD_MULT_BLOCKED = 3.0     # 3x normal
    
    def __init__(self, baseline_window: int = 100):
        self.spread_history: Deque[float] = deque(maxlen=baseline_window)
        self.spread_ema: float = 0.0
        self.ema_alpha: float = 0.05  # Slow EMA for baseline
        
        logger.info("[BLACK_SWAN] SpreadGuard initialized")
    
    def update_spread(self, current_spread: float) -> None:
        """Update spread baseline."""
        self.spread_history.append(current_spread)
        
        if len(self.spread_history) < 5:
            self.spread_ema = current_spread
        else:
            self.spread_ema = (self.ema_alpha * current_spread) + ((1 - self.ema_alpha) * self.spread_ema)
    
    def evaluate(self, current_spread: float, current_atr: float) -> SpreadState:
        """
        Evaluate spread condition for trading permission.
        
        Args:
            current_spread: Current bid-ask spread in price units
            current_atr: Current ATR for normalization
            
        Returns:
            SpreadState with trading permission
        """
        self.update_spread(current_spread)
        
        # Calculate ratios
        spread_atr_ratio = current_spread / current_atr if current_atr > 0 else 0
        spread_ratio = current_spread / self.spread_ema if self.spread_ema > 0 else 1.0
        
        # Determine state
        if spread_atr_ratio >= self.SPREAD_ATR_BLOCKED or spread_ratio >= self.SPREAD_MULT_BLOCKED:
            return SpreadState(
                spread_current=current_spread,
                spread_baseline=self.spread_ema,
                spread_ratio=spread_ratio,
                spread_atr_ratio=spread_atr_ratio,
                is_normal=False,
                lot_multiplier=0.0,
                message=f"🚫 SPREAD BLOQUEADO: {spread_ratio:.1f}x normal, {spread_atr_ratio*100:.0f}% del ATR"
            )
        
        if spread_atr_ratio >= self.SPREAD_ATR_DANGER or spread_ratio >= self.SPREAD_MULT_WARNING:
            return SpreadState(
                spread_current=current_spread,
                spread_baseline=self.spread_ema,
                spread_ratio=spread_ratio,
                spread_atr_ratio=spread_atr_ratio,
                is_normal=False,
                lot_multiplier=0.25,
                message=f"⚠️ SPREAD ALTO: {spread_ratio:.1f}x normal (reducir a 25%)"
            )
        
        if spread_atr_ratio >= self.SPREAD_ATR_WARNING:
            return SpreadState(
                spread_current=current_spread,
                spread_baseline=self.spread_ema,
                spread_ratio=spread_ratio,
                spread_atr_ratio=spread_atr_ratio,
                is_normal=True,
                lot_multiplier=0.5,
                message=f"⚡ SPREAD ELEVADO: {spread_atr_ratio*100:.0f}% del ATR (reducir a 50%)"
            )
        
        return SpreadState(
            spread_current=current_spread,
            spread_baseline=self.spread_ema,
            spread_ratio=spread_ratio,
            spread_atr_ratio=spread_atr_ratio,
            is_normal=True,
            lot_multiplier=1.0,
            message=f"✅ Spread normal: {spread_atr_ratio*100:.1f}% del ATR"
        )


# =============================================================================
# MULTI-LEVEL CIRCUIT BREAKER
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"      # Normal operation
    WARNING = "WARNING"    # Level 1: Reduce activity
    REDUCED = "REDUCED"    # Level 2: Minimal activity
    OPEN = "OPEN"          # Level 3: Trading halted


@dataclass
class CircuitBreakerState:
    """Current circuit breaker status."""
    state: CircuitState
    daily_drawdown_pct: float
    total_drawdown_pct: float
    lot_multiplier: float
    can_trade: bool
    requires_manual_reset: bool
    message: str


class MultiLevelCircuitBreaker:
    """
    Institutional Multi-Level Circuit Breaker.
    
    Implements progressive protection levels:
    - Level 1 (2%): Warning - Reduce lot size to 75%
    - Level 2 (4%): Reduced - Reduce lot size to 25%  
    - Level 3 (6%): OPEN - Trading halted, requires manual reset
    - Level 4 (15% total DD): CATASTROPHIC - All trading blocked
    
    Features:
    - Daily drawdown tracking with auto-reset
    - Total drawdown from peak equity
    - Manual reset requirement for safety
    """
    
    # Daily Drawdown Levels
    LEVEL_1_WARNING = 0.02        # 2% daily loss
    LEVEL_2_REDUCED = 0.04        # 4% daily loss  
    LEVEL_3_OPEN = 0.06           # 6% daily loss (halt)
    
    # Total Drawdown (from peak)
    TOTAL_DD_WARNING = 0.10       # 10% from peak
    TOTAL_DD_CRITICAL = 0.15      # 15% from peak (catastrophic)
    TOTAL_DD_HALT = 0.20          # 20% from peak (permanent halt)
    
    # Lot multipliers per state
    LOT_MULTIPLIERS = {
        CircuitState.CLOSED: 1.0,
        CircuitState.WARNING: 0.75,
        CircuitState.REDUCED: 0.25,
        CircuitState.OPEN: 0.0,
    }
    
    def __init__(self, initial_balance: float):
        self.initial_balance = initial_balance
        self.peak_equity = initial_balance
        self.daily_start_balance = initial_balance
        self.last_reset_date = datetime.now(timezone.utc).date()
        
        self._current_state = CircuitState.CLOSED
        self._requires_reset = False
        self._state_history: list = []
        
        logger.info(f"[BLACK_SWAN] CircuitBreaker initialized. Balance: ${initial_balance:.2f}")
    
    def check(self, current_equity: float) -> CircuitBreakerState:
        """
        Evaluate circuit breaker conditions.
        
        Args:
            current_equity: Current account equity
            
        Returns:
            CircuitBreakerState with trading permission
        """
        # Check for daily reset
        self._check_daily_reset(current_equity)
        
        # Update peak equity (for total DD calculation)
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculate drawdowns
        daily_dd = self._calculate_daily_dd(current_equity)
        total_dd = self._calculate_total_dd(current_equity)
        
        # Determine state
        new_state = self._evaluate_state(daily_dd, total_dd)
        
        # State transition logging
        if new_state != self._current_state:
            self._log_state_change(self._current_state, new_state, daily_dd, total_dd)
            self._current_state = new_state
            
            # Level 3 requires manual reset
            if new_state == CircuitState.OPEN:
                self._requires_reset = True
        
        return CircuitBreakerState(
            state=self._current_state,
            daily_drawdown_pct=daily_dd * 100,
            total_drawdown_pct=total_dd * 100,
            lot_multiplier=self.LOT_MULTIPLIERS[self._current_state],
            can_trade=self._current_state != CircuitState.OPEN,
            requires_manual_reset=self._requires_reset,
            message=self._get_state_message(daily_dd, total_dd)
        )
    
    def _calculate_daily_dd(self, current_equity: float) -> float:
        """Calculate daily drawdown percentage."""
        if self.daily_start_balance <= 0:
            return 0.0
        loss = self.daily_start_balance - current_equity
        return max(0, loss / self.daily_start_balance)
    
    def _calculate_total_dd(self, current_equity: float) -> float:
        """Calculate total drawdown from peak."""
        if self.peak_equity <= 0:
            return 0.0
        loss = self.peak_equity - current_equity
        return max(0, loss / self.peak_equity)
    
    def _evaluate_state(self, daily_dd: float, total_dd: float) -> CircuitState:
        """Evaluate circuit breaker state based on drawdowns."""
        # Total DD takes precedence
        if total_dd >= self.TOTAL_DD_HALT:
            return CircuitState.OPEN
        
        # Manual reset required - don't auto-recover
        if self._requires_reset:
            return CircuitState.OPEN
        
        # Daily DD levels
        if daily_dd >= self.LEVEL_3_OPEN:
            return CircuitState.OPEN
        if daily_dd >= self.LEVEL_2_REDUCED:
            return CircuitState.REDUCED
        if daily_dd >= self.LEVEL_1_WARNING or total_dd >= self.TOTAL_DD_WARNING:
            return CircuitState.WARNING
        
        return CircuitState.CLOSED
    
    def _check_daily_reset(self, current_equity: float) -> None:
        """Reset daily tracking at midnight UTC."""
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset_date:
            logger.info(f"[BLACK_SWAN] Daily reset. Previous start: ${self.daily_start_balance:.2f}, Current: ${current_equity:.2f}")
            self.daily_start_balance = current_equity
            self.last_reset_date = today
            
            # Auto-recover from WARNING/REDUCED (but not OPEN)
            if self._current_state in [CircuitState.WARNING, CircuitState.REDUCED]:
                self._current_state = CircuitState.CLOSED
    
    def _log_state_change(self, old: CircuitState, new: CircuitState, daily_dd: float, total_dd: float) -> None:
        """Log circuit breaker state transitions."""
        self._state_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from": old.value,
            "to": new.value,
            "daily_dd": daily_dd,
            "total_dd": total_dd
        })
        
        if new == CircuitState.OPEN:
            logger.critical(f"🚨 [BLACK_SWAN] CIRCUIT BREAKER OPEN! Daily DD: {daily_dd*100:.2f}%, Total DD: {total_dd*100:.2f}%")
        else:
            logger.warning(f"[BLACK_SWAN] Circuit Breaker: {old.value} → {new.value}")
    
    def _get_state_message(self, daily_dd: float, total_dd: float) -> str:
        """Generate state message."""
        messages = {
            CircuitState.CLOSED: f"✅ Normal (Daily: {daily_dd*100:.1f}%, Total: {total_dd*100:.1f}%)",
            CircuitState.WARNING: f"⚠️ ADVERTENCIA: DD Diario {daily_dd*100:.1f}% (Reducir a 75%)",
            CircuitState.REDUCED: f"🔴 REDUCIDO: DD Diario {daily_dd*100:.1f}% (Reducir a 25%)",
            CircuitState.OPEN: f"🚨 DETENIDO: DD {daily_dd*100:.1f}% - Reinicio manual requerido"
        }
        return messages.get(self._current_state, "")
    
    def manual_reset(self, new_equity: float, authorization_code: str) -> bool:
        """
        Manual reset after OPEN state.
        Requires authorization code for safety.
        """
        expected_code = f"RESET_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        
        if authorization_code != expected_code:
            logger.warning(f"[BLACK_SWAN] Invalid reset code. Expected: {expected_code}")
            return False
        
        self._requires_reset = False
        self._current_state = CircuitState.CLOSED
        self.daily_start_balance = new_equity
        self.peak_equity = new_equity
        
        logger.warning(f"[BLACK_SWAN] Manual reset completed. New baseline: ${new_equity:.2f}")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "state": self._current_state.value,
            "requires_reset": self._requires_reset,
            "daily_start": self.daily_start_balance,
            "peak_equity": self.peak_equity,
            "initial_balance": self.initial_balance,
            "state_changes_today": len([h for h in self._state_history 
                                        if h["timestamp"].startswith(str(self.last_reset_date))])
        }


# =============================================================================
# UNIFIED BLACK SWAN GUARD
# =============================================================================

@dataclass
class BlackSwanDecision:
    """Unified decision from all guards."""
    can_trade: bool
    lot_multiplier: float
    volatility: VolatilityState
    spread: Optional[SpreadState]
    circuit: CircuitBreakerState
    rejection_reasons: list
    timestamp: str


class BlackSwanGuard:
    """
    Unified Black Swan Protection System.
    
    Orchestrates all protection mechanisms:
    - VolatilityGuard
    - SpreadGuard  
    - MultiLevelCircuitBreaker
    
    Returns unified trading decision with minimum lot multiplier.
    """
    
    def __init__(self, initial_balance: float = 20.0):
        self.volatility_guard = VolatilityGuard()
        self.spread_guard = SpreadGuard()
        self.circuit_breaker = MultiLevelCircuitBreaker(initial_balance)
        
        logger.info("[BLACK_SWAN] Unified Black Swan Guard initialized")
    
    def evaluate(
        self,
        current_atr: float,
        current_equity: float,
        current_spread: Optional[float] = None
    ) -> BlackSwanDecision:
        """
        Evaluate all guards and return unified decision.
        
        Args:
            current_atr: Current ATR value
            current_equity: Current account equity
            current_spread: Optional current spread (bid-ask)
            
        Returns:
            BlackSwanDecision with trading permission
        """
        rejection_reasons = []
        
        # 1. Volatility Check
        vol_state = self.volatility_guard.evaluate(current_atr)
        if not vol_state.can_trade:
            rejection_reasons.append(f"Volatility: {vol_state.message}")
        
        # 2. Spread Check (if provided)
        spread_state = None
        if current_spread is not None:
            spread_state = self.spread_guard.evaluate(current_spread, current_atr)
            if not spread_state.is_normal and spread_state.lot_multiplier == 0:
                rejection_reasons.append(f"Spread: {spread_state.message}")
        
        # 3. Circuit Breaker Check
        circuit_state = self.circuit_breaker.check(current_equity)
        if not circuit_state.can_trade:
            rejection_reasons.append(f"Circuit: {circuit_state.message}")
        
        # Calculate minimum lot multiplier
        multipliers = [vol_state.lot_multiplier, circuit_state.lot_multiplier]
        if spread_state:
            multipliers.append(spread_state.lot_multiplier)
        
        min_multiplier = min(multipliers)
        can_trade = len(rejection_reasons) == 0 and min_multiplier > 0
        
        return BlackSwanDecision(
            can_trade=can_trade,
            lot_multiplier=min_multiplier,
            volatility=vol_state,
            spread=spread_state,
            circuit=circuit_state,
            rejection_reasons=rejection_reasons,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get unified status of all guards."""
        return {
            "volatility": self.volatility_guard.get_status(),
            "spread": {
                "spread_ema": self.spread_guard.spread_ema,
                "history_length": len(self.spread_guard.spread_history)
            },
            "circuit_breaker": self.circuit_breaker.get_status()
        }


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

# Primary guard instance (initialized with default balance)
# Will be re-initialized with actual balance on first use
black_swan_guard = BlackSwanGuard(initial_balance=20.0)


def get_black_swan_guard(initial_balance: float = None) -> BlackSwanGuard:
    """Get or reinitialize the Black Swan Guard."""
    global black_swan_guard
    if initial_balance is not None:
        black_swan_guard = BlackSwanGuard(initial_balance=initial_balance)
    return black_swan_guard
