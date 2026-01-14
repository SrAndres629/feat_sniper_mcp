"""
BLACK SWAN GUARD - Unit Tests
=============================
Tests for extreme market event protection mechanisms.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

# Import the guards
import sys
sys.path.insert(0, 'c:/Users/acord/OneDrive/Desktop/Bot/feat_sniper_mcp')

from app.skills.black_swan_guard import (
    VolatilityGuard, VolatilityRegime, VolatilityState,
    SpreadGuard, SpreadState,
    MultiLevelCircuitBreaker, CircuitState, CircuitBreakerState,
    BlackSwanGuard, BlackSwanDecision
)


class TestVolatilityGuard:
    """Tests for VolatilityGuard class."""
    
    def setup_method(self):
        """Fresh guard for each test."""
        self.guard = VolatilityGuard(baseline_window=50)
    
    def test_warmup_returns_normal(self):
        """During warmup, should return NORMAL with conservative multiplier."""
        state = self.guard.evaluate(current_atr=20.0)
        assert state.regime == VolatilityRegime.NORMAL
        assert state.lot_multiplier == 0.5  # Conservative during warmup
        assert state.can_trade == True
        assert "Warmup" in state.message
    
    def test_normal_volatility(self):
        """Normal volatility should allow full trading."""
        # Build baseline with consistent ATR
        for _ in range(20):
            self.guard.evaluate(current_atr=20.0)
        
        state = self.guard.evaluate(current_atr=20.0)
        assert state.regime == VolatilityRegime.NORMAL
        assert state.lot_multiplier == 1.0
        assert state.can_trade == True
    
    def test_extreme_volatility_halts_trading(self):
        """300%+ ATR should halt trading immediately."""
        # Build baseline
        for _ in range(20):
            self.guard.evaluate(current_atr=20.0)
        
        # Flash crash - ATR spikes to 5x normal
        state = self.guard.evaluate(current_atr=100.0)
        
        assert state.regime == VolatilityRegime.EXTREME
        assert state.lot_multiplier == 0.0
        assert state.can_trade == False
        assert "EXTREMA" in state.message or "EXTREME" in state.message
    
    def test_high_volatility_reduces_lot(self):
        """200%+ ATR should reduce to 25%."""
        for _ in range(20):
            self.guard.evaluate(current_atr=20.0)
        
        # High volatility - 2.5x normal
        state = self.guard.evaluate(current_atr=50.0)
        
        assert state.regime == VolatilityRegime.HIGH
        assert state.lot_multiplier == 0.25
        assert state.can_trade == True
    
    def test_cooldown_after_extreme_event(self):
        """After extreme event, should maintain cooldown."""
        for _ in range(20):
            self.guard.evaluate(current_atr=20.0)
        
        # Trigger extreme event
        state = self.guard.evaluate(current_atr=100.0)
        assert state.can_trade == False
        
        # Even if volatility returns to normal, cooldown active
        state = self.guard.evaluate(current_atr=20.0)
        assert state.can_trade == False
        assert state.cooldown_until is not None
        assert "COOLDOWN" in state.message
    
    def test_compressed_volatility(self):
        """Below 60% normal should detect compression."""
        for _ in range(20):
            self.guard.evaluate(current_atr=20.0)
        
        # Very low volatility
        state = self.guard.evaluate(current_atr=10.0)
        
        assert state.regime == VolatilityRegime.COMPRESSED
        assert state.lot_multiplier == 0.5  # Reduce for breakout risk


class TestSpreadGuard:
    """Tests for SpreadGuard class."""
    
    def setup_method(self):
        self.guard = SpreadGuard(baseline_window=100)
    
    def test_normal_spread_allows_trading(self):
        """Normal spread should allow full trading."""
        atr = 2.5  # 2.5 price points ATR
        spread = 0.1  # 4% of ATR - normal
        
        state = self.guard.evaluate(current_spread=spread, current_atr=atr)
        
        assert state.is_normal == True
        assert state.lot_multiplier == 1.0
    
    def test_high_spread_blocks_trading(self):
        """Spread > 50% of ATR should block trading."""
        atr = 2.5
        spread = 1.5  # 60% of ATR - blocked
        
        state = self.guard.evaluate(current_spread=spread, current_atr=atr)
        
        assert state.is_normal == False
        assert state.lot_multiplier == 0.0
        assert "BLOQUEADO" in state.message or "BLOCKED" in state.message
    
    def test_warning_spread_reduces_lot(self):
        """Spread 20-30% of ATR should reduce to 50%."""
        atr = 2.5
        spread = 0.6  # 24% of ATR
        
        state = self.guard.evaluate(current_spread=spread, current_atr=atr)
        
        assert state.lot_multiplier == 0.5


class TestMultiLevelCircuitBreaker:
    """Tests for MultiLevelCircuitBreaker class."""
    
    def setup_method(self):
        self.cb = MultiLevelCircuitBreaker(initial_balance=1000.0)
    
    def test_normal_conditions(self):
        """No drawdown should be CLOSED state."""
        state = self.cb.check(current_equity=1000.0)
        
        assert state.state == CircuitState.CLOSED
        assert state.lot_multiplier == 1.0
        assert state.can_trade == True
    
    def test_level_1_warning(self):
        """2% daily loss should trigger WARNING."""
        state = self.cb.check(current_equity=980.0)  # 2% loss
        
        assert state.state == CircuitState.WARNING
        assert state.lot_multiplier == 0.75
        assert state.can_trade == True
    
    def test_level_2_reduced(self):
        """4% daily loss should trigger REDUCED."""
        state = self.cb.check(current_equity=960.0)  # 4% loss
        
        assert state.state == CircuitState.REDUCED
        assert state.lot_multiplier == 0.25
        assert state.can_trade == True
    
    def test_level_3_open_halts_trading(self):
        """6% daily loss should OPEN circuit."""
        state = self.cb.check(current_equity=940.0)  # 6% loss
        
        assert state.state == CircuitState.OPEN
        assert state.lot_multiplier == 0.0
        assert state.can_trade == False
        assert state.requires_manual_reset == True
    
    def test_manual_reset_required(self):
        """After OPEN, cannot auto-recover."""
        # Trigger OPEN
        self.cb.check(current_equity=940.0)
        
        # Even with profit, still OPEN
        state = self.cb.check(current_equity=1100.0)
        
        assert state.state == CircuitState.OPEN
        assert state.requires_manual_reset == True
    
    def test_valid_manual_reset(self):
        """Manual reset with correct code should work."""
        self.cb.check(current_equity=940.0)
        
        expected_code = f"RESET_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        result = self.cb.manual_reset(new_equity=1000.0, authorization_code=expected_code)
        
        assert result == True
        state = self.cb.check(current_equity=1000.0)
        assert state.state == CircuitState.CLOSED


class TestBlackSwanGuard:
    """Integration tests for unified BlackSwanGuard."""
    
    def setup_method(self):
        self.guard = BlackSwanGuard(initial_balance=1000.0)
    
    def test_all_guards_pass(self):
        """Normal conditions should allow trading."""
        # Build baselines
        for _ in range(20):
            self.guard.volatility_guard.evaluate(20.0)
            self.guard.spread_guard.evaluate(0.1, 2.5)
        
        decision = self.guard.evaluate(
            current_atr=20.0,
            current_equity=1000.0,
            current_spread=0.1
        )
        
        assert decision.can_trade == True
        assert decision.lot_multiplier == 1.0
        assert len(decision.rejection_reasons) == 0
    
    def test_flash_crash_blocks_all(self):
        """Extreme volatility should block regardless of other conditions."""
        for _ in range(20):
            self.guard.volatility_guard.evaluate(20.0)
        
        decision = self.guard.evaluate(
            current_atr=100.0,  # 5x normal
            current_equity=1000.0,
            current_spread=0.1
        )
        
        assert decision.can_trade == False
        assert decision.lot_multiplier == 0.0
        assert len(decision.rejection_reasons) > 0
    
    def test_circuit_breaker_overrides(self):
        """Circuit breaker OPEN should block all trading."""
        for _ in range(20):
            self.guard.volatility_guard.evaluate(20.0)
        
        # Trigger circuit breaker
        decision = self.guard.evaluate(
            current_atr=20.0,
            current_equity=940.0,  # 6% loss
            current_spread=0.1
        )
        
        assert decision.can_trade == False
        assert "Circuit" in str(decision.rejection_reasons)
    
    def test_minimum_multiplier_applied(self):
        """Should use minimum of all multipliers."""
        for _ in range(20):
            self.guard.volatility_guard.evaluate(20.0)
            self.guard.spread_guard.evaluate(0.1, 2.5)
        
        # High spread (50% multiplier) + Level 1 CB (75% multiplier)
        decision = self.guard.evaluate(
            current_atr=20.0,
            current_equity=980.0,  # 2% loss -> WARNING
            current_spread=0.6     # 24% of ATR -> 50%
        )
        
        # Should be min(1.0, 0.5, 0.75) = 0.5
        assert decision.lot_multiplier == 0.5


# =============================================================================
# SIMULATION TESTS
# =============================================================================

class TestFlashCrashSimulation:
    """Simulates real flash crash scenarios."""
    
    def test_covid_march_2020(self):
        """Simulate COVID-19 crash volatility pattern."""
        guard = BlackSwanGuard(initial_balance=10000.0)
        
        # Build 2 weeks of normal data (10 trading days)
        for day in range(10):
            for hour in range(8):
                guard.volatility_guard.evaluate(current_atr=15.0)
        
        # March 9, 2020: Volatility starts rising
        for _ in range(8):
            guard.volatility_guard.evaluate(current_atr=25.0)
        
        # March 12: Flash crash - VIX at 75
        # ATR explodes to 4-5x normal
        decision = guard.evaluate(
            current_atr=65.0,  # 4.3x normal
            current_equity=9500.0,  # 5% DD
            current_spread=0.5
        )
        
        # System should HALT
        assert decision.can_trade == False
        assert "EXTREME" in str(decision.volatility.message) or decision.volatility.regime == VolatilityRegime.EXTREME
    
    def test_daily_drawdown_progression(self):
        """Test escalating daily losses trigger progressive protection."""
        guard = BlackSwanGuard(initial_balance=10000.0)
        
        # Build baseline
        for _ in range(20):
            guard.volatility_guard.evaluate(15.0)
        
        # Trade 1: Small loss (-1%)
        d1 = guard.evaluate(current_atr=15.0, current_equity=9900.0)
        assert d1.can_trade == True
        assert d1.lot_multiplier == 1.0
        
        # Trade 2: Loss compounds (-2.5%)
        d2 = guard.evaluate(current_atr=15.0, current_equity=9750.0)
        assert d2.can_trade == True
        assert d2.lot_multiplier == 0.75  # WARNING level
        
        # Trade 3: Significant loss (-4.5%)
        d3 = guard.evaluate(current_atr=15.0, current_equity=9550.0)
        assert d3.can_trade == True
        assert d3.lot_multiplier == 0.25  # REDUCED level
        
        # Trade 4: Critical loss (-6.5%)
        d4 = guard.evaluate(current_atr=15.0, current_equity=9350.0)
        assert d4.can_trade == False  # HALTED


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
