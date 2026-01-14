"""
BLACK SWAN CHAOS TEST
=====================
Simulates extreme market conditions to validate the guard system.
- ATR 500% spike (Flash Crash)
- Spread 3x normal
- Drawdown 4% (Reduced mode)

Expected Outcome:
- VolatilityGuard should trigger EXTREME regime and block trading
- SpreadGuard should block the order
- All decisions logged to console
"""

import asyncio
import logging
import sys
import os

sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("chaos_test")

from app.skills.black_swan_guard import BlackSwanGuard, VolatilityRegime

async def run_chaos_test():
    print("=" * 60)
    print("🔥 BLACK SWAN CHAOS TEST INITIATED 🔥")
    print("=" * 60)
    
    # Initialize with $100 account
    guard = BlackSwanGuard(initial_balance=100.0)
    
    # --- Warmup Phase (Build baseline) ---
    print("\n[PHASE 1] Warmup: Building ATR baseline...")
    for i in range(20):
        # Normal ATR around 1.5
        decision = guard.evaluate(
            current_atr=1.5 + (i % 5) * 0.1,
            current_equity=100.0,
            current_spread=0.05
        )
    print(f"   Baseline ATR EMA: {guard.volatility_guard.atr_ema:.3f}")
    print(f"   Baseline Spread EMA: {guard.spread_guard.spread_ema:.4f}")
    
    # --- CHAOS: ATR 500% Spike ---
    print("\n[PHASE 2] INJECTING CHAOS: ATR 500% Spike...")
    chaos_atr = guard.volatility_guard.atr_ema * 5.0  # 500% of baseline
    chaos_spread = guard.spread_guard.spread_ema * 3.0  # 3x normal
    
    decision = guard.evaluate(
        current_atr=chaos_atr,
        current_equity=100.0,
        current_spread=chaos_spread
    )
    
    print(f"\n📊 CHAOS DECISION:")
    print(f"   Can Trade: {decision.can_trade}")
    print(f"   Lot Multiplier: {decision.lot_multiplier}")
    print(f"   Volatility Regime: {decision.volatility.regime.value}")
    print(f"   ATR Ratio: {decision.volatility.atr_ratio:.2f}x")
    print(f"   Spread Ratio: {decision.spread.spread_ratio:.2f}x" if decision.spread else "   Spread: N/A")
    print(f"   Rejection Reasons: {decision.rejection_reasons}")
    
    # --- Verify TRADING HALTED ---
    if decision.volatility.regime == VolatilityRegime.EXTREME:
        print("\n✅ TEST PASSED: VolatilityGuard correctly identified EXTREME regime")
    else:
        print("\n❌ TEST FAILED: VolatilityGuard did NOT trigger EXTREME regime")
    
    if not decision.can_trade:
        print("✅ TEST PASSED: Trading correctly blocked")
    else:
        print("❌ TEST FAILED: Trading should be blocked")
    
    # --- CHAOS: Drawdown Simulation ---
    print("\n[PHASE 3] SIMULATING 4% DRAWDOWN...")
    guard2 = BlackSwanGuard(initial_balance=100.0)
    # Warmup
    for _ in range(15):
        guard2.evaluate(current_atr=1.5, current_equity=100.0, current_spread=0.05)
    
    # Simulate loss
    dd_decision = guard2.evaluate(
        current_atr=1.5,
        current_equity=96.0,  # 4% loss
        current_spread=0.05
    )
    
    print(f"\n📊 DRAWDOWN DECISION:")
    print(f"   Circuit State: {dd_decision.circuit.state.value}")
    print(f"   Daily DD: {dd_decision.circuit.daily_drawdown_pct:.2f}%")
    print(f"   Lot Multiplier: {dd_decision.circuit.lot_multiplier}")
    
    if dd_decision.circuit.state.value == "REDUCED":
        print("✅ TEST PASSED: Circuit Breaker correctly entered REDUCED mode at 4%")
    else:
        print("❌ TEST FAILED: Expected REDUCED state at 4% drawdown")

    print("\n" + "=" * 60)
    print("🏁 CHAOS TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_chaos_test())
