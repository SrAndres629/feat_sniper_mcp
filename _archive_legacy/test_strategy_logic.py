import asyncio
import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

# Add project root to path
sys.path.append(os.getcwd())

# Mock MT5
sys.modules['MetaTrader5'] = MagicMock()
import MetaTrader5 as mt5

# Must import AFTER mocking
from app.services.risk_engine import risk_engine, the_vault
from nexus_control import PerformanceTracker, TradingState, TradeRecord

async def test_validation_gate():
    print("\n⚔️  MODEL 3 VALIDATION GATE PROTOCOL  ⚔️")
    print("==========================================")
    
    # TEST 1: THE VAULT TRIGGER
    print("\n[TEST 1] The Vault Trigger (Doubling Capital)...")
    the_vault.initial_capital = 1000.0
    the_vault.last_trigger_equity = 1000.0
    the_vault.vault_balance = 0.0
    
    # Simulate equity doubling (1000 -> 2000)
    current_equity = 2000.0
    alert = the_vault.check_vault_trigger(current_equity)
    
    if alert and alert['type'] == 'VAULT_TRIGGER':
        print(f"  ✅ PASS: Vault Triggered! Transfer: ${alert['vault_amount']}")
    else:
        print("  ❌ FAIL: Vault did NOT trigger on 2x equity.")

    # TEST 2: SNIPER ASSIGNMENT (95% Confidence)
    print("\n[TEST 2] Neural Sniper Assignment (95% Confidence)...")
    
    # Patch get_adaptive_lots to return a base value multiplied by the neural multiplier
    # This proves calculate_dynamic_lot applies the multiplier correctly
    async def mock_get_adaptive_lots(symbol, sl, multiplier=1.0):
        return 1.0 * multiplier

    risk_engine.get_adaptive_lots = mock_get_adaptive_lots
    
    # Standard conf (51%) -> Expect mult 1.0 (actually <0.6 is 0.01 per strict rule, but 0.51 is <0.6???)
    # Wait, my code says <0.60 returns 0.01.
    # Let's test 0.65 (Moderate) -> Mult 1.0
    std_lot = await risk_engine.calculate_dynamic_lot(0.65, 0.5, "XAUUSD")
    
    # High conf (95%) -> Mult 1.5
    sniper_lot = await risk_engine.calculate_dynamic_lot(0.95, 0.5, "XAUUSD")
    
    # Low conf (50%) -> Should be min (0.01)
    min_lot = await risk_engine.calculate_dynamic_lot(0.50, 0.5, "XAUUSD")
    
    print(f"  > Standard Lot (65%): {std_lot}")
    print(f"  > Sniper Lot (95%):   {sniper_lot}")
    print(f"  > Min Lot (50%):      {min_lot}")
    
    if sniper_lot == 1.5 and std_lot == 1.0 and min_lot == 0.01:
        print("  ✅ PASS: Neural Multiplier & Zero Risk Logic verified.")
    else:
        print("  ❌ FAIL: Logic Mismatch.")

    # TEST 3: FSM RECALIBRATION (Losing Streak)
    print("\n[TEST 3] FSM Recalibration (Losing Streak)...")
    tracker = PerformanceTracker(state_file="data/test_fsm.json")
    tracker.trade_history = []
    tracker.current_state = TradingState.SUPERVISED
    tracker.MIN_TRADES_FOR_EVALUATION = 5 
    
    # Simulate 10 losing trades
    for i in range(10):
        tracker.record_trade(TradeRecord(
            trade_id=f"loss_{i}", symbol="XAUUSD", direction="BUY", entry_price=2000, 
            exit_price=1990, profit=-10.0, closed=True
        ))
        
    state = tracker.evaluate_and_transition()
    print(f"  > WinRate: {tracker.calculate_winrate():.2%}")
    print(f"  > Current State: {state.name}")
    
    if state == TradingState.RECALIBRATION:
        print("  ✅ PASS: System entered RECALIBRATION state.")
    else:
        print(f"  ❌ FAIL: System is in {state} instead of RECALIBRATON.")

    # Cleanup
    if os.path.exists("data/test_fsm.json"):
        os.remove("data/test_fsm.json")

if __name__ == "__main__":
    asyncio.run(test_validation_gate())
