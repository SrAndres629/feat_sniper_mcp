I will search for the `RiskEngine` class and `calculate_dynamic_lot` method, as well as `VolatilityGuard` and `SpreadFilter`, to understand their structure and how to mock the data for the stress test.
I will read `app/services/risk_engine.py`, `app/skills/black_swan_guard.py`, `app/services/volatility_guard.py`, and `app/services/spread_filter.py` to understand the logic and dependencies for creating the stress test.
I will read `app/services/circuit_breaker.py` to confirm the logic for the circuit breaker before writing the test script.
I'll create the `stress_test_guards.py` script to simulate the risk scenarios requested by the user. I'll mock the dependencies and verify that the `risk_engine` correctly blocks trading (returns 0.0 lots) for high volatility, toxic spread, and circuit breaker activation.
I'll try to use `run_shell_command` to create the file, since `write_file` appears to be unavailable. If `run_shell_command` is also unavailable, I'll inform the user about the limitation.
No tengo acceso a herramientas de escritura de archivos (`write_file`) o ejecución de comandos (`run_shell_command`) en esta sesión, por lo que no puedo crear ni ejecutar el script automáticamente.

Sin embargo, he diseñado el código completo para `stress_test_guards.py` que cumple con todos tus requisitos:
1.  **VolatilityGuard**: Mockea ATR 500 vs Avg 100 (Ratio 5x).
2.  **SpreadFilter**: Mockea Spread 30 vs Avg 5 (Ratio 6x).
3.  **Circuit Breaker**: Mockea Drawdown del 5% (Nivel 2) y asegura que el retorno sea 0.0 (simulando la "Pausa Operacional").

Puedes guardar este código en `stress_test_guards.py` y ejecutarlo manualmente:

import asyncio
import logging
import sys
from unittest.mock import MagicMock, patch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("StressTest")

async def main():
    print("--- Starting Stress Test Guards ---\n")

    # Patch settings and mt5 dependencies
    with patch("app.core.config.settings") as mock_settings, \
         patch("app.core.mt5_conn.mt5_conn") as mock_mt5:

        # Configure Settings for Circuit Breaker
        mock_settings.CB_LEVEL_1_DD = 2.0
        mock_settings.CB_LEVEL_2_DD = 4.0
        mock_settings.CB_LEVEL_3_DD = 6.0
        mock_settings.MAX_DAILY_DRAWDOWN_PERCENT = 10.0
        mock_settings.ATR_TRAILING_MULTIPLIER = 1.5
        mock_settings.INITIAL_CAPITAL = 1000.0
        mock_settings.EQUITY_UNLOCK_THRESHOLD = 2000.0

        # Import modules AFTER patching to ensure they pick up mocks
        from app.services.risk_engine import risk_engine
        from app.services.circuit_breaker import circuit_breaker
        import MetaTrader5 as mt5

        # Setup MT5 Mock for Spread checks
        mock_tick = MagicMock()
        mock_tick.ask = 1005.0
        mock_tick.bid = 1000.0
        
        async def mock_execute(cmd, *args, **kwargs):
            if cmd == mt5.symbol_info_tick:
                return mock_tick
            return MagicMock()
            
        mock_mt5.execute = MagicMock(side_effect=mock_execute)

        # Helper to set spread dynamically
        def set_spread(spread):
            mock_tick.ask = 1000.0 + spread
            mock_tick.bid = 1000.0

        # Mock circuit_breaker.get_drawdown
        async def mock_get_drawdown():
            return 0.0
        
        # Replace the method on the singleton instance
        circuit_breaker.get_drawdown = MagicMock(side_effect=mock_get_drawdown)

        # ---------------------------------------------------------
        # TEST 1: VolatilityGuard
        # ---------------------------------------------------------
        print("[TEST 1] VolatilityGuard Trigger (ATR 500, Avg 100)")
        market_data_vol = {
            "atr": 500.0,
            "avg_atr": 100.0,  # Ratio 5.0 > 3.0 (Threshold)
            "avg_spread": 5.0
        }
        set_spread(5.0) # Normal spread
        
        # Reset Circuit Breaker
        circuit_breaker.pause_until = 0 
        circuit_breaker.current_level = 0
        circuit_breaker.get_drawdown.side_effect = lambda: asyncio.Future()
        circuit_breaker.get_drawdown.return_value.set_result(0.0)

        # Execute
        lot_vol = await risk_engine.calculate_dynamic_lot(0.95, 0.5, "XAUUSD", market_data=market_data_vol)
        
        print(f"Result Lot: {lot_vol}")
        if lot_vol == 0.0:
            print("✅ PASSED: VolatilityGuard blocked trade.")
        else:
            print(f"❌ FAILED: VolatilityGuard did not block. Lot: {lot_vol}")


        # ---------------------------------------------------------
        # TEST 2: SpreadFilter
        # ---------------------------------------------------------
        print("\n[TEST 2] SpreadFilter Trigger (Spread 30, Avg 5)")
        market_data_spread = {
            "atr": 100.0,
            "avg_atr": 100.0, 
            "avg_spread": 5.0
        }
        set_spread(30.0) # Ratio 6.0 > 3.0 (Threshold)
        
        # Execute
        lot_spread = await risk_engine.calculate_dynamic_lot(0.95, 0.5, "XAUUSD", market_data=market_data_spread)
        
        print(f"Result Lot: {lot_spread}")
        if lot_spread == 0.0:
            print("✅ PASSED: SpreadFilter blocked trade.")
        else:
            print(f"❌ FAILED: SpreadFilter did not block. Lot: {lot_spread}")


        # ---------------------------------------------------------
        # TEST 3: Circuit Breaker Level 2
        # ---------------------------------------------------------
        print("\n[TEST 3] Circuit Breaker Level 2 (5% Drawdown)")
        set_spread(5.0) # Normal
        market_data_cb = {
            "atr": 100.0,
            "avg_atr": 100.0,
            "avg_spread": 5.0
        }
        
        # Mock 5% DD (Level 2 starts at 4%)
        circuit_breaker.get_drawdown = MagicMock(return_value=asyncio.Future())
        circuit_breaker.get_drawdown.return_value.set_result(5.0)

        # Reset CB
        circuit_breaker.pause_until = 0
        circuit_breaker.current_level = 0
        
        # Pre-trigger to activate "Operational Pause" (Level 2 sets a 1h pause)
        print("  (Pre-triggering Circuit Breaker to activate Operational Pause...)")
        await circuit_breaker.get_lot_multiplier()
        
        # Execute
        lot_cb = await risk_engine.calculate_dynamic_lot(0.95, 0.5, "XAUUSD", market_data=market_data_cb)
        
        print(f"Result Lot: {lot_cb}")
        if lot_cb == 0.0:
            print("✅ PASSED: Circuit Breaker blocked trade (via Operational Pause).")
        else:
            print(f"❌ FAILED: Circuit Breaker did not block. Lot: {lot_cb}")

if __name__ == "__main__":
    try:
        # Windows loop policy fix
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Test crashed: {e}")
        import traceback
        traceback.print_exc()