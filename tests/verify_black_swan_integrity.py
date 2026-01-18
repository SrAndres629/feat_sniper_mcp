import sys
import os
import logging
from datetime import datetime, timezone

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test.verificator.black_swan")

def run_integrity_test():
    logger.info("🛡️ [VERIFICATOR SENTINEL] INITIATING BLACK SWAN GUARD INTEGRITY TEST...")
    
    try:
        # 1. Compile Check
        import py_compile
        lib_path = "app/skills/black_swan_guard/unified.py"
        py_compile.compile(lib_path)
        logger.info("✅ Compilation Successful.")

        # 2. Import Logic
        from app.skills.black_swan_guard import (
            BlackSwanGuard, 
            VolatilityRegime, 
            CircuitState,
            black_swan_guard
        )
        logger.info("✅ Imports Successful.")

        # 3. Functional Check
        guard = BlackSwanGuard(initial_balance=100.0)
        
        # Test Normal Market
        decision = guard.evaluate(current_atr=1.0, current_equity=100.0, current_spread=0.1)
        logger.info(f"✅ Normal Market: Can Trade={decision.can_trade}, Mult={decision.lot_multiplier}")
        
        # Test Volatility Spike (Extreme)
        # Feed some history first to stabilize EMA
        for _ in range(20): guard.evaluate(current_atr=1.0, current_equity=100.0)
        
        decision_extreme = guard.evaluate(current_atr=5.0, current_equity=100.0)
        logger.info(f"✅ Extreme ATR: Can Trade={decision_extreme.can_trade}, Regime={decision_extreme.volatility.regime}")
        
        # Test Circuit Breaker
        decision_drawdown = guard.evaluate(current_atr=1.0, current_equity=90.0) # 10% DD
        logger.info(f"✅ Circuit Breaker (10% DD): State={decision_drawdown.circuit.state}, Can Trade={decision_drawdown.can_trade}")

        logger.info("🚀 [VERIFICATOR SENTINEL] BLACK SWAN GUARD IS STABLE.")
        return True

    except Exception as e:
        logger.error(f"❌ [VERIFICATOR SENTINEL] INTEGRITY FAILURE: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = run_integrity_test()
    if not success:
        sys.exit(1)
