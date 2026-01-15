import asyncio
import sys
# Mock mt5 before any import
from unittest.mock import MagicMock
sys.modules['MetaTrader5'] = MagicMock()

from app.services.risk_engine import RiskEngine

async def debug_main():
    print("DEBUG START")
    engine = RiskEngine()
    
    # Mock methods internally to avoid import confusion
    engine.get_neural_allocation = MagicMock(return_value={"lot_multiplier": 1.0})
    engine.get_adaptive_lots = MagicMock(return_value=0.55)
    
    # Make them async-like (since they are async in real code, mocks usually return value on call unless configured)
    # But wait, in the real code they are async.
    # If I mock them on the instance, I replace the real method.
    
    # Let's verify the REAL method signature behavior (without mocking the methods themselves, just the dependencies)
    from app.services.risk_engine import risk_engine
    
    # We must patch the dependency calls inside
    risk_engine.get_neural_allocation = AsyncMock(return_value={"lot_multiplier": 1.0})
    risk_engine.get_adaptive_lots = AsyncMock(return_value=0.55)
    
    print("Calling calculate_dynamic_lot...")
    result = await risk_engine.calculate_dynamic_lot(0.95, 0.5, "XAUUSD")
    
    print(f"Result: {result}")
    print(f"Type: {type(result)}")
    
    if asyncio.iscoroutine(result):
        print("ALERT: Result is a coroutine!")
        real_val = await result
        print(f"Real Value: {real_val}")

from unittest.mock import AsyncMock
if __name__ == "__main__":
    asyncio.run(debug_main())
