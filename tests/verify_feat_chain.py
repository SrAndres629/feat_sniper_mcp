import sys
import os
import logging
import pandas as pd
import numpy as np

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test.feat_chain")

async def run_smoke_test():
    logger.info("🧪 INITIATING FEAT CHAIN SMOKE TEST...")
    
    try:
        # 1. Test Import via __init__.py
        logger.info("Step 1: Testing top-level import...")
        from app.skills.feat_chain import feat_full_chain_institucional, FEATChain, FEATDecision
        logger.info("✅ Import Successful.")

        # 2. Test Instantiation
        logger.info("Step 2: Testing instantiation...")
        chain = FEATChain()
        logger.info("✅ Instantiation Successful.")

        # 3. Test Probabilistic Analysis (Dummy)
        logger.info("Step 3: Testing probabilistic logic...")
        market_data = {"symbol": "XAUUSD", "bid": 2000.5, "ask": 2000.7}
        
        # Create dummy candles
        data = {
            "open": np.random.randn(50) + 2000,
            "high": np.random.randn(50) + 2005,
            "low": np.random.randn(50) + 1995,
            "close": np.random.randn(50) + 2000,
        }
        df = pd.DataFrame(data)
        
        decision = await chain.analyze_probabilistic(market_data, df)
        logger.info(f"✅ Probabilistic Result: {decision.action} (Score: {decision.composite_score:.2f})")
        
        # 4. Test Chain Execution
        logger.info("Step 4: Testing chain validation...")
        # Note: This might fail if market_physics is not initialized or has no window
        # But we want to check if the calls flow through the rules without ImportError
        res = await chain.analyze(market_data, 2000.5)
        logger.info(f"✅ Chain Analysis flow complete (Result: {res})")

        logger.info("🚀 FEAT CHAIN SMOKE TEST PASSED: GREEN LIGHT.")
        return True

    except Exception as e:
        logger.error(f"❌ FEAT CHAIN SMOKE TEST FAILED: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(run_smoke_test())
    if not success:
        sys.exit(1)
