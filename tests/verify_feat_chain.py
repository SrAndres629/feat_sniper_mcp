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
    logger.info("🧪 INITIATING NEURO-FUNCTIONAL FEAT CHAIN SMOKE TEST...")
    
    try:
        # 1. Test Import via __init__.py
        logger.info("Step 1: Testing neuro-channel imports...")
        from app.skills.feat_chain import feat_full_chain_institucional, FEATChain, LiquidityChannel, KineticsChannel, VolatilityChannel
        logger.info("✅ Neuro-Channels Loaded.")

        # 2. Test Instantiation
        logger.info("Step 2: Testing orchestrator instantiation...")
        chain = FEATChain()
        logger.info("✅ Orchestrator Initialized.")

        # 3. Test Probabilistic Analysis with Synthetic Scalping Data
        logger.info("Step 3: Testing probabilistic analysis (M1 simulation)...")
        market_data = {"symbol": "XAUUSD", "bid": 2000.5, "ask": 2000.7}
        
        # Create dummy candles with a clear trend to trigger logic
        trend = np.linspace(2000, 2010, 50)
        data = {
            "open": trend - 1,
            "high": trend + 2,
            "low": trend - 2,
            "close": trend,
        }
        df = pd.DataFrame(data)
        
        decision = await chain.analyze_probabilistic(market_data, df)
        logger.info(f"✅ Neural Output: {decision.action} (Composite: {decision.composite_score:.2f})")
        logger.info(f"✅ Reasoning: {decision.reasoning}")
        
        if decision.action != "HOLD":
            logger.info("🚀 Channels aligned successfully on trending data.")

        logger.info("🚀 NEURO-FUNCTIONAL SMOKE TEST PASSED.")
        return True

    except Exception as e:
        logger.error(f"❌ NEURO-FUNCTIONAL SMOKE TEST FAILED: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(run_smoke_test())
    if not success:
        sys.exit(1)
