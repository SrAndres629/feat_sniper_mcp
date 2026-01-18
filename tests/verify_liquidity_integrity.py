import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test.verificator.liquidity")

def run_integrity_test():
    logger.info("🛡️ [VERIFICATOR SENTINEL] INITIATING LIQUIDITY DETECTOR INTEGRITY TEST...")
    
    try:
        # 1. Compile Check
        import py_compile
        lib_path = "app/skills/liquidity_detector/detector.py"
        py_compile.compile(lib_path)
        logger.info("✅ Compilation Successful.")

        # 2. Import Logic
        from app.skills.liquidity_detector import (
            get_current_kill_zone, 
            detect_liquidity_pools, 
            detect_fvg,
            compute_space_confidence,
            MarketStateTensor
        )
        logger.info("✅ Imports Successful.")

        # 3. Functional Check with Synthetic Data
        # Create dummy data with time index
        times = pd.date_range(start="2026-01-17 12:00:00", periods=100, freq="1min")
        data = {
            "time": times,
            "open": np.random.randn(100) + 2000,
            "high": np.random.randn(100) + 2005,
            "low": np.random.randn(100) + 1995,
            "close": np.random.randn(100) + 2000,
            "volume": np.random.randint(100, 1000, 100)
        }
        df = pd.DataFrame(data)

        # Test Kill Zone (should return string or None)
        kz = get_current_kill_zone()
        logger.info(f"✅ Kill Zone Detection: {kz}")

        # Test FVG Detection
        fvgs = detect_fvg(df)
        logger.info(f"✅ FVG Detection: {len(fvgs)} found")

        # Test Liquidity Pools
        pools = detect_liquidity_pools(df)
        logger.info(f"✅ Liquidity Pools: {pools['total_pools']} detected")

        # Test Confidence Computation
        confidence = compute_space_confidence(df, 2000.5)
        logger.info(f"✅ Space Confidence: {confidence.overall_space_score:.2f}")

        # Test Tensor Builder
        tensor_builder = MarketStateTensor()
        # Mock MTF call (using same df for all layers for simplicity)
        tensor = tensor_builder.build_tensor(df, df, df, df, df)
        logger.info(f"✅ Market State Tensor: Score={tensor['alignment_score']}")

        logger.info("🚀 [VERIFICATOR SENTINEL] LIQUIDITY DETECTOR IS STABLE.")
        return True

    except Exception as e:
        logger.error(f"❌ [VERIFICATOR SENTINEL] INTEGRITY FAILURE: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = run_integrity_test()
    if not success:
        sys.exit(1)
