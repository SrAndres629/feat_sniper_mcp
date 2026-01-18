import sys
import os
import logging
import pandas as pd
import numpy as np

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test.verificator.feat_processor")

def run_integrity_test():
    logger.info("🛡️ [VERIFICATOR SENTINEL] INITIATING FEAT PROCESSOR INTEGRITY TEST...")
    
    try:
        # 1. Compile Check
        import py_compile
        lib_path = "app/ml/feat_processor/engine.py"
        py_compile.compile(lib_path)
        logger.info("✅ Compilation Successful.")

        # 2. Import Logic
        from app.ml.feat_processor import feat_processor, FEATURE_NAMES, tensorize_snapshot
        logger.info("✅ Imports Successful.")
        
        # 3. Basic functionality check (dry run)
        df = pd.DataFrame({
            "open": [100.0, 101.0], "high": [102.0, 103.0], 
            "low": [99.0, 100.0], "close": [101.0, 102.0], 
            "volume": [1000, 1100]
        })
        
        # We need a context for some indicators, but the engine should be robust
        assert feat_processor is not None
        assert len(FEATURE_NAMES) > 0
        
        # Test tensorizer
        vec = tensorize_snapshot({"rsi": 70.0, "accel": 1.5}, ["rsi", "accel"])
        assert len(vec) == 2
        assert vec[0] == 0.7
        
        logger.info("🚀 [VERIFICATOR SENTINEL] FEAT PROCESSOR IS STABLE.")
        return True

    except Exception as e:
        logger.error(f"❌ [VERIFICATOR SENTINEL] INTEGRITY FAILURE: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = run_integrity_test()
    if not success:
        sys.exit(1)
