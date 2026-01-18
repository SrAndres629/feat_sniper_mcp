import sys
import os
import pandas as pd
import numpy as np
import logging

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test.verification")

def run_smoke_test():
    logger.info("🧪 INITIATING STRUCTURE ENGINE SMOKE TEST...")
    
    try:
        # 1. Test Import via __init__.py (Backward Compatibility)
        logger.info("Step 1: Testing top-level import...")
        from nexus_core.structure_engine import structure_engine, StructureEngine
        logger.info("✅ Import Successful.")

        # 2. Test Instantiation
        logger.info("Step 2: Testing instantiation...")
        engine = StructureEngine()
        logger.info("✅ Instantiation Successful.")

        # 3. Test Functionality with Dummy Data
        logger.info("Step 3: Testing logic execution...")
        data = {
            "open": np.random.randn(100) + 2000,
            "high": np.random.randn(100) + 2005,
            "low": np.random.randn(100) + 1995,
            "close": np.random.randn(100) + 2000,
            "tick_volume": np.random.randint(100, 1000, 100)
        }
        df = pd.DataFrame(data)
        
        # Test individual components via the engine
        df = engine.identify_fractals(df)
        logger.info(f"✅ Fractals: {df['fractal_high'].iloc[-10:].values}")
        
        df = engine.detect_structural_shifts(df)
        logger.info("✅ Structural Shifts processed.")
        
        # This will trigger relative imports inside the package
        health = engine.get_structural_health(df)
        logger.info(f"✅ Structural Health: {health}")
        
        report = engine.get_structural_report(df)
        logger.info(f"✅ Structural Report Generated: {report['mae_pattern']['status']}")

        logger.info("🚀 SMOKE TEST PASSED: GREEN LIGHT ENFORCED.")
        return True

    except Exception as e:
        logger.error(f"❌ SMOKE TEST FAILED: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = run_smoke_test()
    if not success:
        sys.exit(1)
