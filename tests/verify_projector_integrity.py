import sys
import os
import pandas as pd
import numpy as np
import logging

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test.verificator.zone_projector")

def run_integrity_test():
    logger.info("🛡️ [VERIFICATOR SENTINEL] INITIATING ZONE PROJECTOR INTEGRITY TEST...")
    
    try:
        # 1. Compile Check
        import py_compile
        lib_path = "nexus_core/zone_projector/engine.py"
        py_compile.compile(lib_path)
        logger.info("✅ Compilation Successful.")

        # 2. Import Logic
        from nexus_core.zone_projector import zone_projector, ZoneType, VolatilityState
        logger.info("✅ Imports Successful.")

        # 3. Functional Check with Synthetic Data
        data = {
            "open": np.random.randn(100) + 2000,
            "high": np.random.randn(100) + 2005,
            "low": np.random.randn(100) + 1995,
            "close": np.random.randn(100) + 2000,
        }
        df = pd.DataFrame(data)
        
        # Test Action Plan Generation
        plan = zone_projector.generate_action_plan(df, 2000.0)
        logger.info(f"✅ Action Plan Generated: Bias={plan.bias}, Zones={len(plan.all_zones)}")
        
        # Test Volatility State
        v_state, v_factor = zone_projector.get_volatility_state(df)
        logger.info(f"✅ Volatility Detected: {v_state.value} (Factor: {v_factor:.2f})")

        logger.info("🚀 [VERIFICATOR SENTINEL] ZONE PROJECTOR IS STABLE.")
        return True

    except Exception as e:
        logger.error(f"❌ [VERIFICATOR SENTINEL] INTEGRITY FAILURE: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = run_integrity_test()
    if not success:
        sys.exit(1)
