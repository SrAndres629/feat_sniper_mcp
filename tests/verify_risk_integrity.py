import sys
import os
import logging
import asyncio

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test.verificator.risk")

def run_integrity_test():
    logger.info("🛡️ [VERIFICATOR SENTINEL] INITIATING RISK ENGINE INTEGRITY TEST...")
    
    try:
        # 1. Compile Check
        import py_compile
        lib_path = "app/services/risk/engine.py"
        py_compile.compile(lib_path)
        logger.info("✅ Compilation Successful.")

        # 2. Import Logic
        from app.services.risk import risk_engine, calculate_damped_kelly
        logger.info("✅ Imports Successful.")
        
        # 3. Math check
        f = calculate_damped_kelly(0.75, 0.04, 1.5)
        logger.info(f"✅ Kelly Math: {f:.4f}")
        assert f >= 0
        
        logger.info("🚀 [VERIFICATOR SENTINEL] RISK ENGINE IS STABLE.")
        return True

    except Exception as e:
        logger.error(f"❌ [VERIFICATOR SENTINEL] INTEGRITY FAILURE: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = run_integrity_test()
    if not success:
        sys.exit(1)
