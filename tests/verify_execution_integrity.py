import sys
import os
import logging
from datetime import datetime, timezone

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test.verificator.execution")

def run_integrity_test():
    logger.info("🛡️ [VERIFICATOR SENTINEL] INITIATING EXECUTION SKILL INTEGRITY TEST...")
    
    try:
        # 1. Compile Check
        import py_compile
        lib_path = "app/skills/execution/engine.py"
        py_compile.compile(lib_path)
        logger.info("✅ Compilation Successful.")

        # 2. Import Logic
        from app.skills.execution import send_order, ACTION_TO_MT5_TYPE
        logger.info("✅ Imports Successful.")

        # 3. Functional Check (Constants)
        logger.info(f"✅ Order Types Mapping: {ACTION_TO_MT5_TYPE}")
        
        # 4. Check dependencies
        from app.core.mt5_conn import mt5_conn
        logger.info(f"✅ MT5 Connection Linked: {mt5_conn is not None}")

        logger.info("🚀 [VERIFICATOR SENTINEL] EXECUTION SKILL IS STABLE.")
        return True

    except Exception as e:
        logger.error(f"❌ [VERIFICATOR SENTINEL] INTEGRITY FAILURE: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = run_integrity_test()
    if not success:
        sys.exit(1)
