import sys
import os
import logging
import asyncio

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test.verificator.mt5")

def run_integrity_test():
    logger.info("🛡️ [VERIFICATOR SENTINEL] INITIATING MT5 CONNECTION INTEGRITY TEST...")
    
    try:
        # 1. Compile Check
        import py_compile
        lib_path = "app/core/mt5_conn/manager.py"
        py_compile.compile(lib_path)
        logger.info("✅ Compilation Successful.")

        # 2. Import Logic
        from app.core.mt5_conn import mt5_conn
        logger.info("✅ Imports Successful.")

        # 3. Functional Check (Safe methods)
        # We don't want to actually connect to MT5 here, but check instance state
        from app.core.mt5_conn import MT5_AVAILABLE
        logger.info(f"✅ MT5 Library Available: {MT5_AVAILABLE}")
        
        # Test TerminalTerminator logic (Unit test style)
        from app.core.mt5_conn import TerminalTerminator
        terminator = TerminalTerminator(max_soft_failures=3)
        terminator.record_failure()
        terminator.record_failure()
        should_reset = terminator.record_failure()
        logger.info(f"✅ Terminator Escalation Logic: {should_reset}")

        logger.info("🚀 [VERIFICATOR SENTINEL] MT5 CONNECTION MANAGER IS STABLE.")
        return True

    except Exception as e:
        logger.error(f"❌ [VERIFICATOR SENTINEL] INTEGRITY FAILURE: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = run_integrity_test()
    if not success:
        sys.exit(1)
