import sys
import os
import logging

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test.verificator.config")

def run_integrity_test():
    logger.info("🛡️ [VERIFICATOR SENTINEL] INITIATING CONFIG INTEGRITY TEST...")
    
    try:
        # 1. Import Logic
        from app.core.config import settings, ExecutionMode
        logger.info("✅ Imports Successful.")
        
        # 2. Field check
        logger.info(f"✅ SYMBOL: {settings.SYMBOL}")
        logger.info(f"✅ Execution Mode: {settings.execution_mode}")
        
        assert settings.SYMBOL is not None
        assert isinstance(settings.LAYER_MICRO_PERIODS, tuple)
        
        logger.info("🚀 [VERIFICATOR SENTINEL] CONFIG IS STABLE.")
        return True

    except Exception as e:
        logger.error(f"❌ [VERIFICATOR SENTINEL] INTEGRITY FAILURE: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = run_integrity_test()
    if not success:
        sys.exit(1)
