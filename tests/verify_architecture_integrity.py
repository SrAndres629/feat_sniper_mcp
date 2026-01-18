import sys
import os
import logging
import asyncio

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test.verificator.master")

async def run_master_integrity_test():
    logger.info("🛡️ [VERIFICATOR SENTINEL] INITIATING MASTER ARCHITECTURE INTEGRITY TEST...")
    
    failures = []
    
    # 1. Config Package
    try:
        from app.core.config import settings
        logger.info(f"✅ Config: Loaded (Symbol: {settings.SYMBOL})")
    except Exception as e:
        failures.append(f"Config: {e}")

    # 2. Risk Package
    try:
        from app.services.risk import risk_engine, calculate_damped_kelly
        logger.info(f"✅ Risk: Loaded (Kelly: {calculate_damped_kelly(0.6, 0.05):.2f})")
    except Exception as e:
        failures.append(f"Risk: {e}")

    # 3. System Guard Package
    try:
        from app.core.system_guard import system_sentinel, validate_model_artifacts
        logger.info("✅ System Guard: Loaded")
    except Exception as e:
        failures.append(f"System Guard: {e}")
        
    # 4. Data Collector Package
    try:
        from app.ml.data_collector import data_collector, SystemState
        logger.info(f"✅ Data Collector: Loaded (State: {data_collector.state})")
    except Exception as e:
        failures.append(f"Data Collector: {e}")

    # 5. Feat Processor Package
    try:
        from app.ml.feat_processor import feat_processor, FEATURE_NAMES
        logger.info(f"✅ Feat Processor: Loaded ({len(FEATURE_NAMES)} features)")
    except Exception as e:
        failures.append(f"Feat Processor: {e}")

    # 6. Lifecycle Package
    try:
        from app.core.lifecycle import lifecycle_manager, ProcessState
        logger.info("✅ Lifecycle: Loaded")
    except Exception as e:
        failures.append(f"Lifecycle: {e}")

    # 7. Health Package
    try:
        from app.core.health import health_sentinel
        logger.info("✅ Health: Loaded")
    except Exception as e:
        failures.append(f"Health: {e}")

    if failures:
        logger.error("❌ MASTER INTEGRITY FAILURE:")
        for f in failures:
            logger.error(f"  - {f}")
        return False
    else:
        logger.info("🚀 [VERIFICATOR SENTINEL] MASTER ARCHITECTURE IS STABLE.")
        return True

if __name__ == "__main__":
    success = asyncio.run(run_master_integrity_test())
    if not success:
        sys.exit(1)
