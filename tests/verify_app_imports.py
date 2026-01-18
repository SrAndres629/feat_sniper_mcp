import sys
import os
import logging
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test.verificator.imports")

def run_import_test():
    logger.info("🛡️ [VERIFICATOR SENTINEL] INITIATING APP IMPORT INTEGRITY TEST...")
    
    failures = []
    
    # 1. Main App
    try:
        import app.main
        logger.info("✅ app.main: Loaded")
    except Exception as e:
        failures.append(f"app.main: {e}")

    # 2. Execution Engine
    try:
        import app.skills.execution.engine
        logger.info("✅ app.skills.execution.engine: Loaded")
    except Exception as e:
        failures.append(f"app.skills.execution.engine: {e}")
        
    # 3. Twin Execution
    try:
        import app.skills.execution.twin
        logger.info("✅ app.skills.execution.twin: Loaded")
    except Exception as e:
        failures.append(f"app.skills.execution.twin: {e}")

    # 4. Nexus Engine
    try:
        import app.core.nexus_engine
        logger.info("✅ app.core.nexus_engine: Loaded")
    except Exception as e:
        failures.append(f"app.core.nexus_engine: {e}")

    # 5. Risk Package (Explicit Check)
    try:
        from app.services.risk import risk_engine
        logger.info("✅ app.services.risk: Loaded")
    except Exception as e:
        failures.append(f"app.services.risk: {e}")

    if failures:
        logger.error("❌ IMPORT INTEGRITY FAILURE:")
        for f in failures:
            logger.error(f"  - {f}")
        return False
    else:
        logger.info("🚀 [VERIFICATOR SENTINEL] APP IMPORTS ARE STABLE.")
        return True

if __name__ == "__main__":
    if not run_import_test():
        sys.exit(1)
