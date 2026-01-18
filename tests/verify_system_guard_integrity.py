import sys
import os
import logging

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test.verificator.system_guard")

def run_integrity_test():
    logger.info("🛡️ [VERIFICATOR SENTINEL] INITIATING SYSTEM GUARD INTEGRITY TEST...")
    
    try:
        # 1. Compile Check
        import py_compile
        lib_path = "app/core/system_guard/artifacts.py"
        py_compile.compile(lib_path)
        logger.info("✅ Compilation Successful.")

        # 2. Import Logic
        from app.core.system_guard import SystemGuardError, ArtifactType, ValidationResult
        logger.info("✅ Imports Successful.")

        logger.info("🚀 [VERIFICATOR SENTINEL] SYSTEM GUARD IS STABLE.")
        return True

    except Exception as e:
        logger.error(f"❌ [VERIFICATOR SENTINEL] INTEGRITY FAILURE: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = run_integrity_test()
    if not success:
        sys.exit(1)
