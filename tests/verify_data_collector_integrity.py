import sys
import os
import logging
import asyncio

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test.verificator.data_collector")

def run_integrity_test():
    logger.info("🛡️ [VERIFICATOR SENTINEL] INITIATING DATA COLLECTOR INTEGRITY TEST...")
    
    try:
        # 1. Compile Check
        import py_compile
        lib_path = "app/ml/data_collector/engine.py"
        py_compile.compile(lib_path)
        logger.info("✅ Compilation Successful.")

        # 2. Import Logic
        from app.ml.data_collector import data_collector, collect_sample, SystemState
        logger.info("✅ Imports Successful.")
        
        # 3. Basic functionality check (dry run)
        sample_candle = {"close": 2000.0, "open": 1999.0, "high": 2001.0, "low": 1998.0, "volume": 100, "time": "2024-01-01T00:00:00Z"}
        sample_inds = {"rsi": 55.0, "atr": 2.5}
        
        # We won't actually hit the DB in this quick check if it's not initialized, 
        # but let's check if the object exists
        assert data_collector is not None
        assert data_collector.state == SystemState.BOOTING
        
        logger.info("🚀 [VERIFICATOR SENTINEL] DATA COLLECTOR IS STABLE.")
        return True

    except Exception as e:
        logger.error(f"❌ [VERIFICATOR SENTINEL] INTEGRITY FAILURE: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = run_integrity_test()
    if not success:
        sys.exit(1)
