import sys
import os
import logging
import asyncio

# Add root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test.verificator.neural")

async def run_neural_integrity_test():
    logger.info("🧠 [NEURAL SENTINEL] INITIATING ML ENGINE INTEGRITY TEST...")
    
    try:
        # 1. Package Import
        from app.ml.ml_engine import ml_engine, MLEngine
        logger.info("✅ Import Successful: app.ml.ml_engine")
        
        # 2. Submodule Check
        from app.ml.ml_engine.inference import InferenceEngine
        from app.ml.ml_engine.loader import ModelLoader
        from app.ml.ml_engine.fractal import HurstBuffer
        logger.info("✅ Submodules Loaded: inference, loader, fractal")
        
        # 3. Instance Check
        assert isinstance(ml_engine, MLEngine)
        logger.info(f"✅ Singleton Active: {ml_engine}")
        
        # 4. Functional Mock Test (Neutral State)
        res = await ml_engine.predict_async("BTCUSD", {"close": 50000, "volume": 100})
        logger.info(f"✅ Prediction Result: {res.get('prediction')} (Uncertainty: {res.get('uncertainty')})")
        
        # 5. RLAIF Buffer Check
        from app.ml.ml_engine.rlaif import ExperienceReplay
        replay = ExperienceReplay()
        logger.info(f"✅ RLAIF Buffer Path: {os.path.join(replay.data_dir, 'experience_replay.jsonl')}")
        
        logger.info("🚀 [NEURAL SENTINEL] PROBABILISTIC ENGINE IS ONLINE.")
        return True

    except Exception as e:
        logger.error(f"❌ [NEURAL SENTINEL] INTEGRITY FAILURE: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = asyncio.run(run_neural_integrity_test())
    if not success:
        sys.exit(1)
