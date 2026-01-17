import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ml.models.anomaly import anomaly_detector
from app.core.system_guard import system_sentinel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | TEST | %(message)s")
logger = logging.getLogger("FEAT.Verify")

def test_immune_system():
    logger.info("🛡️ STARTING IMMUNE SYSTEM VERIFICATION 🛡️")
    
    # 1. Train on Normal Data
    logger.info("1. Training Immune System on 'Normal' Market Data...")
    normal_data = np.random.normal(loc=[1800, 1000, 0.002, 50], scale=[10, 200, 0.001, 10], size=(100, 4))
    anomaly_detector.train(normal_data)
    
    if not anomaly_detector.is_trained:
        logger.error("❌ Stats: Model failed to train.")
        return

    # 2. Check Normal Data
    logger.info("2. Testing Normal Input...")
    normal_input = {"close": 1805.0, "volume": 1050, "atr": 0.0021, "rsi": 55}
    is_anomaly = anomaly_detector.is_anomaly(normal_input)
    if is_anomaly:
        logger.error("❌ False Positive detected on normal data!")
    else:
        logger.info("✅ Normal data accepted.")

    # 3. Simulate Toxic Data Attack (Flash Crash / API Corruption)
    logger.info("3. Injecting TOXIC DATA (Flash Crash)...")
    toxic_input = {"close": 99999.0, "volume": 0, "atr": 500.0, "rsi": 0} # Impossible values
    
    is_toxic = anomaly_detector.is_anomaly(toxic_input)
    
    if is_toxic:
        logger.info("✅ Immune System DETECTED the anomaly.")
        # Simulate ML Engine triggering the switch
        system_sentinel.trigger_kill_switch("TEST_ANOMALY")
    else:
        logger.error("❌ Failed to detect Toxic Data!")
        
    # 4. Check Sentinel State
    logger.info("4. Checking System Sentinel State...")
    if not system_sentinel.is_safe():
        logger.info("✅ KILL SWITCH ACTIVE. Trading Halted.")
        logger.info(f"   Reason: {system_sentinel.kill_reason}")
        print("SUCCESS")
    else:
        logger.error("❌ Kill Switch failed to activate!")

if __name__ == "__main__":
    test_immune_system()
