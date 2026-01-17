import asyncio
import os
import json
import torch
import shutil
import logging
import sys

# Add working dir to path
sys.path.append(os.getcwd())

from nexus_training.retrain import retrain_model, REPLAY_FILE, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyRLAIF")

def create_mock_data():
    """Generates synthetic winning trades."""
    os.makedirs("data", exist_ok=True)
    
    # Mock Feature Context
    # We need features that match FEATURE_NAMES in retrain.py
    # ["close", "volume", "rsi", "atr", "feat_structure_score", "mtf_composite_score", "pvp_energy", "kalman_score", "energy_z_score"]
    
    mock_context = {
        "close": 2050.0, "volume": 1000, "rsi": 65.0, "atr": 2.5,
        "feat_structure_score": 0.8, "mtf_composite_score": 0.7,
        "pvp_energy": 0.9, "kalman_score": 0.1, "energy_z_score": 1.5,
        "trend": "BULLISH" # Should map to Label 2 (BUY)
    }
    
    with open(REPLAY_FILE, "w") as f:
        for i in range(20):
            entry = {
                "ticket": 1000+i,
                "outcome": "WIN",
                "profit": 100.0,
                "symbol": "XAUUSD",
                "raw_context": mock_context
            }
            f.write(json.dumps(entry) + "\n")
            
    logger.info(f"Generated 20 synthetic WIN records at {REPLAY_FILE}")

def run_verification():
    print("\n--- [LEVEL 43] RLAIF Retraining Verification ---\n")
    
    # 1. Create Mock Data
    create_mock_data()
    
    # 2. Ensure Models Dir
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 3. Create Dummy Base Model (to avoid restart)
    base_model_path = os.path.join(MODELS_DIR, "hybrid_XAUUSD_v1.pt")
    # We save a dummy state dict to simulate existence
    # Note: retrain.py handles missing model by initializing scratch, which is fine too.
    # But let's let it init from scratch for testing simplicity.
    if os.path.exists(base_model_path):
        os.remove(base_model_path)
        
    # 4. Run Retraining
    try:
        retrain_model("XAUUSD")
        
        # 5. Check Result
        v2_path = os.path.join(MODELS_DIR, "hybrid_XAUUSD_v2.pt")
        if os.path.exists(v2_path):
            print("\n✅ SUCCESS: RLAIF Loop generated 'hybrid_XAUUSD_v2.pt'")
            # Inspect metadata
            ckpt = torch.load(v2_path)
            print(f"   Timestamp: {ckpt.get('timestamp')}")
            print(f"   Samples Seen: {ckpt.get('samples_seen')}")
        else:
            print("\n❌ FAILED: v2 model not found.")
            
    except Exception as e:
        print(f"\n❌ FAILED with Error: {e}")
        import traceback
        traceback.print_exc()

    # Cleanup
    # if os.path.exists(REPLAY_FILE): os.remove(REPLAY_FILE)
    print("\n------------------------------------------------\n")

if __name__ == "__main__":
    run_verification()
