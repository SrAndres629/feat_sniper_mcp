
import os
import sys
import asyncio
import logging
from datetime import datetime

# Setup paths
sys.path.append(os.getcwd())

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SystemVerifier")

async def verify_ml_engine():
    print("\n[1/4] Verifying ML Engine Dependencies...")
    try:
        from app.ml.ml_engine import ml_engine
        print("✅ app.ml.ml_engine imported successfully.")
    except ImportError as e:
        print(f"❌ Failed to import ml_engine: {e}")
        return False

    print("\n[2/4] Verifying Model Assets (XAUUSD)...")
    try:
        # Manually check for files
        models_dir = os.getenv("MODELS_DIR", "models")
        gbm_path = os.path.join(models_dir, "gbm_XAUUSD_v1.joblib")
        lstm_path = os.path.join(models_dir, "lstm_XAUUSD_v1.pt")
        
        has_gbm = os.path.exists(gbm_path)
        has_lstm = os.path.exists(lstm_path)
        
        if has_gbm: print(f"✅ GBM Model found: {gbm_path}")
        else: print(f"⚠️ GBM Model missing (Using Fallback?): {gbm_path}")
        
        if has_lstm: print(f"✅ LSTM Model found: {lstm_path}")
        else: print(f"⚠️ LSTM Model missing (Usually optional): {lstm_path}")
        
    except Exception as e:
        print(f"❌ Error checking models: {e}")

    print("\n[3/4] Testing Inference (Dry Run)...")
    try:
        # Dummy Features
        features = {
            "close": 2030.50, "open": 2030.10, "high": 2031.00, "low": 2029.00, "volume": 1500,
            "rsi": 55.0, "atr": 1.5, "ema_fast": 2030.00, "ema_slow": 2028.00,
            "feat_score": 75.0, "fsm_state": 1, "liquidity_ratio": 1.2, "volatility_zscore": 0.5,
            # Neural Tensors (Zeros)
            "momentum_kinetic_micro": 0.1, "entropy_coefficient": 0.5,
        }
        
        pred = ml_engine.ensemble_predict("XAUUSD", features)
        print(f"✅ Prediction Output: {pred}")
        
        if pred.get("gbm_available"): print("   -> GBM Active")
        if pred.get("lstm_available"): print("   -> LSTM Active")
        
    except Exception as e:
        print(f"❌ Inference Failed: {e}")
        return False

    print("\n[4/4] Verifying Execution Skills...")
    try:
        from app.skills.execution import send_order
        import MetaTrader5 as mt5
        print("✅ Execution Module imported.")
    except ImportError as e:
        print(f"❌ Execution Import Failed: {e}")

    return True

if __name__ == "__main__":
    asyncio.run(verify_ml_engine())
