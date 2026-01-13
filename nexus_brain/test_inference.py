
import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import logging
from datetime import datetime

# Setup paths
sys.path.append(os.getcwd())
try:
    from app.core.config import settings
    from app.skills.indicators import calculate_feat_layers
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuralValidator")

def run_test():
    print("\n🔮 NEURAL ARCHITECT: VALIDATION GATE")
    print("======================================")
    
    # 1. Generate Dummy OHLC
    print("[1/3] Generating Synthetic Market Data...")
    dates = pd.date_range(end=datetime.now(), periods=5000, freq='1min')
    df = pd.DataFrame({
        'close': np.random.normal(2000, 10, 5000).cumsum().astype(float),
        'tick_time': dates
    })
    df['open'] = df['close'] + np.random.normal(0, 1, 5000)
    df['high'] = df['close'] + 5
    df['low'] = df['close'] - 5
    
    # 2. Run Physics Engine
    print("[2/3] Computing Multifractal Physics Layers...")
    try:
        df_feat = calculate_feat_layers(df)
        print(f"   ✅ Computed {len(df_feat.columns)} columns.")
        
        # Verify required features
        req_cols = [
            'L1_Mean', 'L1_Width', 'L4_Slope', 'Div_L1_L2'
        ]
        
        missing = [c for c in req_cols if c not in df_feat.columns]
        if missing:
            print(f"   ❌ Missing Columns: {missing}")
            return False
            
        print("   ✅ All Tensor Layers Present.")
        
    except Exception as e:
        print(f"   ❌ Physics Calculation Failed: {e}")
        return False

    # 3. Load Models & Infer
    print("[3/3] Testing Inference (Forward Pass)...")
    symbol = settings.SYMBOL
    
    # Select just the last row as sample
    sample_df = df_feat.iloc[-1:][req_cols]
    X_sample = sample_df.values.astype(np.float32)
    
    # --- GBM Test ---
    gbm_path = f"models/gbm_{symbol}_v2.joblib"
    if os.path.exists(gbm_path):
        try:
            data = joblib.load(gbm_path)
            model = data["model"]
            scaler = data["scaler"]
            
            X_scaled = scaler.transform(X_sample)
            prob = model.predict_proba(X_scaled)[0][1]
            print(f"   ✅ GBM: Inference OK (p={prob:.4f})")
        except Exception as e:
            print(f"   ❌ GBM Error: {e}")
    else:
        print(f"   ⚠️ GBM V2 Path Not Found: {gbm_path}")

    # --- LSTM Test ---
    lstm_path = f"models/lstm_{symbol}_v2.pt"
    if os.path.exists(lstm_path):
        try:
            from app.ml.train_models import LSTMWithAttention
            checkpoint = torch.load(lstm_path)
            config = checkpoint["model_config"]
            
            print(f"   🔍 LSTM Config: Input Dim = {config['input_dim']}")
            
            model = LSTMWithAttention(
                input_dim=config["input_dim"],
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"]
            )
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            
            # Sequence simulation (Need 3D tensor)
            seq_len = config["seq_len"]
            # Create dummy sequence
            X_seq = np.random.rand(1, seq_len, config["input_dim"]).astype(np.float32)
            X_tensor = torch.tensor(X_seq)
            
            with torch.no_grad():
                logits = model(X_tensor)
                probs = torch.softmax(logits, dim=1)
                p_win = probs[0][1].item()
                
            print(f"   ✅ LSTM: TENSOR SHAPE OK: {X_tensor.shape} -> Output p={p_win:.4f}")
            
        except Exception as e:
            print(f"   ❌ LSTM Error: {e}")
    else:
        print(f"   ⚠️ LSTM V2 Path Not Found: {lstm_path}")

    return True

if __name__ == "__main__":
    run_test()
