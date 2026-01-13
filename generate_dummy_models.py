
import os
import sys
import numpy as np
import pandas as pd
import logging

# Setup paths
sys.path.append(os.getcwd())
from app.ml.train_models import train_gbm, train_lstm, FEATURE_NAMES, SYMBOL

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DummyModelGenerator")

def generate_and_train():
    print(f"Generating dummy data for {SYMBOL} with {len(FEATURE_NAMES)} features...")
    
    # Generate 1000 random samples
    n_samples = 1000
    X = np.random.rand(n_samples, len(FEATURE_NAMES)).astype(np.float32)
    y = np.random.randint(0, 2, n_samples).astype(np.int64)
    
    # Train GBM
    print("Training GBM...")
    gbm_path = train_gbm(X, y)
    print(f"✅ GBM saved to {gbm_path}")
    
    # Train LSTM
    print("Training LSTM...")
    lstm_path = train_lstm(X, y)
    print(f"✅ LSTM saved to {lstm_path}")
    
    print("\nSUCCESS: Models updated to current schema.")

if __name__ == "__main__":
    generate_and_train()
