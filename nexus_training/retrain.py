import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Fix path
sys.path.append(os.getcwd())

from app.ml.models.hybrid_probabilistic import HybridProbabilistic
from app.ml.feat_processor import feat_processor
from app.core.config import settings

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FEAT.Retrain")

DATA_DIR = "data"
MODELS_DIR = "models"
REPLAY_FILE = os.path.join(DATA_DIR, "experience_replay.jsonl")

# Constants
SEQ_LEN = 32 # Match model config
FEATURE_NAMES = [
    "close", "volume", "rsi", "atr", "feat_structure_score", 
    "mtf_composite_score", "kalman_score", "energy_z_score",
    # Level 45 PVP Integration
    "dist_poc_norm", "pos_in_va", "density_zone", "energy_score",
    "poc_price", "vah_price", "val_price"
] # Must match training features

def load_replay_data():
    """Loads and filters winning trades from experience buffer."""
    if not os.path.exists(REPLAY_FILE):
        logger.warning(f"No replay buffer found at {REPLAY_FILE}")
        return []
    
    winners = []
    with open(REPLAY_FILE, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("outcome") == "WIN":
                    winners.append(record)
            except:
                continue
    
    logger.info(f"Loaded {len(winners)} winning experiences.")
    return winners

def prepare_tensors(experiences, symbol="XAUUSD"):
    """
    Converts win records into Training Tensors (X, y).
    Note: Reconstructing sequences from single snapshots is imperfect.
    For V1 RLAIF, we repeat the snapshot or zero-pad to fit the model shape.
    Ideal V2: Save full sequence in replay buffer.
    """
    X_list = []
    y_list = []
    
    for exp in experiences:
        if exp.get("symbol") != symbol:
            continue
            
        context = exp.get("raw_context", {})
        
        # 1. Feature Extraction (Robust)
        # We try to extract features from the saved context dict
        feature_vector = []
        for name in FEATURE_NAMES:
            val = context.get(name, 0.0)
            try:
                feature_vector.append(float(val))
            except:
                feature_vector.append(0.0)
                
        # 2. Sequence Construction (Snapshot Repeat)
        # We interpret the snapshot as the "current state" of a stationary sequence
        seq = np.tile(feature_vector, (SEQ_LEN, 1)) # (SEQ_LEN, n_feats)
        
        X_list.append(seq)
        
        # 3. Label (Imitation Learning)
        # If it was a WIN, we want to predict the direction of the trade
        # Assume context or trade log implies direction. 
        # For now, we infer from simple logic or assume BUY if not specified (Need improvement)
        # Let's check 'outcome'. If we bought and won, label=BUY(2).
        # We need trade direction. 'is_buy' is not explicitly in 'raw_context' usually.
        # But 'regime.trend' might be.
        trend = context.get("trend", "NEUTRAL")
        if trend == "BULLISH": label = 2 # BUY
        elif trend == "BEARISH": label = 0 # SELL
        else: label = 1 # HOLD
        
        y_list.append(label)

    if not X_list:
        return None, None

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.long)
    return X, y

def retrain_model(symbol="XAUUSD"):
    """
    Fine-tunes the model on winning experiences.
    """
    logger.info(f"Starting RLAIF Retraining for {symbol}...")
    
    # 1. Load Data
    winners = load_replay_data()
    if len(winners) < 10:
        logger.info(f"Not enough data to retrain (Found {len(winners)}, Need 10+). Skipping.")
        return
        
    X, y = prepare_tensors(winners, symbol)
    if X is None:
        logger.warning("Failed to prepare tensors.")
        return
        
    # 2. Load Model
    model_path = os.path.join(MODELS_DIR, f"hybrid_{symbol}_v1.pt")
    input_dim = len(FEATURE_NAMES)
    
    model = HybridProbabilistic(input_dim=input_dim)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        logger.info(f"Loaded existing weights from {model_path}")
    else:
        logger.warning("No base model found. Initializing from scratch (Risky).")

    # 3. Fine-Tuning Setup
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Low LR for fine-tuning
    criterion = nn.CrossEntropyLoss()
    
    X = X.to(device)
    y = y.to(device)
    
    model.train()
    epochs = 5
    
    logger.info(f"Fine-tuning on {X.shape[0]} samples for {epochs} epochs...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Note: feat_input is None for now (V1 RLAIF doesn't support PVP latent replay yet)
        outputs = model(X, force_dropout=True) # Train with dropout!
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 1 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
            
    # 4. Save Updated Model
    new_version_path = os.path.join(MODELS_DIR, f"hybrid_{symbol}_v2.pt")
    
    torch.save({
        "state_dict": model.state_dict(),
        "timestamp": datetime.now().isoformat(),
        "samples_seen": len(winners)
    }, new_version_path)
    
    logger.info(f"✅ Model upgraded and saved to {new_version_path}")

if __name__ == "__main__":
    retrain_model("XAUUSD")
