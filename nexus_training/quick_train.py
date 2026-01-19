import sys
import os
import asyncio
import torch
import logging
import pandas as pd
import numpy as np
from typing import Dict

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ml.feat_processor import feat_processor
from app.ml.models.hybrid_probabilistic import HybridProbabilistic
from app.core.config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger("QUICK_TRAIN")

def quick_train_brain(symbol="XAUUSD"):
    logger.info("🧠 STARTING INSTITUTIONAL RE-TRAINING (Dimensions Alignment Fixed) 🧠")
    
    # 1. Generate Realistic Data using BattlefieldSimulator
    from nexus_training.simulate_warfare import BattlefieldSimulator
    sim = BattlefieldSimulator(symbol)
    df_raw = sim.generate_synthetic_data(n_rows=1000) # More data for stability
    
    # 2. Process through FeatProcessor (Actual Logic, NOT Random)
    logger.info("🧪 Feature Engineering in progress...")
    df = feat_processor.process_dataframe(df_raw)
    
    # [FIX] Populate all 18 Neural Dimensions in a clean DF
    logger.info("🎨 Re-mapping latent dimensions for Neural Alignment...")
    latent_rows = []
    for idx, row in df.iterrows():
        latent_rows.append(feat_processor.compute_latent_vector(row))
    
    # Create a fresh DF with only the needed neural features + close (for target labeling)
    new_df = pd.DataFrame(latent_rows)
    new_df['close'] = df['close'].reset_index(drop=True)
    df = new_df.dropna().reset_index(drop=True)
    
    feat_cols = list(settings.NEURAL_FEATURE_NAMES)
    # Ensure all features are numeric and drop any rows that failed engineering
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    
    seq_len = settings.LSTM_SEQ_LEN
    
    logger.info(f"Dataset Size: {len(df)} rows. Features: {len(feat_cols)}")
    
    # 3. Build Tensors (Aligned with InferenceEngine)
    X_list = []
    y_list = []
    feat_inputs_list = {
        "form": [], "space": [], "accel": [], "time": [], "kinetic": []
    }
    
    # Labeling Logic: Future Return in next 5 bars
    for i in range(seq_len, len(df) - 5):
        # Time-series Sequence (LSTM/TCN)
        seq_window = df.iloc[i-seq_len:i][feat_cols].values
        X_list.append(torch.tensor(seq_window, dtype=torch.float32))
        
        # Latest State (Latent Encoder)
        row = df.iloc[i]
        metrics = feat_processor.compute_latent_vector(row)
        
        # Grouping exactly as app/ml/ml_engine/inference.py:45
        feat_inputs_list["form"].append([metrics["skew"], metrics["entropy"], metrics["form"], 0.0])
        feat_inputs_list["space"].append([metrics["dist_poc"], metrics["pos_in_va"], metrics["space"]])
        feat_inputs_list["accel"].append([metrics["energy"], metrics["accel"], metrics["kalman_score"]])
        feat_inputs_list["time"].append([metrics["dist_micro"], metrics["dist_struct"], metrics["dist_macro"], metrics["time"]])
        feat_inputs_list["kinetic"].append([
            metrics["kinetic_pattern_id"], 
            metrics["kinetic_coherence"], 
            metrics["dist_bias"], 
            metrics["layer_alignment"]
        ])
        
        # Target: 0:SELL, 1:HOLD, 2:BUY
        future_ret = (df.iloc[i+5]['close'] - df.iloc[i]['close']) / df.iloc[i]['close']
        if future_ret > 0.0015: label = 2
        elif future_ret < -0.0015: label = 0
        else: label = 1
        y_list.append(label)
        
    X_tensor = torch.stack(X_list)
    Y_tensor = torch.tensor(y_list, dtype=torch.long)
    
    # Stack sub-features
    F_inputs = {k: torch.tensor(v, dtype=torch.float32) for k, v in feat_inputs_list.items()}
    
    # 4. Initialize Model
    model = HybridProbabilistic(
        input_dim=len(feat_cols),
        hidden_dim=settings.NEURAL_HIDDEN_DIM,
        num_classes=3
    )
    
    # 5. Training Loop
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=settings.NEURAL_LEARNING_RATE, 
        weight_decay=settings.NEURAL_WEIGHT_DECAY
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    logger.info("🚀 Initiating Neural Convergence Loop...")
    model.train()
    
    batch_size = settings.NEURAL_BATCH_SIZE
    epochs = settings.NEURAL_EPOCHS
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            batch_Y = Y_tensor[i:i+batch_size]
            batch_F = {k: v[i:i+batch_size] for k, v in F_inputs.items()}
            
            optimizer.zero_grad()
            out = model(batch_X, feat_input=batch_F)
            
            # Loss Components
            loss_dir = criterion(out["logits"], batch_Y)
            # Confidence calibration: Should target 1.0 for the correct class, 0.5 otherwise?
            # Simpler: MSE between p_win and 1.0 for trades, 0.5 for holds
            target_p = torch.where(batch_Y != 1, 0.8, 0.5) # Soft targets
            loss_conf = torch.mean((out["p_win"].squeeze() - target_p)**2)
            
            loss = loss_dir + 0.5 * loss_conf
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss/(len(X_tensor)/batch_size):.4f}")

    # 6. Weight Persistence
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    path = os.path.join(settings.MODELS_DIR, f"hybrid_prob_{symbol}_v2.pt")
    
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "seq_len": seq_len,
            "input_dim": len(feat_cols),
            "hidden_dim": settings.NEURAL_HIDDEN_DIM
        },
        "version": "4.1"
    }, path)
    
    logger.info(f"✅ MISSION ACCOMPLISHED: Model Weight Persistent at {path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--auto", action="store_true", help="Auto mode for AutoML integration")
    args = parser.parse_args()
    
    quick_train_brain(args.symbol)
