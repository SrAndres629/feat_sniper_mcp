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
    
    # [v5.0 - DOCTORAL] Triple Barrier Labeling Logic
    # 0: SELL (Hits SL first or extreme down), 1: HOLD (Horizontal/No hit), 2: BUY (Hits TP first)
    horizon = 20 # Max bars to wait
    tp_mult = 1.5 
    sl_mult = 1.0
    
    for i in range(seq_len, len(df) - horizon):
        # Build Sequence
        seq_window = df.iloc[i-seq_len:i][feat_cols].values
        X_list.append(torch.tensor(seq_window, dtype=torch.float32))
        
        # Latent Metrics
        row = df.iloc[i]
        metrics = feat_processor.compute_latent_vector(row)
        feat_inputs_list["form"].append([metrics["skew"], metrics["entropy"], metrics["form"], 0.0])
        feat_inputs_list["space"].append([metrics["dist_poc"], metrics["pos_in_va"], metrics["space"]])
        feat_inputs_list["accel"].append([metrics["energy"], metrics.get("accel", 0.0), metrics.get("kalman_score", 0.0)])
        feat_inputs_list["time"].append([metrics.get("dist_micro", 0.0), metrics.get("dist_struct", 0.0), metrics.get("dist_macro", 0.0), metrics["time"]])
        feat_inputs_list["kinetic"].append([
            metrics["kinetic_pattern_id"], 
            metrics["kinetic_coherence"], 
            metrics["dist_bias"], 
            metrics["layer_alignment"]
        ])
        
        # TRIPLE BARRIER LOGIC
        price_start = df.iloc[i]['close']
        atr = df.iloc[i].get('atr_accel', price_start * 0.001)
        up_barrier = price_start + (atr * tp_mult)
        dn_barrier = price_start - (atr * sl_mult)
        
        label = 1 # Default HOLD
        for j in range(1, horizon):
            curr_p = df.iloc[i+j]['close']
            if curr_p >= up_barrier:
                label = 2
                break
            elif curr_p <= dn_barrier:
                label = 0
                break
        y_list.append(label)
        
    X_tensor = torch.stack(X_list)
    Y_tensor = torch.tensor(y_list, dtype=torch.long)
    F_inputs = {k: torch.tensor(v, dtype=torch.float32) for k, v in feat_inputs_list.items()}
    
    # 4. Initialize Model v5.0
    model = HybridProbabilistic(
        input_dim=len(feat_cols),
        hidden_dim=settings.NEURAL_HIDDEN_DIM,
        num_classes=3
    )
    
    # 5. Training Loop with Uncertainty Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss(reduction='none') # For weighting
    
    logger.info("🚀 Initiating Bayesian Neural Convergence Loop...")
    model.train()
    batch_size = 64
    epochs = 20
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            batch_Y = Y_tensor[i:i+batch_size]
            batch_F = {k: v[i:i+batch_size] for k, v in F_inputs.items()}
            
            optimizer.zero_grad()
            out = model(batch_X, feat_input=batch_F)
            
            # [v5.0] ALEATORIC LOSS (Uncertainty Weighting)
            # Loss = (1 / 2*exp(log_var)) * Base_Loss + 0.5 * log_var
            base_loss = criterion(out["logits"], batch_Y)
            log_var = out["log_var"].squeeze()
            
            # Weighting the loss by predicted uncertainty
            weighted_loss = torch.mean(torch.exp(-log_var) * base_loss + 0.5 * log_var)
            
            # Auxiliary Confidence Loss (MSE between p_win and reality)
            reality_p = (batch_Y != 1).float()
            conf_loss = F.mse_loss(out["p_win"].squeeze(), reality_p)
            
            total_loss = weighted_loss + 0.2 * conf_loss
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            
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
