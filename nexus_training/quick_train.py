import sys
import os
import asyncio
import torch
import logging
import pandas as pd
import numpy as np
import torch.nn.functional as F
from typing import Dict, Optional
import json
from datetime import datetime

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ml.feat_processor import feat_processor
from app.ml.models.hybrid_probabilistic import HybridProbabilistic
from app.core.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger("QUICK_TRAIN")

async def evaluate_model(model, X, Y, F_in, criterion):
    """Evaluates the model and returns the average loss."""
    model.eval()
    with torch.no_grad():
        out = model(X, feat_input=F_in)
        base_loss = criterion(out["logits"], Y)
        log_var = out["log_var"].squeeze()
        weighted_loss = torch.mean(torch.exp(-log_var) * base_loss + 0.5 * log_var)
        
        reality_p = (Y != 1).float()
        conf_loss = F.mse_loss(out["p_win"].squeeze(), reality_p)
        
        total_loss = weighted_loss + 0.2 * conf_loss
        return total_loss.item()

async def quick_train_brain(symbol="XAUUSD"):
    logger.info(f"🧠 STARTING INSTITUTIONAL EVOLUTION: {symbol} 🧠")
    
    # 1. SMART DELTA SYNC (Real Data Enforcement)
    from nexus_core.data_collector import smart_sync_data
    
    logger.info("🔄 Initiating Smart Delta Sync...")
    df_raw = await smart_sync_data(symbol)
    
    if df_raw is None or len(df_raw) < 200:
        logger.error("❌ ABORTING: Insufficient REAL data for training. Delta sync returned too few candles.")
        raise Exception("AutoML Aborted: No Real Data Available or insufficient samples.")

    # 2. Feature Engineering
    logger.info(f"🧪 Engineering Features for {len(df_raw)} candles...")
    df = feat_processor.process_dataframe(df_raw)
    
    # Populate Neural Dimensions
    latent_rows = []
    for idx, row in df.iterrows():
        latent_rows.append(feat_processor.compute_latent_vector(row))
    
    new_df = pd.DataFrame(latent_rows)
    new_df['close'] = df['close'].reset_index(drop=True)
    df = new_df.dropna().reset_index(drop=True)
    
    feat_cols = list(settings.NEURAL_FEATURE_NAMES)
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    
    seq_len = settings.LSTM_SEQ_LEN
    if len(df) < seq_len + 120: # 100 for val + 20 for horizon
        logger.error("❌ ABORTING: Dataset too small after feature engineering.")
        return

    # 3. Build Tensors
    X_list, y_list = [], []
    feat_inputs_list = {"form": [], "space": [], "accel": [], "time": [], "kinetic": []}
    
    horizon = 20
    tp_mult, sl_mult = 1.5, 1.0
    
    for i in range(seq_len, len(df) - horizon):
        X_list.append(torch.tensor(df.iloc[i-seq_len:i][feat_cols].values, dtype=torch.float32))
        
        row = df.iloc[i]
        metrics = feat_processor.compute_latent_vector(row)
        
        feat_inputs_list["form"].append([metrics.get("physics_entropy", 0.0), metrics.get("physics_viscosity", 0.0), metrics.get("structural_feat_index", 0.0), 0.0])
        feat_inputs_list["space"].append([metrics.get("confluence_tensor", 0.0), metrics.get("killzone_intensity", 0.0), metrics.get("session_weight", 0.0)])
        feat_inputs_list["accel"].append([metrics.get("physics_energy", 0.0), metrics.get("physics_force", 0.0), metrics.get("volatility_context", 1.0)])
        feat_inputs_list["time"].append([metrics.get("temporal_sin", 0.0), metrics.get("temporal_cos", 0.0), metrics.get("killzone_intensity", 0.0), metrics.get("session_weight", 0.0)])
        feat_inputs_list["kinetic"].append([metrics.get("trap_score", 0.0), metrics.get("structural_feat_index", 0.0), metrics.get("physics_force", 0.0), metrics.get("physics_viscosity", 0.0)])
        
        # Triple Barrier
        price_start = df.iloc[i]['close']
        atr = df.iloc[i].get('atr_accel', price_start * 0.001)
        up_b, dn_b = price_start + (atr * tp_mult), price_start - (atr * sl_mult)
        label = 1
        for j in range(1, horizon):
            curr_p = df.iloc[i+j]['close']
            if curr_p >= up_b: label = 2; break
            elif curr_p <= dn_b: label = 0; break
        y_list.append(label)
        
    X_all = torch.stack(X_list)
    Y_all = torch.tensor(y_list, dtype=torch.long)
    F_all = {k: torch.tensor(v, dtype=torch.float32) for k, v in feat_inputs_list.items()}

    # 4. CHALLENGER SPLIT (Recent 100 for Validation)
    val_size = 100
    train_size = len(X_all) - val_size
    
    X_train, X_val = X_all[:train_size], X_all[train_size:]
    Y_train, Y_val = Y_all[:train_size], Y_all[train_size:]
    F_train = {k: v[:train_size] for k, v in F_all.items()}
    F_val = {k: v[train_size:] for k, v in F_all.items()}

    # 5. CHALLENGER PROTOCOL: Evaluate Existing Model
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    model_path = os.path.join(settings.MODELS_DIR, f"hybrid_prob_{symbol}_v2.pt")
    old_val_loss = float('inf')

    if os.path.exists(model_path):
        logger.info("🕵️ Evaluating current Veteran Model on out-of-sample data...")
        try:
            checkpoint = torch.load(model_path)
            old_model = HybridProbabilistic(input_dim=len(feat_cols), hidden_dim=settings.NEURAL_HIDDEN_DIM, num_classes=3)
            old_model.load_state_dict(checkpoint["state_dict"])
            old_val_loss = await evaluate_model(old_model, X_val, Y_val, F_val, criterion)
            logger.info(f"📉 Veteran Model Validation Loss: {old_val_loss:.4f}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load old model for comparison: {e}")

    # 6. Train Challenger
    challenger = HybridProbabilistic(input_dim=len(feat_cols), hidden_dim=settings.NEURAL_HIDDEN_DIM, num_classes=3)
    if os.path.exists(model_path):
        challenger.load_state_dict(torch.load(model_path)["state_dict"]) # Warm start
    
    optimizer = torch.optim.AdamW(challenger.parameters(), lr=1e-4, weight_decay=1e-5)
    
    logger.info(f"🚀 Training Challenger on {train_size} samples...")
    challenger.train()
    batch_size, epochs = 64, 20
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            b_X, b_Y = X_train[i:i+batch_size], Y_train[i:i+batch_size]
            b_F = {k: v[i:i+batch_size] for k, v in F_train.items()}
            
            optimizer.zero_grad()
            out = challenger(b_X, feat_input=b_F)
            base_l = criterion(out["logits"], b_Y)
            l_var = out["log_var"].squeeze()
            w_loss = torch.mean(torch.exp(-l_var) * base_l + 0.5 * l_var)
            c_loss = F.mse_loss(out["p_win"].squeeze(), (b_Y != 1).float())
            
            total_l = w_loss + 0.2 * c_loss
            total_l.backward()
            optimizer.step()
            epoch_loss += total_l.item()

    # 7. FINAL EVALUATION: Verify if Challenger is better
    new_val_loss = await evaluate_model(challenger, X_val, Y_val, F_val, criterion)
    logger.info(f"📊 Challenger Validation Loss: {new_val_loss:.4f} (v Veteran: {old_val_loss:.4f})")

    if new_val_loss < old_val_loss or old_val_loss == float('inf'):
        logger.info("✅ CHALLENGER WINS. Promoting and Persisting update...")
        os.makedirs(settings.MODELS_DIR, exist_ok=True)
        torch.save({
            "state_dict": challenger.state_dict(),
            "config": {"seq_len": seq_len, "input_dim": len(feat_cols), "hidden_dim": settings.NEURAL_HIDDEN_DIM},
            "version": "5.0_evolution",
            "val_loss": new_val_loss,
            "timestamp": datetime.now().isoformat()
        }, model_path)
    else:
        logger.warning("❌ CHALLENGER FAILED to improve validation metrics. Retaining Veteran Model to prevent brain degradation.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    args = parser.parse_args()
    asyncio.run(quick_train_brain(args.symbol))
