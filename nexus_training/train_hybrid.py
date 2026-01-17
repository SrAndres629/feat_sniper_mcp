
import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# [LEVEL 50] Use the Advanced Probabilistic Model
from app.ml.models.hybrid_probabilistic import HybridProbabilistic
from nexus_training.loss import PhysicsAwareLoss
from app.core.config import settings

# Logger Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HybridTrainer")

class SniperDataset(Dataset):
    """
    Dataset wrapper for FEAT Sniper data.
    Now handles Heterogeneous Inputs:
    1. Sequences (TCN)
    2. Static/Latent Features (FeatEncoder)
    3. Physics Labels (Loss)
    """
    def __init__(self, sequences, labels, physics, static_features: Dict[str, np.ndarray]):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.physics = torch.FloatTensor(physics)
        
        # [LEVEL 41] Static Features for FeatEncoder
        # Check if keys exist, else zero-init
        self.form = torch.FloatTensor(static_features.get("form", np.zeros((len(labels), 4))))
        self.space = torch.FloatTensor(static_features.get("space", np.zeros((len(labels), 3))))
        self.accel = torch.FloatTensor(static_features.get("accel", np.zeros((len(labels), 3))))
        self.time = torch.FloatTensor(static_features.get("time", np.zeros((len(labels), 4))))
        self.kinetic = torch.FloatTensor(static_features.get("kinetic", np.zeros((len(labels), 4)))) # [LEVEL 50]
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        # Package feat_input on the fly
        feat_input = {
            "form": self.form[idx].unsqueeze(0), # (1, 4) to match batch dim expectation if needed usually just (dim)
            "space": self.space[idx].unsqueeze(0),
            "accel": self.accel[idx].unsqueeze(0),
            "time": self.time[idx].unsqueeze(0),
            "kinetic": self.kinetic[idx].unsqueeze(0)
        }
        # Note: Dataloader collate will stack these. 
        # Actually standard collate works well if we return dicts, but let's correct shapes in loop
        
        # Return flattened dict elements to be stacked by Collate? 
        # Simpler: Return Tuple, reconstruct Dict in loop
        return self.sequences[idx], self.labels[idx], self.physics[idx], \
               self.form[idx], self.space[idx], self.accel[idx], self.time[idx], self.kinetic[idx]

def train_hybrid_model(symbol: str, data_path: str, epochs=50, batch_size=32):
    """
    [LEVEL 50] Training Loop for HybridProbabilistic.
    Uses PhysicsAwareLoss and Kinetic Latent Features.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Starting Hybrid Training for {symbol} on {device}")
    
    # 1. Load Data
    if not os.path.exists(data_path):
        logger.error(f"Data file {data_path} not found.")
        return

    logger.info("Loading training data...")
    raw_data = np.load(data_path, allow_pickle=True)
    
    # Expect standard arrays
    X = raw_data['X'] # (N, Seq, Channels) for TCN. Note: TCN expects (N, C, L) usually, verify model.
    y = raw_data['y'] # (N,) Classes 0,1,2
    phys = raw_data['physics'] # (N,)
    
    # [LEVEL 50] Load Static Features dictionary
    # File should have 'static_features.npy' or keys in the npz
    # We assume 'static_features' key contains a pickled dict or similar
    if 'static_features' in raw_data:
        static_feats = raw_data['static_features'].item()
    else:
        logger.warning("No 'static_features' found. Using Zeros.")
        static_feats = {}

    dataset = SniperDataset(X, y, phys, static_feats)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Initialize Model
    # Verify input dims. TCN expects input_channels.
    # X shape: (N, Len, Chan) or (N, Chan, Len)? 
    # Usually data is (N, L, C). Model needs (N, C, L). We transpose in loop or dataset.
    seq_len = X.shape[1]
    input_dim = X.shape[2]
    
    model = HybridProbabilistic(input_dim=input_dim, hidden_dim=64, num_classes=3).to(device)
    
    # 3. Loss & Optimizer
    criterion = PhysicsAwareLoss(physics_lambda=0.5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # 4. Training Loop
    best_loss = float('inf')
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for seq, label, phys_val, f_form, f_space, f_accel, f_time, f_kin in progress:
            
            # Prepare Data
            # Model expects (Batch, Seq_Len, Channels) and handles permute internally
            seq = seq.to(device) 
            label = label.to(device)
            phys_val = phys_val.to(device)
            
            # Reconstruct Feat Input Dict
            feat_input = {
                "form": f_form.to(device),
                "space": f_space.to(device),
                "accel": f_accel.to(device),
                "time": f_time.to(device),
                "kinetic": f_kin.to(device)
            }
            
            optimizer.zero_grad()
            
            # Forward
            logits = model(seq, feat_input=feat_input) # (Batch, 3)
            
            # Loss (Physics Aware)
            loss = criterion(logits, label, phys_val)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Metrics
            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)
            
            progress.set_postfix(loss=loss.item())
            
        avg_loss = epoch_loss / len(dataloader)
        acc = correct / total
        
        logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
        scheduler.step(avg_loss)
        
        # Save Best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(settings.MODELS_DIR, f"hybrid_prob_{symbol}_v2.pt")
            torch.save({
                "state_dict": model.state_dict(),
                "best_acc": acc,
                "config": {"input_dim": input_dim, "hidden_dim": 64}
            }, save_path)
            logger.info(f"💾 Model checkpoint saved to {save_path}")

    logger.info("✅ Training Complete.")

def load_real_data(symbol: str, db_path: str = "data/market_data.db", seq_len=60, limit=50000):
    """
    [LEVEL 50] Ingests Real Market Data from SQLite for The Great Retraining.
    Performs on-the-fly Tensorization and Feature Extraction.
    """
    import sqlite3
    
    if not os.path.exists(db_path):
        logger.error(f"❌ Database not found: {db_path}")
        return None, None, None, None

    logger.info(f"🔌 Connecting to {db_path} for {symbol}...")
    
    conn = sqlite3.connect(db_path)
    # Fetch only labeled data
    query = f"""
        SELECT * FROM market_data 
        WHERE symbol = '{symbol}' 
        AND label IS NOT NULL 
        ORDER BY tick_time ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        logger.error("❌ No labeled data found in DB. Run DataCollector first.")
        return None, None, None, None
        
    # [ROBUSTNESS] Ensure DatetimeIndex for Feat/Structure Engines
    if 'tick_time' in df.columns:
        df['tick_time'] = pd.to_datetime(df['tick_time'])
        df = df.set_index('tick_time').sort_index()

    logger.info(f"✅ Loaded {len(df)} labeled samples. Regenerating Features via Sensor Fusion...")
    
    # [LEVEL 50] RE-HYDRATE FEATURES
    # We use the latest FeatProcessor to ensure the data matches current logic exactly.
    # This fills: layer_alignment, kinetic_pattern_id, dist_micro, etc.
    from app.ml.feat_processor import feat_processor
    
    # Ensure DF has specific columns expected by feat_processor (lowercase)
    # DB columns are usually correct.
    # feat_processor expects 'close', 'high', 'low', 'volume'
    
    # Run the full pipeline
    try:
        df = feat_processor.apply_feat_engineering(df)
        logger.info("✅ Feature Engineering Complete.")
    except Exception as e:
        logger.error(f"⚠️ Feature Engineering Failed: {e}")
        # Fallback to existing columns
    
    # [FEATURE ENGINEERING]
    # 1. Normalize/Preprocess
    # Log Returns for Stationarity
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    df['vol_norm'] = (df['volume'] - df['volume'].rolling(50).mean()) / (df['volume'].rolling(50).std() + 1e-9)
    df['vol_norm'] = df['vol_norm'].fillna(0)
    
    # 2. Select Sequence Features (Dynamic TCN Channels)
    # Now we can use the exact columns generated by feat_processor
    seq_cols = ['log_ret', 'vol_norm', 'rsi', 'dist_micro', 'dist_struct', 'dist_macro']
    # If columns missing, fill 0
    for c in seq_cols:
        if c not in df.columns: df[c] = 0.0
            
    # 3. Create Sliding Windows (X) and Targets (y)
    sequences = []
    labels = []
    physics_vals = []
    
    # Static Features Bags
    static_feats = {
        "form": [], "space": [], "accel": [], "time": [], "kinetic": []
    }
    
    data_values = df[seq_cols].values
    label_values = df['label'].values
    
    # Kinetic Columns existing in DF (from FeatProcessor)
    # micro_compression, micro_slope, layer_alignment, bias_slope
    
    for i in range(seq_len, len(df)):
        # X: Window [i-seq_len : i]
        seq = data_values[i-seq_len : i] # (60, Channels)
        
        # y: Target at i
        target = int(label_values[i])
        
        sequences.append(seq)
        labels.append(target)
        physics_vals.append(0.0)
        
        # Static Features (Current State at i)
        row = df.iloc[i]
        
        # Kinetic Tensor Construction
        # Uses regenerated accurate features
        k_vec = [
            row.get('micro_compression', 0), 
            row.get('micro_slope', 0), 
            row.get('layer_alignment', 0), # Now populated!
            row.get('bias_slope', 0)
        ]
        static_feats["kinetic"].append(k_vec)
        
        # Context placeholders (can be expanded with real logic)
        static_feats["form"].append([row.get('entropy_proxy',0), 0,0,0])
        static_feats["space"].append([row.get('dist_poc_norm',0), 0,0])
        static_feats["accel"].append([row.get('energy_z_score',0), 0,0])
        static_feats["time"].append([0,0,0,0])

    # Convert to Arrays
    X = np.array(sequences) # (N, 60, C)
    y = np.array(labels)
    phys = np.array(physics_vals)
    
    for k in static_feats:
        static_feats[k] = np.array(static_feats[k])
        
    # [CRITICAL] Transpose X for TCN: (N, L, C) -> (N, C, L) if model expects C first?
    # Model HybridProbabilistic permutes internally: x.permute(0, 2, 1). 
    # So we provide (N, Seq, Chan). Matches 'sequences' shape.
    
    return X, y, phys, static_feats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="SYNTHETIC_TEST")
    parser.add_argument("--real", action="store_true", help="Use Real DB Data")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    if args.real:
        print(f"🧬 INITIATING GREAT RETRAINING PROTOCOL FOR {args.symbol}...")
        X, y, phys, feats = load_real_data(args.symbol, limit=None)
        
        if X is not None:
            # Save temporary npz to reuse existing function or modify train function?
            # Existing train_hybrid_model loads from path. Let's patch it or save temp.
            temp_path = "data/temp_real_train.npz"
            np.savez(temp_path, X=X, y=y, physics=phys, static_features=feats)
            
            train_hybrid_model(args.symbol, temp_path, epochs=args.epochs, batch_size=64)
            
            # Clean up
            if os.path.exists(temp_path):
               os.remove(temp_path)
            print("🚀 RETRAINING COMPLETE. SYSTEM UPGRADED.")
    else:
        # Legacy Smoke Test
        fake_data = "data/synthetic_convergence.npz"
        if os.path.exists(fake_data):
            print("🧪 RUNNING CONVERGENCE PIPELINE SMOKE TEST...")
            train_hybrid_model("SYNTHETIC_TEST", fake_data, epochs=5, batch_size=32)
        else:
            print("⚠️ Synthetic data not found. Run generate_synthetic_convergence.py first.")
