
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
from nexus_training.loss import ConvergentSingularityLoss
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
    def __init__(self, sequences, labels, physics, static_features: Dict[str, np.ndarray], spatial_maps: np.ndarray = None):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.physics = torch.FloatTensor(physics)
        
        # [LEVEL 41] Static Features for FeatEncoder
        self.form = torch.FloatTensor(static_features.get("form", np.zeros((len(labels), 4))))
        self.space = torch.FloatTensor(static_features.get("space", np.zeros((len(labels), 3))))
        self.accel = torch.FloatTensor(static_features.get("accel", np.zeros((len(labels), 3))))
        self.time = torch.FloatTensor(static_features.get("time", np.zeros((len(labels), 4))))
        self.kinetic = torch.FloatTensor(static_features.get("kinetic", np.zeros((len(labels), 4))))
        
        # [LEVEL 54] Spatial Energy Maps
        if spatial_maps is not None:
            self.spatial_maps = torch.FloatTensor(spatial_maps)
        else:
            self.spatial_maps = torch.zeros((len(labels), 1, 50, 50))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.physics[idx], \
               self.form[idx], self.space[idx], self.accel[idx], self.time[idx], \
               self.kinetic[idx], self.spatial_maps[idx]

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
    criterion = ConvergentSingularityLoss(
        kinetic_lambda=settings.NEURAL_LOSS_KINETIC_LAMBDA, 
        spatial_lambda=settings.NEURAL_LOSS_SPATIAL_LAMBDA
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=settings.NEURAL_LEARNING_RATE, 
        weight_decay=settings.NEURAL_WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    # 4. Training Loop
    best_loss = float('inf')
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for seq, label, phys_val, f_form, f_space, f_accel, f_time, f_kin, f_map in progress:
            
            # Prepare Data
            seq = seq.to(device) 
            label = label.to(device)
            phys_val = phys_val.to(device)
            f_map = f_map.to(device)
            
            # Reconstruct Feat Input Dict
            feat_input = {
                "form": f_form.to(device),
                "space": f_space.to(device),
                "accel": f_accel.to(device),
                "time": f_time.to(device),
                "kinetic": f_kin.to(device),
                "spatial_map": f_map  # [LEVEL 54]
            }
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(seq, feat_input=feat_input) 
            logits = outputs["logits"] # (Batch, 3)
            
            # Loss (Singularity Aware)
            loss = criterion(logits, label, phys_val, x_map=f_map)
            
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

def load_real_data(symbol: str, data_path: str = None, seq_len=60):
    """
    [v5.0 - SMC DOCTORAL] Ingests and Hydrates data with full Institutional Structure.
    Supports Parquet (preferred) or SQLite.
    """
    if data_path and data_path.endswith(".parquet"):
        logger.info(f"📂 Loading Parquet Data: {data_path}")
        df = pd.read_parquet(data_path)
        # Ensure timestamp index
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
    else:
        # Fallback to DB (Legacy)
        import sqlite3
        db_path = "data/market_data.db"
        if not os.path.exists(db_path):
            logger.error(f"❌ Database not found: {db_path}")
            return None, None, None, None
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM market_data WHERE symbol = '{symbol}' ORDER BY tick_time ASC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        if 'tick_time' in df.columns:
            df['tick_time'] = pd.to_datetime(df['tick_time'])
            df.set_index('tick_time', inplace=True)

    if df.empty:
        logger.error("❌ No data found.")
        return None, None, None, None

    # [v5.0] DEEP STRUCTURAL HYDRATION
    # We must call the Structure Engine and FeatProcessor singletons to fill the NEW columns
    from app.ml.feat_processor.engine import FeatProcessor
    processor = FeatProcessor()
    
    logger.info("🧪 Hydrating Structural Narrative (SMC Stack)...")
    df = processor.process_dataframe(df)
    
    # [LABELING] Apply Triple Barrier if labels are missing
    if 'label' not in df.columns:
        from nexus_training.labeling import label_dataset_triple_barrier
        logger.info("🏷️ Labeling data with Triple Barrier protocol...")
        prices = df['close'].values
        # Label every bar (sliding window target)
        indices = np.arange(len(df))
        # pt/sl roughly 2*ATR
        atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
        df['label'] = label_dataset_triple_barrier(prices, indices, pt=atr*2/prices[-1], sl=atr*2/prices[-1], h=100)
        # Map -1,0,1 to 0,1,2 (Classes)
        df['label'] = df['label'].map({-1.0: 0, 0.0: 1, 1.0: 2})

    # [TOPOLOGICAL NORMALIZATION]
    # Ensure all dist_* columns are correctly ATR-normalized (should be done by processor)
    seq_cols = [
        'dist_micro', 'dist_struct', 'dist_macro', 'dist_bias',
        'volume_z_score', 'ofi_z', 'energy_z', 'accel_score'
    ]
    
    # Validate columns
    for c in seq_cols:
        if c not in df.columns: df[c] = 0.0
    
    df[seq_cols] = df[seq_cols].fillna(0.0)

    sequences, labels, physics_vals = [], [], []
    static_feats = {
        "form": [], "space": [], "accel": [], "time": [], "kinetic": []
    }
    
    # Pre-calculate data for loop speed
    data_values = df[seq_cols].values
    label_values = df['label'].values
    
    logger.info(f"🌀 Generating {len(df)-seq_len} sliding windows...")
    for i in range(seq_len, len(df)):
        sequences.append(data_values[i-seq_len : i])
        labels.append(int(label_values[i]))
        physics_vals.append(0.0) # To be used for Physics-Aware loss if needed
        
        row = df.iloc[i]
        
        # [v5.0] SATURATING STATIC BAGS WITH TOPOLOGY
        # 1. Kinetic: Structural Stability & Gaps
        static_feats["kinetic"].append([
            row.get('struct_age', 0), 
            row.get('fvg_bull', 0) - row.get('fvg_bear', 0),
            row.get('layer_alignment', 0),
            row.get('is_inducement', 0)
        ])
        
        # 2. Form: Range Maturity & Premium/Discount
        static_feats["form"].append([
            row.get('range_pos', 0.5), # 0.0 to 1.0
            row.get('in_discount', 0),
            row.get('in_premium', 0),
            row.get('feat_form', 0)
        ])
        
        # 3. Space: Liquidity Gravity
        static_feats["space"].append([
            row.get('is_eqh', 0) or row.get('is_eql', 0),
            row.get('test_count', 0),
            row.get('is_mitigated', 0)
        ])
        
        # 4. Accel: Order Flow Aggression (OFI)
        static_feats["accel"].append([
            row.get('ofi_z', 0),
            row.get('energy_z', 0),
            row.get('accel_score', 0)
        ])
        
        # 5. Time: Session Weights
        static_feats["time"].append([
            row.get('session_weight', 0),
            1.0 if row.get('session_type') == 'NY_OPEN' else 0,
            1.0 if row.get('session_type') == 'LONDON' else 0,
            0
        ])

    X = np.array(sequences) 
    y = np.array(labels)
    phys = np.array(physics_vals)
    
    for k in static_feats:
        static_feats[k] = np.array(static_feats[k])
        
    return X, y, phys, static_feats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="XAUUSD")
    parser.add_argument("--data_path", type=str, default=None, help="Path to Parquet data")
    parser.add_argument("--real", action="store_true", help="Use Real Data Pipeline")
    parser.add_argument("--epochs", type=int, default=settings.NEURAL_EPOCHS)
    args = parser.parse_args()
    
    if args.real or args.data_path:
        print(f"🧬 INITIATING GREAT RETRAINING PROTOCOL FOR {args.symbol}...")
        X, y, phys, feats = load_real_data(args.symbol, data_path=args.data_path)
        
        if X is not None:
            temp_path = "data/temp_real_train.npz"
            np.savez(temp_path, X=X, y=y, physics=phys, static_features=feats)
            
            train_hybrid_model(args.symbol, temp_path, epochs=args.epochs, batch_size=settings.NEURAL_BATCH_SIZE)
            
            if os.path.exists(temp_path):
               os.remove(temp_path)
            print("🚀 RETRAINING COMPLETE. SYSTEM UPGRADED.")
    else:
        print("🧪 RUNNING CONVERGENCE PIPELINE SMOKE TEST...")
        # ...
