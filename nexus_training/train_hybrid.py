import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# [LEVEL 50] Advanced Models & Config
from app.ml.models.hybrid_probabilistic import HybridProbabilistic
from nexus_training.loss import ConvergentSingularityLoss
from app.core.config import settings

# Logger Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HybridTrainer")

# --- DATASET ---
class SniperDataset(Dataset):
    """
    Dataset wrapper for FEAT Sniper data.
    Now handles Heterogeneous Inputs & Static Features properly.
    """
    def __init__(self, sequences, labels, physics, static_features: Dict[str, np.ndarray], spatial_maps: np.ndarray = None):
        # Keep as numpy for memory efficiency until __getitem__
        self.sequences = sequences
        self.labels = labels
        self.physics = physics # Placeholder (mostly zeros in raw load)
        
        # Static Bags
        self.form = static_features.get("form", np.zeros((len(labels), 4)))
        self.space = static_features.get("space", np.zeros((len(labels), 3)))
        self.accel = static_features.get("accel", np.zeros((len(labels), 3)))
        self.time = static_features.get("time", np.zeros((len(labels), 4)))
        self.kinetic = static_features.get("kinetic", np.zeros((len(labels), 4)))
        
        # Spatial Maps
        if spatial_maps is not None:
            self.spatial_maps = spatial_maps
        else:
            self.spatial_maps = np.zeros((len(labels), 1, 50, 50))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        # Convert to Tensor ON DEMAND (Saves RAM)
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.LongTensor([self.labels[idx]]).squeeze(),
            torch.FloatTensor([self.physics[idx]]),
            torch.FloatTensor(self.form[idx]),
            torch.FloatTensor(self.space[idx]),
            torch.FloatTensor(self.accel[idx]),
            torch.FloatTensor(self.time[idx]),
            torch.FloatTensor(self.kinetic[idx]),
            torch.FloatTensor(self.spatial_maps[idx])
        )

# --- UTILS ---
def pre_flight_guard(symbol: str, real: bool = False):
    logger.info("🛡️ Running Pre-Flight Guard...")
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    logger.info(f"✅ Hardware check passed (Running on {device}).")
    return True

# --- TRAINING LOOP ---
def train_hybrid_model(symbol: str, data_path: str, epochs=50, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Starting Hybrid Training for {symbol} on {device}")
    
    # 1. Load Data
    if not os.path.exists(data_path):
        logger.error(f"Data file {data_path} not found.")
        return

    logger.info("Loading training data...")
    raw_data = np.load(data_path, allow_pickle=True)
    
    X = raw_data['X'] 
    y = raw_data['y'] 
    phys = raw_data['physics']
    
    if 'static_features' in raw_data:
        static_feats = raw_data['static_features'].item()
    else:
        static_feats = {}

    # 2. Prepare Dataset & Split (VALIDATION SET ADDED)
    full_dataset = SniperDataset(X, y, phys, static_feats)
    
    # Split 80/20 to prevent Overfitting
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 3. Dynamic Sampling (Fix Class Imbalance)
    logger.info("⚖️ Calculating Class Weights for Sampler...")
    train_indices = train_dataset.indices
    train_labels = y[train_indices]
    
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / (class_counts + 1e-9)
    sample_weights = class_weights[train_labels]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 4. Initialize Model
    seq_len = X.shape[1]
    input_dim = X.shape[2]
    
    model = HybridProbabilistic(input_dim=input_dim, hidden_dim=64, num_classes=3).to(device)
    
    # 5. Loss & Optimizer
    criterion = ConvergentSingularityLoss(
        kinetic_lambda=settings.NEURAL_LOSS_KINETIC_LAMBDA, 
        spatial_lambda=settings.NEURAL_LOSS_SPATIAL_LAMBDA
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=settings.NEURAL_LEARNING_RATE, 
        weight_decay=settings.NEURAL_WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # 6. Training Loop
    best_val_loss = float('inf')
    early_stop_patience = 8
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        total = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for seq, label, _, f_form, f_space, f_accel, f_time, f_kin, f_map in progress:
            
            # [REVERT] The model (HybridProbabilistic) already performs .permute(0, 2, 1) internally.
            # Passing (Batch, Length, Channels) ensures the model's TCN sees (Batch, Channels, Length).
            seq = seq.to(device)
            
            label = label.to(device)
            f_map = f_map.to(device)
            
            feat_input = {
                "form": f_form.to(device),
                "space": f_space.to(device),
                "accel": f_accel.to(device),
                "time": f_time.to(device),
                "kinetic": f_kin.to(device),
                "spatial_map": f_map
            }
            
            # [FIX 2] Real Physics Tensor Construction from Bags
            p_tensor = torch.stack([
                f_accel[:, 1].to(device), # Energy
                f_accel[:, 2].to(device), # Force
                f_kin[:, 0].to(device),   # Entropy proxy
                f_kin[:, 2].to(device)    # Viscosity proxy
            ], dim=1)

            optimizer.zero_grad()
            
            # Forward
            outputs = model(seq, feat_input=feat_input, physics_tensor=p_tensor) 
            logits = outputs["logits"]
            log_var = outputs.get("log_var") # Aleatoric Uncertainty
            
            # [FIX 3] Bayesian Loss
            base_loss = criterion(logits, label, p_tensor, x_map=f_map)
            
            if log_var is not None:
                loss = torch.mean(torch.exp(-log_var.squeeze()) * base_loss + 0.5 * log_var.squeeze())
            else:
                loss = base_loss.mean()
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_acc += (preds == label).sum().item()
            total += label.size(0)
            
            progress.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / total
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for seq, label, _, f_form, f_space, f_accel, f_time, f_kin, f_map in val_loader:
                seq = seq.to(device) # No transpose needed here
                label = label.to(device)
                f_map = f_map.to(device)
                
                feat_input = {
                    "form": f_form.to(device), "space": f_space.to(device),
                    "accel": f_accel.to(device), "time": f_time.to(device),
                    "kinetic": f_kin.to(device), "spatial_map": f_map
                }
                
                p_tensor = torch.stack([
                    f_accel[:, 1].to(device), f_accel[:, 2].to(device),
                    f_kin[:, 0].to(device), f_kin[:, 2].to(device)
                ], dim=1)

                outputs = model(seq, feat_input=feat_input, physics_tensor=p_tensor)
                logits = outputs["logits"]
                
                loss = criterion(logits, label, p_tensor, x_map=f_map).mean()
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == label).sum().item()
                val_total += label.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        logger.info(f"Ep {epoch+1} | Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f}")
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(settings.MODELS_DIR, f"hybrid_prob_{symbol}_v2.pt")
            torch.save({
                "state_dict": model.state_dict(),
                "best_acc": val_acc,
                "config": {"input_dim": input_dim, "hidden_dim": 64}
            }, save_path)
            logger.info(f"💾 NEW BEST MODEL SAVED (Val Loss: {avg_val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                logger.info("🛑 Early stopping triggered.")
                break

    logger.info("✅ Hybrid Training Complete.")

def load_real_data(symbol: str, data_path: str = None, seq_len=60):
    """
    [v5.0 - SMC DOCTORAL] Ingests and Hydrates data with full Institutional Structure.
    Supports Parquet (preferred) or SQLite.
    """
    if data_path and data_path.endswith(".parquet"):
        logger.info(f"📂 Loading Parquet Data: {data_path}")
        df = pd.read_parquet(data_path)
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
    else:
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

    from app.ml.feat_processor.engine import FeatProcessor
    processor = FeatProcessor()
    
    logger.info("🧪 Hydrating Structural Narrative (SMC Stack)...")
    df = processor.process_dataframe(df)
    
    if 'label' not in df.columns:
        from nexus_training.labeling import label_dataset_triple_barrier
        logger.info("🏷️ Labeling data with Triple Barrier protocol...")
        prices = df['close'].values
        indices = np.arange(len(df))
        atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
        df['label'] = label_dataset_triple_barrier(prices, indices, pt=atr*2/prices[-1], sl=atr*2/prices[-1], h=100)
        df['label'] = df['label'].map({-1.0: 0, 0.0: 1, 1.0: 2})

    seq_cols = list(settings.NEURAL_FEATURE_NAMES)
    for c in seq_cols:
        if c not in df.columns: df[c] = 0.0
    
    df[seq_cols] = df[seq_cols].fillna(0.0)

    sequences, labels, physics_vals = [], [], []
    static_feats = {"form": [], "space": [], "accel": [], "time": [], "kinetic": []}
    
    data_values = df[seq_cols].values
    label_values = df['label'].values
    
    logger.info(f"🌀 Generating {len(df)-seq_len} sliding windows...")
    for i in range(seq_len, len(df)):
        sequences.append(data_values[i-seq_len : i])
        labels.append(int(label_values[i]))
        physics_vals.append(0.0)
        
        row = df.iloc[i]
        
        static_feats["kinetic"].append([
            row.get('struct_age', 0), 
            row.get('fvg_bull', 0) - row.get('fvg_bear', 0),
            row.get('layer_alignment', 0),
            row.get('is_inducement', 0)
        ])
        
        static_feats["form"].append([
            row.get('range_pos', 0.5), 
            row.get('in_discount', 0),
            row.get('in_premium', 0),
            row.get('feat_form', 0)
        ])
        
        static_feats["space"].append([
            row.get('is_eqh', 0) or row.get('is_eql', 0),
            row.get('test_count', 0),
            row.get('is_mitigated', 0)
        ])
        
        static_feats["accel"].append([
            row.get('ofi_z', 0),
            row.get('energy_z', 0),
            row.get('accel_score', 0)
        ])
        
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
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    # [D.A.L.C. INTEGRATION]
    print("\n🕵️ INVOKING D.A.L.C. AUDITOR...")
    import subprocess
    audit_result = subprocess.run(["python", "tests/dalc_sanity_check.py"])
    
    if audit_result.returncode != 0:
        print("🛑 TRAINING ABORTED BY D.A.L.C. (Logic Errors Detected).")
        print("Check the logs above to fix the Silent Killers.")
        sys.exit(1)
        
    print("✅ AUDIT PASSED. IGNITING TRAINING ENGINE.\n")
    
    if args.real:
        if pre_flight_guard(args.symbol, real=True):
            X, y, phys, feats = load_real_data(args.symbol)
            if X is not None:
                temp_path = "data/temp_real_train.npz"
                np.savez(temp_path, X=X, y=y, physics=phys, static_features=feats)
                train_hybrid_model(args.symbol, temp_path, epochs=args.epochs, batch_size=args.batch_size)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
