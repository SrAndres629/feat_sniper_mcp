import os
import sys
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from tqdm import tqdm
import logging
import wandb
from typing import Dict, List, Tuple

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# [LEVEL 50] Advanced Models & Config
from app.ml.models.hybrid_probabilistic import HybridProbabilistic
from nexus_training.loss import SovereignQuantLoss
from app.core.config import settings

# Logger Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HybridTrainer")

# --- ADVERSARIAL HARDENING ---
class ChaosInjector:
    """
    [DATA AUGMENTATION]
    Simulates Broker hostile conditions (Slippage, Spread Spikes, Noise).
    Makes the Neural Net robust against LiteFinance reality.
    """
    def __init__(self, noise_level=0.0001, spread_shock_prob=0.1):
        self.noise = noise_level
        self.spread_prob = spread_shock_prob

    def apply(self, sequence):
        """
        Input: Sequence (Length, Channels)
        Output: Augmented Sequence
        """
        # 1. Gaussian Noise (Jitter) - Nervous market simulation
        if np.random.random() < 0.5:
            noise = np.random.normal(0, self.noise, sequence.shape)
            sequence += noise

        # 2. Spread Shock / Volatility Distortion
        if np.random.random() < self.spread_prob:
            # Shift features to simulate extreme spread conditions
            sequence *= (1.0 + np.random.uniform(-0.01, 0.01, sequence.shape))
            
        return sequence

# --- DATASET ---
class SniperDataset(Dataset):
    """
    Dataset wrapper for FEAT Sniper data.
    Now handles Heterogeneous Inputs & Chaos Augmentation.
    """
    def __init__(self, sequences, labels, physics, static_features: Dict[str, np.ndarray], spatial_maps: np.ndarray = None, augment=False):
        # Keep as numpy for memory efficiency until __getitem__
        self.sequences = sequences
        self.labels = labels
        self.physics = physics 
        self.augment = augment
        self.injector = ChaosInjector()
        
        # Static Bags
        self.form = static_features.get("form", np.zeros((len(labels), 4)))
        self.space = static_features.get("space", np.zeros((len(labels), 3)))
        
        # [CIRCUIT BREAKER] Auto-fix accel dimension (3 -> 4) for Physics Viscosity
        raw_accel = static_features.get("accel", np.zeros((len(labels), 3)))
        if hasattr(raw_accel, 'shape') and raw_accel.shape[1] == 3:
            print("⚠️ [DATA-PATCH] Padding Accel from 3 to 4 columns (Adding Viscosity=0)")
            padding = np.zeros((len(labels), 1))
            self.accel = np.hstack([raw_accel, padding])
        else:
            self.accel = raw_accel
            
        self.time = static_features.get("time", np.zeros((len(labels), 4)))
        self.kinetic = static_features.get("kinetic", np.zeros((len(labels), 4)))
        
        # Spatial Maps
        if spatial_maps is not None:
            self.spatial_maps = spatial_maps
        else:
            self.spatial_maps = np.zeros((len(labels), 1, 50, 50))
        
        # [V6.1 DOCTORAL] Normalize Time for Recency Weighting
        # Range 0.0 (Oldest) to 1.0 (Newest)
        self.norm_time = np.linspace(0, 1, len(labels))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        # Convert to Tensor ON DEMAND (Saves RAM)
        seq = self.sequences[idx].copy()
        
        # Apply Chaos in Training
        if self.augment:
            seq = self.injector.apply(seq)
            
        return (
            torch.FloatTensor(seq),
            torch.LongTensor([self.labels[idx]]).squeeze(),
            torch.FloatTensor([self.physics[idx]]),
            torch.FloatTensor(self.form[idx]),
            torch.FloatTensor(self.space[idx]),
            torch.FloatTensor(self.accel[idx]),
            torch.FloatTensor(self.time[idx]),
            torch.FloatTensor(self.kinetic[idx]),
            torch.FloatTensor(self.spatial_maps[idx]),
            torch.FloatTensor([self.norm_time[idx]])
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
def train_hybrid_model(symbol: str, data_path: str, epochs=50, batch_size=32, fvg_dual=False, fractal_sync=False, force_cpu=False):
    """
    [V6.1.3 CORTEX HARDENING] Stable Training Protocol for Laptop GPUs.
    Implementation of 'Illegal Instruction' Defense Mechanisms.
    """
    # --- HARDWARE SECURITY LAYER ---
    if force_cpu:
        device = torch.device("cpu")
        logger.info("⚠️ FORCED CPU MODE (User requested)")
    else:
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                props = torch.cuda.get_device_properties(0)
                vram_gb = props.total_memory / 1e9
                logger.info(f"🛡️ GPU DETECTED: {props.name} | VRAM: {vram_gb:.1f} GB")

                # [PROTOCOL: CORTEX HARDENING]
                # 1. Disable cuDNN to prevent Kernel Panics on RTX 3060 Mobile
                # Standard cuDNN kernels can be unstable on consumer laptop GPUs under high channel density
                torch.backends.cudnn.enabled = False 
                logger.warning("🛡️ SECURITY: cuDNN Disabled (Avoiding Illegal Instruction Faults)")
                
                # 2. Force Sync for Error Tracking
                os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
                
                # 3. Disable TF32 (Reduces risk of illegal instructions in modern drivers)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False

                # 4. Auto-Scale Batch Size for Safety
                if vram_gb < 8.0 and batch_size > 8:
                    logger.warning(f"⚠️ VRAM Constraint ({vram_gb:.1f}GB). Downgrading Batch Size {batch_size} -> 8.")
                    batch_size = 8
                
                torch.cuda.empty_cache()
            else:
                device = torch.device("cpu")
        except Exception as e:
            logger.error(f"⚠️ CUDA Critical Failure: {e}. Fallback to CPU.")
            device = torch.device("cpu")

    logger.info(f"🚀 Starting Hybrid Training for {symbol} on {device} | Batch: {batch_size}")
    
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
    # Augmentation enabled ONLY for training set
    full_dataset = SniperDataset(X, y, phys, static_feats, augment=False)
    
    # Split 80/20 to prevent Overfitting
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    print(f"DEBUG: Dataset size: {len(full_dataset)}, Train: {train_size}, Val: {val_size}")
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    
    # Enable Augmentation on the training split wrapper
    train_dataset = SniperDataset(
        X[train_set.indices], y[train_set.indices], phys[train_set.indices], 
        {k: v[train_set.indices] for k, v in static_feats.items()}, 
        augment=True
    )
    val_dataset = SniperDataset(
        X[val_set.indices], y[val_set.indices], phys[val_set.indices], 
        {k: v[val_set.indices] for k, v in static_feats.items()}, 
        augment=False
    )
    
    # 3. Dynamic Sampling (Fix Class Imbalance)
    logger.info("⚖️ Calculating Class Weights for Sampler...")
    train_indices = train_set.indices
    train_labels = y[train_indices]
    
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / (class_counts + 1e-9)
    sample_weights = class_weights[train_labels]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Loaders - Optimized for Windows Stability (Main Thread Loading)
    # num_workers=0: Prevents multiprocessing crash on Windows
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    print(f"DEBUG: Loaders initialized. Train Batches: {len(train_loader)}")
    
    # 4. Initialize Model
    seq_len = X.shape[1]
    input_dim = X.shape[2]
    
    model = HybridProbabilistic(input_dim=input_dim, hidden_dim=settings.NEURAL_HIDDEN_DIM, num_classes=3).to(device)
    
    # 5. Loss & Optimizer (Elite Quant Grade)
    criterion = SovereignQuantLoss(
        k_lambda=settings.NEURAL_LOSS_KINETIC_LAMBDA, 
        s_lambda=settings.NEURAL_LOSS_SPATIAL_LAMBDA,
        fvg_dual_mode=fvg_dual,
        fractal_sync=fractal_sync
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=settings.NEURAL_LEARNING_RATE, 
        weight_decay=settings.NEURAL_WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    logger.info("Initializing Weights & Biases (W&B) Telemetry...")
    wandb.init(
        project="FEAT_SNIPER_XAUUSD",
        mode="online", 
        name=f"Hybrid_V5_Evolution_{symbol}",
        config={
            "learning_rate": settings.NEURAL_LEARNING_RATE,
            "batch_size": batch_size,
            "architecture": "TCN-BiLSTM-Probabilistic-PGU",
            "physics_aware": True,
            "initial_balance": 20.0,
            "leverage": "1:1000",
            "broker": "LiteFinance",
            "early_stopping_patience": 7,
            "fvg_dual_mode": fvg_dual,
            "fractal_sync": fractal_sync
        }
    )
    # [PILOT] Watch gradients and weights (Vital Signs)
    wandb.watch(model, log="all", log_freq=10)
    
    # 6. Training Loop
    best_val_loss = float('inf')
    early_stop_patience = 8
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        # [CURRICULUM] SLIDING BALANCE PRESSURE & ELITE ALPHA
        # Start gentle, end strict to avoid early "Decision Paralysis" (Freeze vs Risk)
        # Sovereign Protocol: Linear increase of elite penalties from Epoch 1 to 30
        pressure_factor = min(1.0, (epoch + 1) / max(1, epochs // 2))
        elite_alpha = min(1.0, (epoch + 1) / 30.0) 
        
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        total = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        current_balance = 20.0
        for i, (seq, label, _, f_form, f_space, f_accel, f_time, f_kin, f_map, t_norm) in enumerate(progress):
            
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
            
            # [V6 ELITE QUANT] 6-Channel Physics Tensor Construction
            p_tensor = torch.stack([
                f_accel[:, 0].to(device), # Energy
                f_accel[:, 1].to(device), # Force
                f_accel[:, 2].to(device), # Entropy
                f_accel[:, 3].to(device), # Viscosity
                f_time[:, 0].to(device),  # Volatility ATR
                f_time[:, 1].to(device)   # Killzone Intensity
            ], dim=1)

            optimizer.zero_grad()
            
            # [CORTEX HARDENING] NaN Guard for Forward Pass
            try:
                # Forward
                outputs = model(seq, feat_input=feat_input, physics_tensor=p_tensor) 
                logits = outputs["logits"]
                log_var = outputs.get("log_var") # Aleatoric Uncertainty
                
                # [FIX 3] Bayesian Loss with Curriculum Pressure & Recency Weighting
                base_loss = criterion(logits, label, p_tensor, x_map=f_map, current_balance=current_balance, alpha=elite_alpha, timestamps=t_norm.to(device))
                base_loss *= pressure_factor
                
                if log_var is not None:
                    loss = torch.mean(torch.exp(-log_var.squeeze()) * base_loss + 0.5 * log_var.squeeze())
                else:
                    loss = base_loss.mean()
                
                # [CORTEX HARDENING] Check for Infinite Loss BEFORE Backward
                if not torch.isfinite(loss):
                    logger.warning(f"⚠️ Infinite Loss detected at Step {i}. Skipping Batch.")
                    continue
                    
                loss.backward()
                
                # [CORTEX HARDENING] Gradient Clipping (Vital for LSTMs/TCNs)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct_batch = (preds == label).sum().item()
                train_acc += correct_batch
                total += label.size(0)
                
                progress.set_postfix(loss=loss.item())
                
            except RuntimeError as e:
                if "illegal instruction" in str(e) or "CUDA" in str(e):
                    logger.critical(f"🛑 CUDA FAULT at Step {i}. Attempting recovery...")
                    torch.cuda.empty_cache()
                    continue # Skip this batch, try to save the Epoch
                else:
                    raise e
            
            # [TELEMETRY] MICRO-LOGGING (High Frequency - Step by Step)
            if i % 10 == 0:
                # Calculate kinetic compliance
                probs = torch.softmax(logits, dim=1)
                kinetic_penalty = torch.mean(torch.abs(probs[:, 2] - probs[:, 0]) * (1.0 - torch.clamp(p_tensor[:, 1] / 5.0, 0.0, 1.0)))
                
                wandb.log({
                    "step_loss": loss.item(),  # Micro heartbeat
                    "batch_accuracy": correct_batch / label.size(0),
                    "physics_compliance_penalty": kinetic_penalty.item(),
                    "Elite_Penalty_Weight": elite_alpha,
                    "grad_norm": torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

            progress.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / total
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for seq, label, _, f_form, f_space, f_accel, f_time, f_kin, f_map, t_norm in val_loader:
                seq = seq.to(device) # No transpose needed here
                label = label.to(device)
                f_map = f_map.to(device)
                
                feat_input = {
                    "form": f_form.to(device), "space": f_space.to(device),
                    "accel": f_accel.to(device), "time": f_time.to(device),
                    "kinetic": f_kin.to(device), "spatial_map": f_map
                }
                
                p_tensor = torch.stack([
                    f_accel[:, 0].to(device), f_accel[:, 1].to(device),
                    f_accel[:, 2].to(device), f_accel[:, 3].to(device),
                    f_time[:, 0].to(device), f_time[:, 1].to(device)
                ], dim=1)

                outputs = model(seq, feat_input=feat_input, physics_tensor=p_tensor)
                logits = outputs["logits"]
                
                # Pass timestamps for recency-aware validation loss if needed
                loss = criterion(logits, label, p_tensor, x_map=f_map, timestamps=t_norm.to(device)).mean()
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(label.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        
        # [V6.1 DOCTORAL] Institutional Metrics Calculation
        all_targets_arr = np.array(all_targets)
        all_preds_arr = np.array(all_preds)
        val_acc = np.mean(all_preds_arr == all_targets_arr)
        
        # [V6.1 DOCTORAL] Institutional Metrics calculation
        # Simplified PnL for Sharpe/DD: Correct = +1, Wrong = -1.5, Neutral = 0
        pnl_series = np.where((all_preds_arr == all_targets_arr) & (all_preds_arr != 0), 1.0, 0.0)
        pnl_series = np.where((all_preds_arr != all_targets_arr) & (all_preds_arr != 0), -1.5, pnl_series)
        
        equity_series = [current_balance]
        max_eq = current_balance
        val_max_dd = 0.0
        for pnl in pnl_series:
            new_eq = equity_series[-1] + pnl * 0.1
            equity_series.append(new_eq)
            if new_eq > max_eq: max_eq = new_eq
            val_max_dd = max(val_max_dd, (max_eq - new_eq) / max_eq if max_eq > 0 else 0)
        
        current_balance = equity_series[-1]
        val_sharpe = (np.mean(pnl_series) / (np.std(pnl_series) + 1e-9)) * np.sqrt(252) if len(pnl_series) > 1 else 0.0

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": avg_train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "sharpe_ratio": val_sharpe,
            "max_drawdown": val_max_dd,
            "simulated_balance": current_balance,
            "capital_survival_status": 1.0 if current_balance > 8.0 else 0.0,
            "alpha": elite_alpha
        })

        logger.info(f"Ep {epoch+1} | Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f} | Sharpe: {val_sharpe:.2f} DD: {val_max_dd:.2%} | Bal: ${current_balance:.2f}")
        
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

def load_real_data(symbol: str, data_path: str = None, seq_len=60, limit=None, since_time=None):
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
        limit_clause = f" LIMIT {limit}" if limit else ""
        
        # [V6.1.1] Incremental Fetching logic
        if since_time is not None:
            # Fetch 200-candle buffer before since_time for structural continuity
            # We want to ensure M02/M03 have context
            buffer_query = f"""
                SELECT * FROM (
                    SELECT * FROM market_data 
                    WHERE symbol = '{symbol}' AND tick_time < '{since_time}' 
                    ORDER BY tick_time DESC LIMIT 200
                ) ORDER BY tick_time ASC
            """
            delta_query = f"SELECT * FROM market_data WHERE symbol = '{symbol}' AND tick_time >= '{since_time}' ORDER BY tick_time ASC"
            
            df_buffer = pd.read_sql_query(buffer_query, conn)
            df_delta = pd.read_sql_query(delta_query, conn)
            df = pd.concat([df_buffer, df_delta]).drop_duplicates(subset=['tick_time'])
            logger.info(f"⚡ DELTA SYNC: Fetched {len(df_delta)} new rows (+200 buffer).")
        else:
            query = f"SELECT * FROM market_data WHERE symbol = '{symbol}' ORDER BY tick_time ASC{limit_clause}"
            df = pd.read_sql_query(query, conn)
            
        conn.close()
        if 'tick_time' in df.columns:
            df['tick_time'] = pd.to_datetime(df['tick_time'])
            df.set_index('tick_time', inplace=True)

    if df.empty:
        logger.error("❌ No data found.")
        return None, None, None, None
        
    # [FIX] Alias volume to tick_volume (MT5 compatibility)
    if 'volume' in df.columns and 'tick_volume' not in df.columns:
        df['tick_volume'] = df['volume']

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
        # Map and sanitize: Fill unknowns with HOLD (1), Force Integer
        df['label'] = df['label'].map({-1.0: 0, 0.0: 1, 1.0: 2})
        df['label'] = df['label'].fillna(1).astype(int)
        
        # Verify Integrity
        unique_labels = df['label'].unique()
        logger.info(f"✅ Label Integrity Check: Found classes {unique_labels}")
        if not set(unique_labels).issubset({0, 1, 2}):
             logger.warning(f"⚠️ FOUND INVALID LABELS: {unique_labels}. Clamping...")
             df['label'] = df['label'].clip(0, 2)

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
            float(row.get('struct_age', 0)), 
            float(row.get('fvg_bull', 0)) - float(row.get('fvg_bear', 0)),
            float(row.get('layer_alignment', 0)),
            float(row.get('is_inducement', 0))
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
            float(row.get('physics_energy', 0)),
            float(row.get('physics_force', 0)),
            float(row.get('physics_entropy', 0)),
            float(row.get('physics_viscosity', 0))
        ])
        if len(static_feats["accel"]) == 1:
            print(f"DEBUG: First accel row: {static_feats['accel'][0]}")
        
        static_feats["time"].append([
            float(row.get('volatility_context', 1.0)),
            float(row.get('killzone_intensity', 0.5)),
            float(row.get('session_weight', 1.0)),
            float(row.get('day_of_week', 0)) # Added 4th column for FeatEncoder parity
        ])

    # [V6.1.1] Stitching Slicer: Return only samples >= since_time
    if since_time is not None:
        # We need to find the first index in the multi-channel processed df that matches or exceeds since_time
        # But we must be careful: FeatProcessor might drop some initial rows.
        # We'll use the tick_time index for robust slicing.
        mask = df.index >= pd.to_datetime(since_time)
        # However, sequences/labels/static_feats are built from the slidding window starting at seq_len
        # We need to align the sliding window indices with the mask.
        
        # New robust logic:
        # sequences[i] corresponds to df.iloc[i + seq_len]
        # So we only keep sequences[i] if df.index[i + seq_len] > since_time
        # (Using > since_time because since_time was the LAST sample in the previous cache)
        pivot_time = pd.to_datetime(since_time)
        valid_indices = []
        for i in range(len(sequences)):
            # The label and static features at index i correspond to the row at df index seq_len + i
            row_time = df.index[seq_len + i]
            if row_time > pivot_time:
                valid_indices.append(i)
        
        if not valid_indices:
            logger.warning("⚠️ No new samples found after stitching slice.")
            return np.array([]), np.array([]), np.array([]), {k: np.array([]) for k in static_feats}

        X = np.array([sequences[i] for i in valid_indices])
        y = np.array([labels[i] for i in valid_indices])
        phys = np.array([physics_vals[i] for i in valid_indices])
        new_static = {}
        for k in static_feats:
            new_static[k] = np.array([static_feats[k][i] for i in valid_indices])
        return X, y, phys, new_static

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
    parser.add_argument("--limit", type=int, default=0, help="Limit rows for smoke testing")
    parser.add_argument("--fvg-dual-mode", action="store_true", help="Enable FVG Dual-Objective Loss")
    parser.add_argument("--fractal-time-sync", action="store_true", help="Sync Fractal Time features")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 batch only for verification")
    parser.add_argument("--force-rehydration", action="store_true", help="Ignore cache and re-process entire dataset")
    args = parser.parse_args()
    print(f"DEBUG: Args parsed: {args}")

    # [D.A.L.C. INTEGRATION]
    print("\n[D.A.L.C.] INVOKING AUDITOR...")
    import subprocess
    audit_result = subprocess.run(["python", "tests/dalc_sanity_check.py"])
    
    if audit_result.returncode != 0:
        print("[!] TRAINING ABORTED BY D.A.L.C. (Logic Errors Detected).")
        print("Check the logs above to fix the Silent Killers.")
        sys.exit(1)
        
    print("[+] AUDIT PASSED. IGNITING TRAINING ENGINE.\n")
    
    if args.real:
        # [SMOKE TEST OVERRIDE] Force batch_size=32 for GPU stability
        smoke_batch_size = 32 if args.limit else args.batch_size
        
        temp_path = "data/temp_real_train.npz"
        do_full_hydration = True
        cache_data = None
        
        if os.path.exists(temp_path) and not args.force_rehydration:
            try:
                cache_data = np.load(temp_path, allow_pickle=True)
                # Check for last_time metadata
                if 'last_time' in cache_data:
                    cache_time = str(cache_data['last_time'])
                    logger.info(f"⚡ CACHE FOUND: Last processed time: {cache_time}")
                    
                    # Check DB for newer data
                    import sqlite3
                    conn = sqlite3.connect("data/market_data.db")
                    db_max = pd.read_sql_query(f"SELECT MAX(tick_time) as mt FROM market_data WHERE symbol = '{args.symbol}'", conn).iloc[0]['mt']
                    conn.close()
                    
                    if db_max and str(db_max) > cache_time:
                        logger.info(f"🚀 INCREMENTAL SYNC: New data detected ({db_max}).")
                        X_new, y_new, phys_new, static_new = load_real_data(args.symbol, since_time=cache_time)
                        
                        if X_new is not None and len(X_new) > 0:
                            logger.info(f"🔗 MERGING: Appending {len(X_new)} new samples to cache...")
                            X = np.concatenate([cache_data['X'], X_new])
                            y = np.concatenate([cache_data['y'], y_new])
                            phys_tensor = np.concatenate([cache_data['physics'], phys_new])
                            
                            static_feats = {}
                            cached_static = cache_data['static_features'].item()
                            for k in cached_static:
                                static_feats[k] = np.concatenate([cached_static[k], static_new[k]])
                            
                            # Update last_time to db_max
                            last_time = db_max
                            do_full_hydration = False
                        else:
                            logger.info("✅ Cache is already up to date (Delta was empty).")
                            do_full_hydration = False
                    else:
                        logger.info("✅ Cache is up to date. Using JUMPSTART.")
                        do_full_hydration = False
                else:
                    logger.warning("⚠️ Legacy cache found (No last_time). Forcing full hydration...")
            except Exception as e:
                logger.error(f"❌ Cache error: {e}. Forcing full hydration.")
        
        if do_full_hydration:
            logger.info(f"🧪 Hydrating Structural Narrative (SMC Stack)...")
            X, y, phys_tensor, static_feats = load_real_data(args.symbol, limit=args.limit if args.limit > 0 else None)
            
            # Fetch max time for metadata
            import sqlite3
            conn = sqlite3.connect("data/market_data.db")
            last_time = pd.read_sql_query(f"SELECT MAX(tick_time) as mt FROM market_data WHERE symbol = '{args.symbol}'", conn).iloc[0]['mt']
            conn.close()

        # Save/Update Cache
        if not do_full_hydration or (X is not None and len(X) > 0):
            np.savez_compressed(
                temp_path, 
                X=X if not cache_data or do_full_hydration else cache_data['X'], 
                y=y if not cache_data or do_full_hydration else cache_data['y'], 
                physics=phys_tensor if not cache_data or do_full_hydration else cache_data['physics'], 
                static_features=static_feats if not cache_data or do_full_hydration else cache_data['static_features'],
                last_time=last_time if not cache_data or do_full_hydration else cache_data['last_time']
            )
            logger.info(f"✅ Cache updated at {temp_path}.")

        train_hybrid_model(
            args.symbol, 
            temp_path, 
            epochs=args.epochs, 
            batch_size=smoke_batch_size,
            fvg_dual=args.fvg_dual_mode,
            fractal_sync=args.fractal_time_sync
        )
