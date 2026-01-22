import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import logging
import argparse
import torch._dynamo # [V6.2.2] Import necesario para control de errores

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | [%(name)s] | %(levelname)s | %(message)s')
logger = logging.getLogger("QuantumForge")

# [V6.2.2] SUPRESIÓN DE ERRORES DE COMPILADOR (Para Windows sin C++)
torch._dynamo.config.suppress_errors = True

# Añadir rutas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from app.core.config import settings
    from app.ml.models.hybrid_probabilistic import HybridProbabilistic
    from nexus_training.loss import SovereignQuantLoss
except ImportError:
    # Fallback para ejecución directa
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from app.core.config import settings
    from app.ml.models.hybrid_probabilistic import HybridProbabilistic
    from nexus_training.loss import SovereignQuantLoss

# --- HELPER: CLEAN STATE DICT [V6.2.2] ---
def clean_state_dict(state_dict):
    """
    Limpia prefijos de compilación (_orig_mod, module) para evitar 'Missing Keys'.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        # Eliminar prefijos de TorchCompile o DDP
        if name.startswith('_orig_mod.'):
            name = name[10:]
        elif name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict

# --- QUANTUM CURRICULUM MANAGER ---
class QuantumCurriculum:
    def __init__(self, total_epochs=50):
        self.total_epochs = total_epochs
        self.p1 = int(total_epochs * 0.3)
        self.p2 = int(total_epochs * 0.7)

    def get_phase_mask(self, epoch):
        if epoch < self.p1:
            return {"kinetic": 0.0, "spatial": 0.1, "alpha": 0.0, "name": "PHASE 1: STRUCTURAL VISION"}
        elif epoch < self.p2:
            return {"kinetic": 0.5, "spatial": 0.5, "alpha": 0.2, "name": "PHASE 2: MARKET DYNAMICS"}
        else:
            return {"kinetic": 1.0, "spatial": 1.0, "alpha": 1.0, "name": "PHASE 3: ELITE ALPHA"}

# --- DATASET CON SOFT LABELS ---
class SniperDataset(Dataset):
    def __init__(self, X, y, physics, static_feats, soft_label_smoothing=0.1):
        self.X = torch.FloatTensor(X)
        y_indices = torch.LongTensor(y)
        num_classes = 3
        y_soft = torch.zeros(len(y), num_classes)
        y_soft.scatter_(1, y_indices.view(-1, 1), 1)
        y_soft = y_soft * (1.0 - soft_label_smoothing) + (soft_label_smoothing / (num_classes - 1))
        self.y = y_soft
        self.physics = torch.FloatTensor(physics)
        
        if isinstance(static_feats, dict):
            self.form = torch.FloatTensor(static_feats.get('formation_vector', np.zeros((len(X), 10))))
            self.space = torch.FloatTensor(static_feats.get('spatial_grid', np.zeros((len(X), 50))))
            self.accel = torch.FloatTensor(static_feats.get('acceleration_matrix', np.zeros((len(X), 4))))
            self.time = torch.FloatTensor(static_feats.get('temporal_encoding', np.zeros((len(X), 2))))
            self.kin = torch.FloatTensor(static_feats.get('kinetic_energy', np.zeros((len(X), 1))))
        else:
            self.form = torch.zeros((len(X), 10))
            self.space = torch.zeros((len(X), 50))
            self.accel = torch.zeros((len(X), 4))
            self.time = torch.zeros((len(X), 2))
            self.kin = torch.zeros((len(X), 1))
        self.spatial_map = torch.zeros((len(X), 1, 50, 50))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx], self.y[idx], self.physics[idx],
            self.form[idx], self.space[idx], self.accel[idx],
            self.time[idx], self.kin[idx], self.spatial_map[idx]
        )

# --- CARGA DE DATOS ---
def load_real_data(symbol: str, limit=0):
    data_path = "data/temp_real_train.npz"
    if os.path.exists(data_path):
        logger.info(f"⚡ CACHE FOUND: Loading {data_path}...")
        try:
            data = np.load(data_path, allow_pickle=True)
            X = data['X']
            y = data['y']
            if 'physics' in data: physics = data['physics']
            elif 'phys' in data: physics = data['phys']
            else: physics = np.zeros((len(X), 6))
            
            if 'static_features' in data and data['static_features'].shape == ():
                static_feats = data['static_features'].item()
            else: static_feats = {}

            if limit > 0:
                logger.warning(f"⚠️ LIMIT APPLIED: Truncating dataset to {limit} rows.")
                X, y, physics = X[:limit], y[:limit], physics[:limit]
                for k, v in static_feats.items():
                    if hasattr(v, '__len__') and len(v) == len(data['X']):
                        static_feats[k] = v[:limit]
            
            return X, y, physics, static_feats
        except Exception as e:
            logger.error(f"Cache Error: {e}")
            raise e
    raise FileNotFoundError("Cache not found. Run hydration first.")

# --- MOTOR DE ENTRENAMIENTO ---
def train_hybrid_model(symbol: str, data_path: str, epochs=50, batch_size=64, force_cpu=False, **kwargs):
    """
    [V6.2.2 CORTEX HARDENING] - Robust Compilation & State Management
    """
    limit = kwargs.get('limit', 0)
    dry_run = kwargs.get('dry_run', False)

    # 1. HARDWARE SELECTION
    use_amp = True
    if force_cpu:
        device = torch.device("cpu")
        scaler_device = 'cpu'
        logger.info(f"🛡️ I9 SOVEREIGN MODE | Device: CPU (AMP Enabled) | Batch: {batch_size}")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            scaler_device = 'cuda'
            logger.info(f"🛡️ QUANTUM GPU MODE | Device: RTX 3060 (AMP Enabled) | Batch: {batch_size}")
        else:
            device = torch.device("cpu")
            scaler_device = 'cpu'
            logger.info(f"⚠️ CPU FALLBACK | Device: CPU (AMP Enabled) | Batch: {batch_size}")

    # 2. DATA LOAD
    try:
        X, y, phys_tensor, static_feats = load_real_data(symbol, limit=limit)
        logger.info(f"📊 Dataset Size: {len(X)}")
    except Exception as e:
        logger.error(f"Data Load Failed: {e}")
        return

    dataset = SniperDataset(X, y, phys_tensor, static_feats, soft_label_smoothing=0.1)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    workers = 0 
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=workers, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=workers, pin_memory=(device.type == 'cuda')
    )

    # 3. MODEL SETUP
    input_dim = X.shape[2]
    model = HybridProbabilistic(input_dim=input_dim, hidden_dim=settings.NEURAL_HIDDEN_DIM, num_classes=3).to(device)
    
    # [V6.2.2] ROBUST COMPILATION FALLBACK
    if not dry_run:
        try:
            logger.info("🔥 Compiling Neural Engine (Quantum Optimization)...")
            # En Windows sin VC++, esto suele fallar con Inductor.
            # El try/except + suppress_errors permite continuar en modo 'Eager' si falla.
            model = torch.compile(model) 
        except Exception as e:
            logger.warning(f"⚠️ Compiler Failed (Missing cl.exe?): {e}. Running in Eager Mode.")

    # 4. CURRICULUM & OPTIMIZER
    curriculum = QuantumCurriculum(total_epochs=epochs)
    criterion = SovereignQuantLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=settings.NEURAL_LEARNING_RATE, weight_decay=settings.NEURAL_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    scaler = torch.amp.GradScaler(scaler_device, enabled=use_amp)

    if not dry_run:
        try:
            import wandb
            if wandb.run is None:
                wandb.init(project=f"FEAT_SNIPER_{symbol}", name=f"Quantum_I9_B{batch_size}", reinit=True)
        except: pass

    # [V6.2.2] ROBUST AUTO-RESUME
    start_epoch = 0
    checkpoint_dir = "models/checkpoints"
    if not dry_run and os.path.exists(checkpoint_dir):
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
        if checkpoints:
            latest = checkpoints[-1]
            try:
                state = torch.load(os.path.join(checkpoint_dir, latest), map_location=device)
                
                # --- STATE DICT CLEANING ---
                if isinstance(state, dict) and 'model_state_dict' in state:
                    clean_state = clean_state_dict(state['model_state_dict'])
                    model.load_state_dict(clean_state, strict=False) # Strict=False para tolerar cambios menores
                    if 'optimizer_state_dict' in state:
                        optimizer.load_state_dict(state['optimizer_state_dict'])
                    start_epoch = state['epoch']
                else:
                    # Legacy load
                    clean_state = clean_state_dict(state)
                    model.load_state_dict(clean_state, strict=False)
                    
                logger.info(f"🔄 RESUMED from {latest} (Epoch {start_epoch})")
            except Exception as e:
                logger.warning(f"⚠️ Could not resume from {latest}: {e}. Starting fresh.")

    # 5. TRAINING LOOP
    logger.info(f"⚔️ IGNITING FORGE | BATCH: {batch_size} | EPOCHS: {epochs}")

    for epoch in range(start_epoch, epochs):
        phase_config = curriculum.get_phase_mask(epoch)
        model.train()
        train_loss = 0.0
        
        progress = tqdm(train_loader, desc=f"Ep {epoch+1} [{phase_config['name'][:10]}]", leave=False)
        
        for batch in progress:
            seq = batch[0].to(device)
            label = batch[1].to(device)
            
            # Aux Feats
            f_form = batch[3].to(device)
            f_space = batch[4].to(device)
            f_accel = batch[5].to(device)
            f_time = batch[6].to(device)
            f_kin = batch[7].to(device)
            f_map = batch[8].to(device)

            feat_input = { "form": f_form, "space": f_space, "accel": f_accel,
                           "time": f_time, "kinetic": f_kin, "spatial_map": f_map }
            
            p_tensor = torch.stack([f_accel[:, 0], f_accel[:, 1], f_accel[:, 2], f_accel[:, 3],
                                    f_time[:, 0], f_time[:, 1]], dim=1)

            optimizer.zero_grad()

            amp_dev = 'cuda' if device.type == 'cuda' else 'cpu'
            with torch.amp.autocast(device_type=amp_dev, enabled=use_amp):
                outputs = model(seq, feat_input=feat_input, physics_tensor=p_tensor)
                logits = outputs["logits"]
                loss = criterion(logits, label, p_tensor, x_map=f_map, alpha=1.0, phase_mask=phase_config)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
             for batch in val_loader:
                seq = batch[0].to(device)
                label = batch[1].to(device)
                f_form = batch[3].to(device); f_space = batch[4].to(device); f_accel = batch[5].to(device)
                f_time = batch[6].to(device); f_kin = batch[7].to(device); f_map = batch[8].to(device)
                feat_input = { "form": f_form, "space": f_space, "accel": f_accel, "time": f_time, "kinetic": f_kin, "spatial_map": f_map }
                p_tensor = torch.stack([f_accel[:, 0], f_accel[:, 1], f_accel[:, 2], f_accel[:, 3], f_time[:, 0], f_time[:, 1]], dim=1)
                
                with torch.amp.autocast(device_type=amp_dev, enabled=use_amp):
                    outputs = model(seq, feat_input=feat_input, physics_tensor=p_tensor)
                    loss = criterion(outputs["logits"], label, p_tensor, x_map=f_map, phase_mask=phase_config)
                val_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        logger.info(f"🏁 Ep {epoch+1} | Loss: {avg_loss:.4f} | Val: {val_loss/len(val_loader):.4f}")
        
        if not dry_run:
            if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, f"{checkpoint_dir}/epoch_{epoch+1}.pt")
            scheduler.step(avg_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Forge Training")
    parser.add_argument("--symbol", type=str, default="XAUUSD")
    parser.add_argument("--data_path", type=str, default="data/temp_real_train.npz")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Limit dataset size for debugging")
    parser.add_argument("--dry-run", action="store_true", help="Run without saving")
    # Legacy args
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--fvg-dual-mode", action="store_true")
    parser.add_argument("--fractal-time-sync", action="store_true")
    parser.add_argument("--force-rehydration", action="store_true")
    
    args = parser.parse_args()
    
    train_hybrid_model(
        symbol=args.symbol,
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        force_cpu=not args.gpu,
        limit=args.limit,
        dry_run=args.dry_run
    )
