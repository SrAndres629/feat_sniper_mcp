
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

from app.ml.models.hybrid_v1 import HybridSniper
from nexus_training.loss import PhysicsAwareLoss
from app.core.config import settings

# Logger Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HybridTrainer")

class SniperDataset(Dataset):
    """Dataset wrapper for FEAT Sniper data."""
    def __init__(self, sequences, labels, physics):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.physics = torch.FloatTensor(physics)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.physics[idx]

def train_hybrid_model(symbol: str, data_path: str, epochs=50, batch_size=32):
    """
    [LEVEL 25] Training Loop for Hybrid TCN-BiLSTM-Attention.
    Uses PhysicsAwareLoss to regularize predictions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Starting Hybrid Training for {symbol} on {device}")
    
    # 1. Load Data
    # For now, we assume data is pre-processed and saved as .npz or .pt
    # In production, this would call data_collector + feature_engineering
    if not os.path.exists(data_path):
        logger.error(f"Data file {data_path} not found.")
        return

    logger.info("Loading training data...")
    raw_data = np.load(data_path, allow_pickle=True)
    X = raw_data['X'] # (N, Seq, Feat)
    y = raw_data['y'] # (N,) Classes 0,1,2
    phys = raw_data['physics'] # (N,) Acceleration 0-1
    
    dataset = SniperDataset(X, y, phys)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Initialize Model
    input_dim = X.shape[2]
    model = HybridSniper(input_dim=input_dim, hidden_dim=128, num_classes=3).to(device)
    
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
        for seq, label, phys_val in progress:
            seq, label, phys_val = seq.to(device), label.to(device), phys_val.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            logits = model(seq) # (Batch, 3)
            
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
            save_path = os.path.join(settings.MODELS_DIR, f"hybrid_{symbol}_v1.pt")
            torch.save({
                "state_dict": model.state_dict(),
                "best_acc": acc,
                "config": {"input_dim": input_dim}
            }, save_path)
            logger.info(f"💾 Model checkpoint saved to {save_path}")

    logger.info("✅ Training Complete.")

if __name__ == "__main__":
    # Example Usage
    # Generate dummy data for testing the script structure
    try:
        # Dummy generation
        pass 
    except Exception as e:
        logger.error(f"Error: {e}")
