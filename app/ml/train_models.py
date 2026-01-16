"""
FEAT ML Training Pipeline - Production Grade
=============================================
Trains LSTM and LightGBM models for market prediction.
Uses causal normalization to prevent data leakage.
"""

import os
import logging
import numpy as np
from datetime import datetime
from typing import Tuple, Optional

logger = logging.getLogger("feat.ml.train")

# Configuration
MODELS_DIR = "models"
SYMBOL = "XAUUSD"
SEQ_LEN = 50


class LSTMWithAttention:
    """
    LSTM with self-attention mechanism for sequential patterns.
    Proper PyTorch implementation with causal scaling.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 2):
        import torch
        import torch.nn as nn
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Build model
        self.model = self._build_model()
        
    def _build_model(self):
        import torch.nn as nn
        
        class LSTMNet(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_dim, hidden_dim, num_layers,
                    batch_first=True, dropout=0.2
                )
                self.attention = nn.Linear(hidden_dim, 1)
                self.fc = nn.Linear(hidden_dim, num_classes)
                
            def forward(self, x):
                import torch.nn.functional as F
                lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
                
                # Self-attention
                attn_weights = F.softmax(self.attention(lstm_out), dim=1)
                context = (lstm_out * attn_weights).sum(dim=1)
                
                return self.fc(context)
        
        return LSTMNet(
            self.input_dim, 
            self.hidden_dim, 
            self.num_layers, 
            self.num_classes
        )


def train_lstm(X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN) -> str:
    """
    Trains LSTM with Attention for sequential pattern recognition.
    
    Uses CAUSAL NORMALIZATION: Scaler is fit only on training data (past),
    then applied to all data. This prevents data leakage.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        seq_len: Sequence length for LSTM input
        
    Returns:
        Path to saved model file
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    
    logger.info("=" * 60)
    logger.info("PHASE: Training LSTM with Attention (Causal Scaling)")
    logger.info("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 1. CAUSAL NORMALIZATION - Fit scaler ONLY on training portion
    split_idx = int(len(X) * 0.8)
    X_train_raw = X[:split_idx]
    
    scaler = StandardScaler()
    scaler.fit(X_train_raw)  # Fit only on past data
    X_scaled = scaler.transform(X)  # Apply to all
    
    logger.info(f"Scaler fitted on first {split_idx} samples (causal).")
    
    # 2. Create sequences for LSTM
    def make_sequences(X_in: np.ndarray, y_in: np.ndarray, seq_len: int) -> Tuple:
        """Convert tabular data to 3D sequences."""
        seqs, labels = [], []
        for i in range(seq_len, len(X_in)):
            seqs.append(X_in[i - seq_len:i, :])
            labels.append(y_in[i])
        return (
            torch.tensor(np.stack(seqs), dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
        )
    
    X_seq, y_seq = make_sequences(X_scaled, y, seq_len)
    logger.info(f"Created {len(X_seq)} sequences of length {seq_len}")
    
    # 3. Temporal split (80/20)
    train_size = int(0.8 * len(X_seq))
    X_train, X_val = X_seq[:train_size], X_seq[train_size:]
    y_train, y_val = y_seq[:train_size], y_seq[train_size:]
    
    # 4. DataLoaders (NO SHUFFLE for time series)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64, shuffle=False
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=64, shuffle=False
    )
    
    # 5. Initialize model
    input_dim = X.shape[1]
    lstm_wrapper = LSTMWithAttention(input_dim=input_dim)
    model = lstm_wrapper.model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0.0
    best_model_state = None
    
    # 6. Training loop
    logger.info("Starting LSTM training...")
    for epoch in range(20):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = model(batch_X)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == batch_y).sum().item()
                total_val += batch_y.size(0)
        
        val_acc = val_correct / total_val if total_val > 0 else 0
        logger.info(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, ValAcc={val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict()
    
    # 7. Save model with scaler stats
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"lstm_{SYMBOL}_v2.pt")
    
    scaler_stats = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }
    
    torch.save({
        "state_dict": best_model_state,
        "scaler_stats": scaler_stats,
        "model_config": {
            "input_dim": input_dim,
            "hidden_dim": 64,
            "num_layers": 2,
            "seq_len": seq_len
        },
        "trained_at": datetime.utcnow().isoformat(),
        "best_acc": best_acc
    }, model_path)
    
    logger.info(f"✅ LSTM Model saved to {model_path} (Acc: {best_acc:.4f})")
    return model_path


def train_lightgbm(X: np.ndarray, y: np.ndarray) -> str:
    """
    Trains LightGBM classifier with time-series cross-validation.
    
    Args:
        X: Feature matrix
        y: Labels
        
    Returns:
        Path to saved model file
    """
    import joblib
    from sklearn.model_selection import TimeSeriesSplit
    
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("LightGBM not installed. Run: pip install lightgbm")
        raise
    
    logger.info("=" * 60)
    logger.info("PHASE: Training LightGBM (TimeSeriesSplit CV)")
    logger.info("=" * 60)
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1
    }
    
    best_model = None
    best_score = 0.0
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10)]
        )
        
        # Evaluate
        preds = (model.predict(X_val) > 0.5).astype(int)
        acc = (preds == y_val).mean()
        logger.info(f"Fold {fold+1}: Accuracy={acc:.4f}")
        
        if acc > best_score:
            best_score = acc
            best_model = model
    
    # Save best model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"lgbm_{SYMBOL}.joblib")
    
    joblib.dump({
        "model": best_model,
        "trained_at": datetime.utcnow().isoformat(),
        "best_acc": best_score
    }, model_path)
    
    logger.info(f"✅ LightGBM Model saved to {model_path} (Acc: {best_score:.4f})")
    return model_path


if __name__ == "__main__":
    # Example usage - requires feature data
    logging.basicConfig(level=logging.INFO)
    logger.info("ML Training Pipeline Ready")
    logger.info("Usage: train_lstm(X, y) or train_lightgbm(X, y)")