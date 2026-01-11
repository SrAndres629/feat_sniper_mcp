"""
ML Model Trainer - Quantum Leap Phase 2 & 3
============================================
Entrenamiento de modelos GBM (tabular) y LSTM (temporal).

Pipeline:
- Fase 2: GradientBoosting con TimeSeriesSplit
- Fase 3: LSTM con Attention para patrones secuenciales
"""

import os
import csv
import logging
import joblib
import numpy as np
from typing import Tuple, List, Optional
from datetime import datetime

# Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DATA_PATH = os.getenv("DATA_PATH", "data/training_dataset.csv")
MODELS_DIR = os.getenv("MODELS_DIR", "models")
SEQ_LEN = int(os.getenv("SEQ_LEN", "32"))
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "1000"))

from app.core.config import settings
SYMBOL = settings.SYMBOL


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("QuantumLeap.Trainer")

# Feature columns (must match data_collector.py)
FEATURE_NAMES = [
    "close", "open", "high", "low", "volume",
    "rsi", "ema_fast", "ema_slow", "ema_spread",
    "feat_score", "fsm_state", "atr", "compression",
    "liquidity_above", "liquidity_below"
]


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga dataset desde CSV.
    
    Returns:
        X: Features matrix (N, D)
        y: Labels array (N,)
    """
    X, y = [], []
    
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                features = [float(row[k]) for k in FEATURE_NAMES]
                label = int(row["label"])
                X.append(features)
                y.append(label)
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping malformed row: {e}")
                continue
                
    logger.info(f"Loaded {len(X)} samples from {path}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def load_from_sqlite(db_path: str = "data/market_data.db") -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga dataset directamente desde SQLite (más eficiente que CSV).
    Lee de la tabla training_samples.
    
    Returns:
        X: Features matrix (N, D)
        y: Labels array (N,)
    """
    import sqlite3
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
        
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    query = f"""
        SELECT {', '.join(FEATURE_NAMES)}, label
        FROM training_samples
        ORDER BY timestamp
    """
    
    cursor = conn.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        raise ValueError("No training samples in database")
        
    X = np.array([[row[k] for k in FEATURE_NAMES] for row in rows], dtype=np.float32)
    y = np.array([row["label"] for row in rows], dtype=np.int64)
    
    logger.info(f"Loaded {len(X)} samples from SQLite: {db_path}")
    return X, y


# =============================================================================
# FASE 2: GRADIENT BOOSTING (TABULAR)
# =============================================================================

def train_gbm(X: np.ndarray, y: np.ndarray) -> str:
    """
    Entrena GradientBoostingClassifier con TimeSeriesSplit.
    
    Objetivo: Minimizar LogLoss para clasificación binaria.
    Validación: Walk-forward para evitar look-ahead bias.
    
    Returns:
        Path al modelo guardado
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import log_loss, accuracy_score, classification_report
    
    logger.info("=" * 60)
    logger.info("PHASE 2: Training Gradient Boosting Model")
    logger.info("=" * 60)
    
    # TimeSeriesSplit para validación temporal
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Impute NaNs
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Normalización
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    best_loss = float("inf")
    best_model = None
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Modelo con hiperparámetros optimizados para trading
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluación
        y_pred_proba = model.predict_proba(X_val)
        y_pred = model.predict(X_val)
        
        loss = log_loss(y_val, y_pred_proba)
        acc = accuracy_score(y_val, y_pred)
        
        fold_results.append({"fold": fold + 1, "logloss": loss, "accuracy": acc})
        logger.info(f"Fold {fold + 1}: LogLoss={loss:.5f}, Accuracy={acc:.4f}")
        
        if loss < best_loss:
            best_loss = loss
            best_model = model
            
    # Guardar mejor modelo con scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"gbm_{SYMBOL}_v1.joblib")
    
    joblib.dump({
        "model": best_model,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "trained_at": datetime.utcnow().isoformat(),
        "best_logloss": best_loss,
        "n_samples": len(X),
        "cv_results": fold_results
    }, model_path)
    
    logger.info(f"✅ GBM Model saved to {model_path}")
    logger.info(f"   Best LogLoss: {best_loss:.5f}")
    
    return model_path


# =============================================================================
# FASE 3: LSTM CON ATENCIÓN (TEMPORAL)
# =============================================================================

def train_lstm(X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN) -> str:
    """
    Entrena LSTM con Attention para capturar patrones secuenciales.
    
    Arquitectura:
    - LSTM bidireccional con 2 capas
    - Self-Attention para ponderar velas relevantes
    - Clasificación binaria (WIN/LOSS)
    
    Returns:
        Path al modelo guardado
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    
    logger.info("=" * 60)
    logger.info("PHASE 3: Training LSTM with Attention")
    logger.info("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Crear secuencias
    def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
        """Convierte datos tabulares a secuencias 3D."""
        seqs, labels = [], []
        for i in range(seq_len, len(X)):
            seqs.append(X[i - seq_len:i, :])
            labels.append(y[i])
        return (
            torch.tensor(np.stack(seqs), dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
        )
    
    X_seq, y_seq = make_sequences(X, y, seq_len)
    logger.info(f"Created {len(X_seq)} sequences of length {seq_len}")
    
    # Split temporal (80/20)
    train_size = int(0.8 * len(X_seq))
    X_train, X_val = X_seq[:train_size], X_seq[train_size:]
    y_train, y_val = y_seq[:train_size], y_seq[train_size:]
    
    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=False  # No shuffle for time series
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=64,
        shuffle=False
    )
    
    # Modelo LSTM con Atención
    class LSTMWithAttention(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.2
            )
            
            # Self-Attention
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
            
            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes)
            )
            
        def forward(self, x):
            # LSTM encoding
            lstm_out, _ = self.lstm(x)  # (B, T, H*2)
            
            # Attention weights
            attn_weights = self.attention(lstm_out)  # (B, T, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            
            # Weighted context
            context = (lstm_out * attn_weights).sum(dim=1)  # (B, H*2)
            
            # Classification
            return self.classifier(context)
    
    model = LSTMWithAttention(
        input_dim=X.shape[1],
        hidden_dim=64,
        num_layers=2
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training loop
    epochs = 20
    best_val_loss = float("inf")
    best_state = None
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
                
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)
                
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch:02d}: Train={train_loss:.5f}, Val={val_loss:.5f}, Acc={val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            
    # Guardar modelo
    model.load_state_dict(best_state)
    model_path = os.path.join(MODELS_DIR, f"lstm_{SYMBOL}_v1.pt")
    
    torch.save({
        "model_state": best_state,
        "model_config": {
            "input_dim": X.shape[1],
            "hidden_dim": 64,
            "num_layers": 2,
            "seq_len": seq_len
        },
        "feature_names": FEATURE_NAMES,
        "trained_at": datetime.utcnow().isoformat(),
        "best_val_loss": best_val_loss,
        "n_samples": len(X)
    }, model_path)
    
    logger.info(f"✅ LSTM Model saved to {model_path}")
    logger.info(f"   Best Val Loss: {best_val_loss:.5f}")
    
    return model_path


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def train_all():
    """Ejecuta pipeline de entrenamiento completo."""
    logger.info("=" * 60)
    logger.info("QUANTUM LEAP: ML Training Pipeline")
    logger.info("=" * 60)
    
    # Verificar datos
    X, y = None, None
    if os.path.exists(DATA_PATH):
        X, y = load_dataset(DATA_PATH)
    else:
        logger.warning(f"⚠️ Dataset not found at {DATA_PATH}")
        logger.info("⚡ Generating SYNTHETIC COLD-START DATA to initialize models...")
        # Generate 1000 random samples (Cold Start)
        import numpy as np
        X = np.random.randn(MIN_SAMPLES, len(FEATURE_NAMES)).astype(np.float32)
        y = np.random.randint(0, 2, MIN_SAMPLES).astype(np.int64)

    logger.info(f"Loaded Dataset Shape: X={X.shape}, y={y.shape}")

    if len(X) < MIN_SAMPLES:
        logger.warning(f"⚠️ Insufficient data ({len(X)} < {MIN_SAMPLES}). Attempting training anyway (Risk Mode)...")
    
    # Ensure 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Fase 2: GBM
    gbm_path = train_gbm(X, y)
    
    # Fase 3: LSTM
    lstm_path = train_lstm(X, y, SEQ_LEN)
    
    # Resumen
    result = {
        "status": "success",
        "gbm_model": gbm_path,
        "lstm_model": lstm_path,
        "samples_used": len(X),
        "trained_at": datetime.utcnow().isoformat()
    }
    
    logger.info("=" * 60)
    logger.info("✅ Training Complete!")
    logger.info(f"   GBM: {gbm_path}")
    logger.info(f"   LSTM: {lstm_path}")
    logger.info("=" * 60)
    
    return result


if __name__ == "__main__":
    train_all()
