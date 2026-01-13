"""
ML Model Trainer - Quantum Leap Phase 2 & 3
============================================
Entrenamiento de modelos GBM (tabular) y LSTM (temporal).

Pipeline:
- Fase 2: GradientBoosting con TimeSeriesSplit
- Fase 3: LSTM con Attention para patrones secuenciales
"""

import os
import sys
import csv
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

# Feature columns (Multifractal Physics Layers)
FEATURE_NAMES = [
    'L1_Mean', 'L1_Width', 'L4_Slope', 'Div_L1_L2'
]


# ARGUMENT PARSING
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--quick_run", action="store_true")
args, unknown = parser.parse_known_args()

# FEATURE ENGINEERING
from app.skills.indicators import calculate_feat_layers

def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Generates Multifractal Features and returns X, y."""
    # 1. Calculate Physical Layers
    df_feat = calculate_feat_layers(df)
    
    # 2. Select Features (The Tensors)
    feature_cols = [
        'L1_Mean', 'L1_Width', 'L4_Slope', 'Div_L1_L2'
    ]
    
    # Merge features back to original df to preserve 'label'
    df = pd.concat([df, df_feat], axis=1)
    
    # Check for NaNs created by EMAs/Diff
    df = df.dropna(subset=feature_cols + ['label'])
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.int64)
    
    logger.info(f"Generated {X.shape[0]} samples with {X.shape[1]} features (Multifractal Physics).")
    return X, y

def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads CSV and computes features."""
    df = pd.read_csv(path)
    return prepare_features(df)

def load_from_sqlite(db_path: str = "data/market_data.db") -> Tuple[np.ndarray, np.ndarray]:
    """Loads raw OHLC from SQLite and computes features."""
    import sqlite3
    
    if not os.path.exists(db_path):
        # Fallback to dummy data if DB missing (Genesis Mode)
        logger.warning(f"Database {db_path} not found. GENERATING DUMMY DATA (Genesis Protocol).")
        dates = pd.date_range(end=datetime.now(), periods=2500, freq='1min')
        df = pd.DataFrame({
            'close': np.random.normal(2000, 10, 2500).cumsum(), # Random walk
            'tick_time': dates
        })
        df['open'] = df['close'] + np.random.normal(0, 1, 2500)
        df['high'] = df['close'] + 2
        df['low'] = df['close'] - 2
        df['label'] = np.random.randint(0, 2, 2500)
        return prepare_features(df)
        
    conn = sqlite3.connect(db_path)
    query = "SELECT tick_time, open, high, low, close, volume, label FROM market_data ORDER BY tick_time"
    df = pd.read_sql(query, conn)
    conn.close()
    
    if df.empty:
         raise ValueError("Database empty.")
         
    return prepare_features(df)


# =============================================================================
# FASE 2: GRADIENT BOOSTING (TABULAR)
# =============================================================================

def train_gbm(X: np.ndarray, y: np.ndarray) -> str:
    """
    Entrena GradientBoostingClassifier con TimeSeriesSplit.
    
    Objetivo: Minimizar LogLoss para clasificacin binaria.
    Validacin: Walk-forward para evitar look-ahead bias.
    
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
    
    # TimeSeriesSplit para validacin temporal
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Impute NaNs
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Normalizacin
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    best_loss = float("inf")
    best_model = None
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Modelo con hiperparmetros optimizados para trading
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
        
        # Evaluacin
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
    model_path = os.path.join(MODELS_DIR, f"gbm_{SYMBOL}_v2.joblib")
    
    joblib.dump({
        "model": best_model,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "trained_at": datetime.utcnow().isoformat(),
        "best_logloss": best_loss,
        "n_samples": len(X),
        "cv_results": fold_results
    }, model_path)
    
    logger.info(f" GBM Model saved to {model_path}")
    logger.info(f"   Best LogLoss: {best_loss:.5f}")
    
    return model_path


# =============================================================================
# FASE 3: LSTM CON ATENCIN (TEMPORAL)
# =============================================================================

def train_lstm(X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN) -> str:
    """
    Entrena LSTM con Attention para capturar patrones secuenciales.
    
    Arquitectura:
    - LSTM bidireccional con 2 capas
    - Self-Attention para ponderar velas relevantes
    - Clasificacin binaria (WIN/LOSS)
    
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
    
    # Model init
    input_dim = X.shape[1]
    model = LSTMWithAttention(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        num_classes=2
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_acc = 0.0
    best_model_state = None
    
    # Training Loop
    logger.info("Starting LSTM training...")
    for epoch in range(20): # 20 Epochs
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
        logger.info(f"Epoch {epoch+1}: Loss={(train_loss/len(train_loader)):.4f}, ValAcc={val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict()
            
    # Save Model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"lstm_{SYMBOL}_v2.pt")
    
    torch.save({
        "model_state": best_model_state,
        "model_config": {
            "input_dim": input_dim,
            "hidden_dim": 64,
            "num_layers": 2,
            "seq_len": seq_len
        },
        "trained_at": datetime.utcnow().isoformat(),
        "best_acc": best_acc
    }, model_path)
    
    logger.info(f" LSTM Model saved to {model_path} (Acc: {best_acc:.4f})")
    return model_path


# =============================================================================
# EXPORTED CLASSES
# =============================================================================

import torch
import torch.nn as nn

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
        
        # Weighted sum (Context vector)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        # Classification
        logits = self.classifier(context)
        return logits


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
        logger.warning(f" Dataset not found at {DATA_PATH}")
        logger.info(" Generating SYNTHETIC COLD-START DATA to initialize models...")
        # Generate 1000 random samples (Cold Start)
        import numpy as np
        X = np.random.randn(MIN_SAMPLES, len(FEATURE_NAMES)).astype(np.float32)
        y = np.random.randint(0, 2, MIN_SAMPLES).astype(np.int64)

    logger.info(f"Loaded Dataset Shape: X={X.shape}, y={y.shape}")

    if len(X) < MIN_SAMPLES:
        logger.warning(f" Insufficient data ({len(X)} < {MIN_SAMPLES}). Attempting training anyway (Risk Mode)...")
    
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
    logger.info(" Training Complete!")
    logger.info(f"   GBM: {gbm_path}")
    logger.info(f"   LSTM: {lstm_path}")
    logger.info("=" * 60)
    
    return result


if __name__ == "__main__":
    train_all()
