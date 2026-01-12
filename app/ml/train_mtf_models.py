"""
HIERARCHICAL MTF TRAINER (MIP v6.0)
==================================
Trains specialized models for different timeframes and roles.
Roles:
- sniper: Precision models (M1, M5)
- strategist: Bias models (H1, H4, D1)
"""

import os
import sys
import logging
import joblib
import sqlite3
import numpy as np
import pandas as pd
import argparse
from typing import Tuple, List, Dict, Any
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.core.config import settings
from app.ml.data_collector import FEATURE_NAMES, DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("QuantumLeap.MTFTrainer")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data_mtf(symbol: str, timeframe: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads multitemporal data from institutional schema."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
        
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    query = f"""
        SELECT {', '.join(FEATURE_NAMES)}, label
        FROM market_data
        WHERE symbol = ? AND timeframe = ? AND label IS NOT NULL
        ORDER BY tick_time ASC
    """
    
    cursor = conn.execute(query, (symbol, timeframe))
    rows = cursor.fetchall()
    conn.close()
    
    if len(rows) < 500:
        logger.warning(f" Insufficient data for {symbol} {timeframe} ({len(rows)} samples). Using bootstrap data.")
        # Synthetic cold start for demo/initialization
        X = np.random.randn(500, len(FEATURE_NAMES)).astype(np.float32)
        y = np.random.randint(0, 2, 500).astype(np.int64)
        return X, y
        
    X = np.array([[row[k] for k in FEATURE_NAMES] for row in rows], dtype=np.float32)
    y = np.array([row["label"] for row in rows], dtype=np.int64)
    
    return X, y

def train_sniper(symbol: str, timeframe: str):
    """Trains a precision GBM model for entry timing."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import log_loss
    
    logger.info(f" Training SNIPER model for {symbol} [{timeframe}]")
    X, y = load_data_mtf(symbol, timeframe)
    
    model = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42
    )
    model.fit(X, y)
    
    path = os.path.join(MODELS_DIR, f"gbm_{symbol}_{timeframe}_sniper.joblib")
    joblib.dump({
        "model": model,
        "features": FEATURE_NAMES,
        "timeframe": timeframe,
        "role": "sniper",
        "trained_at": datetime.utcnow().isoformat()
    }, path)
    
    logger.info(f" Sniper model saved: {path}")

def train_strategist(symbol: str, timeframe: str):
    """Trains a structural bias model for trend identification."""
    from sklearn.ensemble import RandomForestClassifier
    
    logger.info(f" Training STRATEGIST model for {symbol} [{timeframe}]")
    X, y = load_data_mtf(symbol, timeframe)
    
    # Strategists use more robust, deeper trees with higher sampling
    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=20, random_state=42
    )
    model.fit(X, y)
    
    path = os.path.join(MODELS_DIR, f"rf_{symbol}_{timeframe}_strategist.joblib")
    joblib.dump({
        "model": model,
        "features": FEATURE_NAMES,
        "timeframe": timeframe,
        "role": "strategist",
        "trained_at": datetime.utcnow().isoformat()
    }, path)
    
    logger.info(f" Strategist model saved: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FEAT MTF Hierarchical Trainer")
    parser.add_argument("--symbol", type=str, default=settings.SYMBOL)
    parser.add_argument("--tf", type=str, default="M1", choices=["M1", "M5", "M15", "H1", "H4", "D1"])
    parser.add_argument("--role", type=str, default="sniper", choices=["sniper", "strategist"])
    
    args = parser.parse_args()
    
    if args.role == "sniper":
        train_sniper(args.symbol, args.tf)
    else:
        train_strategist(args.symbol, args.tf)
