"""
FEAT NEXUS: HYBRID TRAINING PIPELINE (RLAIF-Enabled)
=====================================================
Trains the HybridFEATNetwork using:
1. Real market data from SQLite/Supabase
2. Correction datasets from LLM feedback
3. Trigger retraining via MCP for continuous learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import sqlite3
import json
import os
import logging
from datetime import datetime, timezone
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional

# Import model
from nexus_brain.hybrid_model import HybridFEATNetwork, save_hybrid_model

logger = logging.getLogger("FEAT.Training")

# Configuration
DATA_DIR = os.getenv("DATA_DIR", "data")
DB_PATH = os.path.join(DATA_DIR, "market_data.db")
MODELS_DIR = "models"
CORRECTIONS_FILE = os.path.join(DATA_DIR, "llm_corrections.jsonl")


class FeatHybridDataset(Dataset):
    """
    Dataset for Hybrid Training with RLAIF support.
    Loads from real DB and incorporates LLM correction samples.
    """

    def __init__(self, db_path: str = DB_PATH, correction_weight: float = 2.0):
        """
        Args:
            db_path: Path to SQLite market data
            correction_weight: Weight multiplier for LLM correction samples
        """
        self.correction_weight = correction_weight
        self.samples = []

        # Load from database
        self._load_from_db(db_path)

        # Load LLM corrections (with higher weight)
        self._load_corrections()

        if not self.samples:
            logger.warning("No samples loaded. Using synthetic data as fallback.")
            self._generate_synthetic_fallback()

    def _load_from_db(self, db_path: str):
        """Load labeled samples from market_data database and calculate fractal features."""
        if not os.path.exists(db_path):
            logger.warning(f"Database not found: {db_path}")
            return

        try:
            conn = sqlite3.connect(db_path)

            # Fetch labeled samples with OHLC
            query = """
                SELECT tick_time, open, high, low, close, label
                FROM market_data
                WHERE label IS NOT NULL
                ORDER BY tick_time DESC
                LIMIT 10000
            """
            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                logger.warning("No labeled data in database.")
                return

            # Sort by time to ensure indicators are calculated correctly
            df = df.sort_values("tick_time")

            # Calculate Fractal Features
            from app.skills.indicators import calculate_feat_layers

            df_feat = calculate_feat_layers(df)

            # Merge features back to df
            df = pd.concat([df, df_feat], axis=1).dropna(subset=["L1_Mean", "label"])

            # Convert to tensors
            for _, row in df.iterrows():
                # Create 50x50 energy map from features (simplified)
                energy_map = self._create_energy_map(row)

                # Dense features (4 dimensions: L1_Mean, L1_Width, L4_Slope, Div_L1_L2)
                dense = np.array(
                    [
                        row["L1_Mean"],
                        row["L1_Width"],
                        row["L4_Slope"],
                        row["Div_L1_L2"],
                    ],
                    dtype=np.float32,
                )

                # Normalize (Simple)
                dense = np.nan_to_num(dense, 0)

                target = float(row["label"])

                self.samples.append(
                    {
                        "energy_map": torch.tensor(
                            energy_map, dtype=torch.float32
                        ).unsqueeze(0),
                        "dense": torch.tensor(dense, dtype=torch.float32),
                        "target": torch.tensor([target], dtype=torch.float32),
                        "weight": 1.0,
                    }
                )

            logger.info(
                f"Loaded {len(self.samples)} samples from database with Fractal features."
            )

        except Exception as e:
            logger.error(f"Error loading from DB: {e}")

    def _load_corrections(self):
        """Load LLM correction samples with higher weight."""
        if not os.path.exists(CORRECTIONS_FILE):
            return

        try:
            corrections_loaded = 0
            with open(CORRECTIONS_FILE, "r") as f:
                for line in f:
                    corr = json.loads(line)

                    # Create sample from correction
                    energy_map = np.random.randn(50, 50).astype(np.float32) * 0.1
                    dense = np.array(corr.get("features", [0] * 4), dtype=np.float32)
                    target = 1.0 if corr.get("correct_direction") == "BUY" else 0.0

                    self.samples.append(
                        {
                            "energy_map": torch.tensor(energy_map).unsqueeze(0),
                            "dense": torch.tensor(dense),
                            "target": torch.tensor([target]),
                            "weight": self.correction_weight,  # Higher weight
                        }
                    )
                    corrections_loaded += 1

            logger.info(
                f"Loaded {corrections_loaded} correction samples (weight: {self.correction_weight}x)"
            )

        except Exception as e:
            logger.warning(f"Could not load corrections: {e}")

    def _create_energy_map(self, row) -> np.ndarray:
        """
        Create a REAL 50x50 tensor representation of market state.
        Replaces random noise with a Gramian Angular Field-like encoding
        and Time-Series embedding.
        """
        # 1. Extract raw metrics
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]

        # Avoid division by zero
        rng = (h - l) if h != l else 0.0001

        # 2. Normalize candle shape (0-1)
        body_top = max(o, c)
        body_bottom = min(o, c)
        upper_wick = (h - body_top) / rng
        lower_wick = (body_bottom - l) / rng
        body_size = (body_top - body_bottom) / rng
        direction = 1.0 if c > o else 0.0

        # 3. Create a Feature Matrix (Real Data)
        # We construct a 50x50 map where rows representing different feature signals
        # and columns represent 'intensity' or 'time interaction'

        matrix = np.zeros((50, 50), dtype=np.float32)

        # Channel 0-10: Price Action Intensity (Body Size)
        matrix[0:10, :] = body_size

        # Channel 10-20: Volatility (Wicks)
        matrix[10:20, :] = upper_wick + lower_wick

        # Channel 20-30: Directional Bias (Gradient based on direction)
        if direction > 0.5:
            matrix[20:30, :] = np.linspace(0, 1, 50)  # Bullish Gradient
        else:
            matrix[20:30, :] = np.linspace(1, 0, 50)  # Bearish Gradient

        # Channel 30-40: Feature Interactions (L1 * L2)
        l1 = row.get("L1_Mean", 0)
        l2 = row.get("L1_Width", 0)
        matrix[30:40, :] = np.tanh(l1 * l2)  # Non-linear interaction

        # Channel 40-50: Noise Injection (Small amount for regularization only)
        matrix[40:50, :] = np.random.normal(0, 0.01, (10, 50))

        return matrix

    def _generate_synthetic_fallback(self, n_samples: int = 1000):
        """
        [MILITARY GRADE] Geometric Brownian Motion (GBM) fallback.
        Simulates price paths with memory and drift instead of white noise.
        Formula: S(t) = S(0) * exp((mu - 0.5 * sigma^2) * t + sigma * W(t))
        """
        s0 = 100.0  # Initial Price
        mu = 0.0001 # Drift (Bullish bias)
        sigma = 0.02 # Volatility
        dt = 1.0     # Time step
        
        for i in range(n_samples):
            # Generate 32 steps (matching LSTM sequence length context)
            steps = 32
            # Brownian Motion: cumulative sum of normal random variables
            dw = np.random.normal(0, np.sqrt(dt), steps)
            w = np.cumsum(dw)
            
            t = np.linspace(0, steps*dt, steps)
            # GBM Path calculation
            drift = (mu - 0.5 * sigma**2) * t
            diffusion = sigma * w
            path = s0 * np.exp(drift + diffusion)
            
            # Extract final price and metrics
            final_price = path[-1]
            start_price = path[0]
            
            # Synthetic Dense (L1, L1_W, L4_S, Div)
            dense = np.array([
                np.mean(path),                   # L1 Mean (Average Price)
                np.std(path),                    # L1 Width (Volatility)
                (final_price - start_price) / s0, # L4 Slope (Normalized Return)
                1.1 if final_price > start_price else 0.9 # Structural Divergence proxy
            ], dtype=np.float32)
            
            target = 1.0 if final_price > start_price else 0.0
            
            # Update s0 for the next sample to maintain continuity (Markov chain)
            s0 = final_price

            self.samples.append(
                {
                    "energy_map": torch.randn(1, 50, 50) * 0.05, # Background noise
                    "dense": torch.tensor(dense),
                    "target": torch.tensor([target]),
                    "weight": 1.0,
                }
            )
        logger.warning(f"Generated {n_samples} GBM (Stochastic) samples as fallback.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["energy_map"], s["dense"], s["target"], s["weight"]


def create_correction_dataset(
    trade_id: str,
    nn_direction: str,
    llm_direction: str,
    features: List[float],
    market_result: Dict[str, Any],
) -> bool:
    """
    Create a correction sample when LLM corrected NN's prediction.
    These samples will have higher weight in future training.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Determine correct direction based on market result
    profit = market_result.get("profit", 0)

    # If LLM was right, use LLM direction as label
    # If NN was right, use NN direction
    correct_direction = llm_direction if profit > 0 else nn_direction

    correction = {
        "trade_id": trade_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "nn_direction": nn_direction,
        "llm_direction": llm_direction,
        "correct_direction": correct_direction,
        "features": features[:4]
        if len(features) >= 4
        else features + [0] * (4 - len(features)),
        "profit": profit,
        "llm_was_correct": (llm_direction != nn_direction and profit > 0),
    }

    try:
        with open(CORRECTIONS_FILE, "a") as f:
            f.write(json.dumps(correction) + "\n")
        logger.info(f"[RLAIF] Correction saved: {trade_id} -> {correct_direction}")
        return True
    except Exception as e:
        logger.error(f"Failed to save correction: {e}")
        return False


def train_hybrid_brain(
    epochs: int = 5,
    batch_size: int = 32,
    save_path: str = None,
    learning_rate: float = 0.001,
) -> Dict[str, Any]:
    """
    Train the FEAT Hybrid Network with RLAIF-enabled dataset.

    Returns:
        Training metrics including loss history and model path.
    """
    if save_path is None:
        save_path = os.path.join(MODELS_DIR, "feat_hybrid_v1.pth")

    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"🚀 Initializing FEAT Hybrid Training (Epochs: {epochs})...")

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    # Load Data with RLAIF support
    dataset = FeatHybridDataset()
    if len(dataset) == 0:
        return {"error": "No training data available", "samples": 0}

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model
    model = HybridFEATNetwork(dense_input_dim=4).to(device)
    criterion = nn.BCELoss(reduction="none")  # Per-sample loss for weighting
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    model.train()
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_samples = 0

        for energy, dense, target, weight in dataloader:
            energy = energy.to(device)
            dense = dense.to(device)
            target = target.to(device)
            weight = weight.to(device)

            optimizer.zero_grad()
            outputs = model(energy, dense)

            # Use the main output head for training
            output = outputs["p_win"]

            # Weighted loss (corrections have higher weight)
            loss = criterion(output, target)
            weighted_loss = (loss * weight.unsqueeze(1)).mean()

            weighted_loss.backward()
            optimizer.step()

            epoch_loss += weighted_loss.item() * len(energy)
            epoch_samples += len(energy)

        avg_loss = epoch_loss / epoch_samples
        loss_history.append(avg_loss)
        print(f"   Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

    # Save Model
    save_hybrid_model(model, save_path)
    print(f"✅ Training Complete. Model saved to {save_path}")

    return {
        "status": "success",
        "epochs": epochs,
        "samples": len(dataset),
        "final_loss": loss_history[-1] if loss_history else None,
        "loss_history": loss_history,
        "model_path": save_path,
        "device": str(device),
    }


# =============================================================================
# MCP-COMPATIBLE ASYNC WRAPPERS
# =============================================================================


async def trigger_retraining(
    dataset_id: str = None, epochs: int = 5, learning_rate: float = 0.001
) -> Dict[str, Any]:
    """
    MCP Tool: Trigger model retraining.
    Can be called by n8n to recalibrate the neural network.

    Args:
        dataset_id: Optional identifier for specific correction dataset
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    """
    logger.info(
        f"[RLAIF] Retraining triggered (Dataset: {dataset_id}, Epochs: {epochs})"
    )

    # Run training
    result = train_hybrid_brain(epochs=epochs, learning_rate=learning_rate)

    result["triggered_by"] = "mcp_tool"
    result["dataset_id"] = dataset_id
    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    return result


async def add_correction_sample(
    trade_id: str,
    nn_direction: str,
    llm_direction: str,
    features: List[float],
    profit: float,
) -> Dict[str, Any]:
    """
    MCP Tool: Add a correction sample for future training.
    Called when LLM feedback differs from NN prediction.
    """
    success = create_correction_dataset(
        trade_id=trade_id,
        nn_direction=nn_direction,
        llm_direction=llm_direction,
        features=features,
        market_result={"profit": profit},
    )

    return {
        "status": "success" if success else "failed",
        "trade_id": trade_id,
        "correction_file": CORRECTIONS_FILE,
    }


if __name__ == "__main__":
    train_hybrid_brain()
