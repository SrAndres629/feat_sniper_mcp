import torch
import numpy as np
import logging
import sys
import os

# Fix path for module imports
sys.path.append(os.getcwd())

from app.ml.models.hybrid_probabilistic import HybridProbabilistic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyNeural")

def verify_probabilistic_inference():
    """
    Verifies that the HybridProbabilistic model generates uncertainty 
    via Monte Carlo Dropout.
    """
    logger.info("Initializing HybridProbabilistic Model [Level 40]...")
    
    # Random input: (Batch=1, Seq=50, Feat=30)
    input_dim = 30
    seq_len = 50
    x = torch.randn(1, seq_len, input_dim)
    
    model = HybridProbabilistic(input_dim=input_dim)
    model.train() # Enable Dropout
    
    logger.info("Running Monte Carlo Dropout (N=50)...")
    predictions = []
    
    with torch.no_grad():
        for i in range(50):
            # Force Dropout
            logits = model(x, force_dropout=True)
            probs = torch.softmax(logits, dim=1)
            p_buy = probs[0, 2].item()
            predictions.append(p_buy)
            
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions)
    std_dev = np.std(predictions)
    
    logger.info(f"Mean Prediction: {mean_pred:.4f}")
    logger.info(f"Uncertainty (StdDev): {std_dev:.4f}")
    
    if std_dev > 0.0001:
        logger.info("[SUCCESS] Model exhibits Epistemic Uncertainty (Probabilistic Behavior Verified).")
    else:
        logger.error("[FAILURE] Model is Deterministic! Dropout not active during inference.")

if __name__ == "__main__":
    verify_probabilistic_inference()
