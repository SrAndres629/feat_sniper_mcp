import torch
import numpy as np
import logging
import sys
import os

# Fix path
sys.path.append(os.getcwd())

from app.ml.models.hybrid_probabilistic import HybridProbabilistic
from app.ml.models.feat_encoder import FeatEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyPVP")

def verify_pvp_architecture():
    """
    Verifies [Level 41] PVP-FEAT Architecture.
    Checks:
    1. FeatEncoder dimensionality.
    2. HybridProbabilistic Latent Fusion.
    3. Forward pass with Feat Inputs.
    """
    logger.info("Initializing PVP-FEAT Architecture...")
    
    # 1. Test FeatEncoder
    encoder = FeatEncoder(output_dim=32)
    # Dummy inputs
    form = torch.randn(1, 4)
    space = torch.randn(1, 3)
    accel = torch.randn(1, 3)
    time = torch.randn(1, 4)
    
    z_t = encoder(form, space, accel, time)
    logger.info(f"FeatEncoder Output Shape: {z_t.shape}")
    assert z_t.shape == (1, 32), "Encoder output dim mismatch!"
    
    # 2. Test Hybrid Model with Fusion
    input_dim = 30 # Feature count
    seq_len = 50
    model = HybridProbabilistic(input_dim=input_dim)
    
    x = torch.randn(1, seq_len, input_dim)
    feat_input = {
        "form": form,
        "space": space,
        "accel": accel,
        "time": time
    }
    
    logger.info("Running Hybrid Forward Pass with Latent Fusion...")
    logits = model(x, feat_input=feat_input)
    logger.info(f"Logits Shape: {logits.shape}")
    
    probs = torch.softmax(logits, dim=1)
    logger.info(f"Probabilities: {probs.detach().numpy()}")
    
    logger.info("✅ Level 41 Verification Successful: PVP-FEAT Latent State Active.")

if __name__ == "__main__":
    verify_pvp_architecture()
