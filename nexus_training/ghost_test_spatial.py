
import torch
import numpy as np
from app.ml.models.hybrid_probabilistic import HybridProbabilistic

def run_ghost_test():
    """
    [LEVEL 54] INFERENCE GHOST TEST.
    Verifies if injecting a "Spatial Energy Wall" (Energy Map) 
    shifts the model's decision bias without changing the price sequence.
    """
    print("🚀 Initiating Inference Ghost Test (Spatial Bias Injection)...")
    
    input_dim = 25 # Institutional Latent Space
    model = HybridProbabilistic(input_dim=input_dim, hidden_dim=64, num_classes=3)
    model.eval()
    
    # 1. Base Sequence (Neutral / Flat)
    seq = torch.zeros((1, 32, input_dim)) # (Batch, Seq, Features) -> SeqLen 32 per config
    feat_input = {
        "form": torch.zeros((1, 4)),
        "space": torch.zeros((1, 3)),
        "accel": torch.zeros((1, 3)),
        "time": torch.zeros((1, 4)),
        "kinetic": torch.zeros((1, 4))
    }
    
    # 2. Case A: Neutral Energy Map
    feat_input["spatial_map"] = torch.zeros((1, 1, 50, 50))
    with torch.no_grad():
        outputs_neutral = model(seq, feat_input=feat_input)
        prob_neutral = torch.softmax(outputs_neutral["logits"], dim=1)
    
    # 3. Case B: Bullish Energy Wall
    bull_map = torch.zeros((1, 1, 50, 50))
    bull_map[:, :, 25:, :] = 1.0 
    feat_input["spatial_map"] = bull_map
    with torch.no_grad():
        outputs_bull = model(seq, feat_input=feat_input)
        prob_bull = torch.softmax(outputs_bull["logits"], dim=1)
        
    # 4. Case C: Bearish Energy Wall
    bear_map = torch.zeros((1, 1, 50, 50))
    bear_map[:, :, :25, :] = 1.0 
    feat_input["spatial_map"] = bear_map
    with torch.no_grad():
        outputs_bear = model(seq, feat_input=feat_input)
        prob_bear = torch.softmax(outputs_bear["logits"], dim=1)

    print("\n--- TEST RESULTS ---")
    print(f"Neutral Bias [SELL/HOLD/BUY]: {prob_neutral[0].tolist()}")
    print(f"Bullish Space Bias:        {prob_bull[0].tolist()}")
    print(f"Bearish Space Bias:        {prob_bear[0].tolist()}")
    print(f"Bullish Alpha:             {outputs_bull['alpha'].item():.4f}")
    print(f"Bearish Alpha:             {outputs_bear['alpha'].item():.4f}")
    
    diff_bull = (prob_bull - prob_neutral).abs().sum().item()
    diff_bear = (prob_bear - prob_neutral).abs().sum().item()
    
    if diff_bull > 1e-6 or diff_bear > 1e-6:
        print("\n✅ PASS: The Spatial Cortex is successfully injecting latent bias into the decision manifold.")
    else:
        print("\n❌ FAIL: Spatial maps are being ignored by the fusion layer.")

if __name__ == "__main__":
    run_ghost_test()
