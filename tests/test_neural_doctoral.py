
import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from app.ml.models.hybrid_probabilistic import HybridProbabilistic
from app.core.config import settings

def test_doctoral_model():
    print("🧪 Testing Doctoral Model v4.0...")
    
    # Mock settings if needed (assuming defaults handled in init)
    input_dim = 16
    batch_size = 5
    seq_len = 60
    
    model = HybridProbabilistic(input_dim=input_dim)
    print(f"✅ Instantiation Successful. Meta: {model.metadata}")
    
    # Create Dummy Inputs
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Physics Tensor: [OFI_Z, Illiquidity, Impact, Kinetic]
    physics = torch.randn(batch_size, 4)
    
    # FEAT Inputs (Latent Encoder)
    # Generate random floats but ensure the first column (PatternID) is in [0, 5]
    kinetic_raw = torch.randn(batch_size, 4)
    # Force first column to be valid integers 0-5
    kinetic_raw[:, 0] = torch.randint(0, 6, (batch_size,)).float()

    feat_input = {
        "form": torch.randn(batch_size, 4),
        "space": torch.randn(batch_size, 3),
        "accel": torch.randn(batch_size, 3),
        "time": torch.randn(batch_size, 4),
        "kinetic": kinetic_raw
    }
    
    # Forward Pass
    output = model(x, feat_input=feat_input, physics_tensor=physics)
    
    assert "logits" in output
    assert "confidence" in output
    assert output["logits"].shape == (batch_size, 3)
    
    print("✅ Forward Pass Successful.")
    print("🎓 Doctoral Architecture Verified.")

if __name__ == "__main__":
    try:
        test_doctoral_model()
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
