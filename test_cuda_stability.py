import torch
import torch.nn as nn
from torch.nn.utils import parametrizations
import sys

# Fixed TCN-like block for stress testing
class SimpleTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(SimpleTemporalBlock, self).__init__()
        # Use circular padding to keep same length
        self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(n_outputs)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

def test_cuda_high_stress():
    print(f"PyTorch version: {torch.version.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Simple but deep model to test cuDNN kernels
    layers = []
    in_ch = 24
    for out_ch in [64, 128, 256, 512]:
        layers.append(SimpleTemporalBlock(in_ch, out_ch))
        in_ch = out_ch
    
    layers.extend([
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(512, 1)
    ])
    
    model = nn.Sequential(*layers).to(device)

    # Batch 64, 24 channels, 60 time steps
    x = torch.randn(64, 24, 60).to(device)
    target = torch.randn(64, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    try:
        print("🚀 STRESS TEST: Forward Pass...")
        output = model(x)
        print("✅ Forward Pass Successful.")

        print("🚀 STRESS TEST: Backward Pass...")
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print("✅ Backward Pass Successful.")
        
        print("🎉 SUCCESS: CUDA is handling sequential convolutions correctly.")
    except Exception as e:
        print(f"❌ CUDA CRASH detected: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cuda_high_stress()
