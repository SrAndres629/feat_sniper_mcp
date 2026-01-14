import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any
import os

class HybridFEATNetwork(nn.Module):
    """
    Neural Risk-Aware Trading Network.
    
    Outputs 3 tensors for dynamic risk management:
    - p_win: Trade success probability
    - alpha_confidence: Directional certainty for lot sizing
    - volatility_regime: For adaptive SL/TP
    """
    
    def __init__(self, dense_input_dim=None):
        super(HybridFEATNetwork, self).__init__()
        
        # 1. CNN Stream (For 50x50 Energy Map)
        # Input: (1, 50, 50) -> Heatmap
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Output after 3 pools (50->25->12->6): 64 * 6 * 6
        self.cnn_fc = nn.Linear(64 * 6 * 6, 128)
        
        # 2. Dense Stream (For FourJarvis Scores & Physics)
        # Using LazyLinear to support dynamic indicator counts
        self.dense1 = nn.LazyLinear(64)
        self.dense2 = nn.Linear(64, 32)
        
        # 3. Fusion Stream
        # Input: 128 (CNN) + 32 (Dense) = 160
        self.fusion1 = nn.Linear(160, 80)
        self.fusion2 = nn.Linear(80, 40)
        self.dropout = nn.Dropout(0.3)
        
        # 4. Multi-Head Output Layers
        self.head_p_win = nn.Linear(40, 1)
        self.head_alpha = nn.Linear(40, 1)
        self.head_volatility = nn.Linear(40, 1)
        self.head_urgency = nn.Linear(40, 1)

    def forward(self, energy_map: torch.Tensor, dense_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # CNN Flow
        x1 = self.pool(F.relu(self.conv1(energy_map)))
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = self.pool(F.relu(self.conv3(x1)))
        x1 = x1.view(-1, 64 * 6 * 6)
        x1 = F.relu(self.cnn_fc(x1))
        
        # Dense Flow
        x2 = F.relu(self.dense1(dense_features))
        x2 = F.relu(self.dense2(x2))
        
        # Fusion
        combined = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fusion1(combined))
        x = self.dropout(x)
        x = F.relu(self.fusion2(x))
        
        # Multi-Head Outputs (all sigmoid for [0,1] range)
        outputs = {
            "p_win": torch.sigmoid(self.head_p_win(x)),
            "alpha_confidence": torch.sigmoid(self.head_alpha(x)),
            "volatility_regime": torch.sigmoid(self.head_volatility(x)),
            "urgency": torch.sigmoid(self.head_urgency(x))
        }
        
        return outputs
    
    def forward_legacy(self, energy_map: torch.Tensor, dense_features: torch.Tensor) -> torch.Tensor:
        """Backward-compatible single output (p_win only)."""
        outputs = self.forward(energy_map, dense_features)
        return outputs["p_win"]
    
    def get_risk_decision(self, energy_map: torch.Tensor, dense_features: torch.Tensor) -> Dict[str, float]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(energy_map, dense_features)
            
            p_win = outputs["p_win"].item()
            alpha = outputs["alpha_confidence"].item()
            vol = outputs["volatility_regime"].item()
            urgency = outputs["urgency"].item()
            
            # Neural lot allocation based on alpha
            if alpha > 0.85:
                lot_allocation = 0.75
            elif alpha > 0.70:
                lot_allocation = 0.50
            elif alpha > 0.55:
                lot_allocation = 0.25
            else:
                lot_allocation = 0.10
            
            # Neural SL/TP multipliers based on volatility
            if vol > 0.7:
                sl_mult = 1.5
                tp_mult = 0.8
            elif vol < 0.3:
                sl_mult = 0.8
                tp_mult = 1.3
            else:
                sl_mult = 1.0
                tp_mult = 1.0
            
            return {
                "p_win": round(p_win, 4),
                "alpha_confidence": round(alpha, 4),
                "volatility_regime": round(vol, 4),
                "urgency": round(urgency, 4),
                "neural_lot_allocation": round(lot_allocation, 2),
                "neural_sl_multiplier": round(sl_mult, 2),
                "neural_tp_multiplier": round(tp_mult, 2),
                "execute_trade": p_win > 0.6 and alpha > 0.5
            }

def save_hybrid_model(model: HybridFEATNetwork, path: str = "models/feat_hybrid_v2.pth"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "version": "2.0-neural-risk",
        "output_heads": ["p_win", "alpha_confidence", "volatility_regime", "urgency"]
    }, path)
    
def load_hybrid_model(path: str = "models/feat_hybrid_v2.pth") -> HybridFEATNetwork:
    model = HybridFEATNetwork()
    try:
        checkpoint = torch.load(path, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    except FileNotFoundError:
        return None

class HybridModel:
    """
    Local Wrapper for HybridFEATNetwork.
    Bypasses Docker dependency by running PyTorch locally.
    """
    def __init__(self, model_path: str = "models/feat_hybrid_v2.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = load_hybrid_model(model_path)
        
        if not self.net:
            # Initialize fresh network if no checkpoint
            self.net = HybridFEATNetwork()
            
        self.net.to(self.device)
        self.net.eval()
        print(f"🧠 HybridModel Loaded on {self.device}")

    def predict(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inference Interface.
        Converts dictionary data into PyTorch tensors and runs forward pass.
        """
        try:
            # TODO: Real Tensor Construction from FEAT Data.
            # Currently mocking input for integration testing.
            
            # 1. Mock Energy Map (1, 50, 50)
            energy_map = torch.randn(1, 1, 50, 50).to(self.device)
            
            # 2. Mock Dense Features (Batch, 64) - LazyLinear will adapt
            # If model initialized with 64, we need 64 features.
            # But LazyLinear adapts on first forward? 
            # Yes, but we need to match what it expects if loaded.
            # Assuming 64 dims for now.
            # 2. Real Dense Features
            if 'features' in context_data and isinstance(context_data['features'], list):
                # Convert list to tensor (Batch=1, Dims=10)
                feat_list = context_data['features']
                dense_features = torch.tensor([feat_list], dtype=torch.float32).to(self.device)
            else:
                # Fallback if no features provided
                dense_features = torch.randn(1, 10).to(self.device) 
            
            # Inference
            decision = self.net.get_risk_decision(energy_map, dense_features)
            return decision
            
        except Exception as e:
            print(f"Inference Error: {e}")
            return {"error": str(e), "execute_trade": False}
