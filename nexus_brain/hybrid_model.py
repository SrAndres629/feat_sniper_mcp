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
    
# Sincronizado con train_models.py (Arquitecto Jules Protocol)
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
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        return self.classifier(context)

def load_hybrid_model(path: str = "models/lstm_XAUUSD_v2.pt") -> nn.Module:
    """Cargador universal sincronizado con el Architect Protocol."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        # Extraer configuración
        config = checkpoint.get("model_config", {})
        input_dim = config.get("input_dim", 4)
        
        model = LSTMWithAttention(input_dim=input_dim)
        
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint)
            # If the saved state_dict itself is a wrapper (nested)
            if isinstance(state_dict, dict) and "model_state" in state_dict:
                state_dict = state_dict["model_state"]
            
            # Filter out non-model keys just in case
            model_state = {k: v for k, v in state_dict.items() if not k in ["scaler_stats", "model_config", "trained_at", "best_acc"]}
            model.load_state_dict(model_state, strict=False)
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        return model
    except Exception as e:
        print(f"[MODEL ERROR] Load Failed: {e}")
        return None

class HybridModel:
    """
    Wrapper institucional para inferencia causal.
    Maneja el escalamiento 4D y la arquitectura LSTMWithAttention.
    """
    def __init__(self, model_path: str = "models/lstm_XAUUSD_v2.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler_stats = None
        
        checkpoint = self._load_checkpoint(model_path)
        
        if checkpoint:
            config = checkpoint.get("model_config", {"input_dim": 4})
            self.net = LSTMWithAttention(input_dim=config["input_dim"])
            
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                if isinstance(state_dict, dict) and "model_state" in state_dict:
                    state_dict = state_dict["model_state"]
                
                # Filter non-weights keys
                model_state = {k: v for k, v in state_dict.items() if k not in ["scaler_stats", "model_config", "trained_at", "best_acc"]}
                self.net.load_state_dict(model_state, strict=False)
                self.scaler_stats = checkpoint.get("scaler_stats")
            else:
                self.net.load_state_dict(checkpoint, strict=False)
                
            self.net.to(self.device).eval()
            print(f"🧠 LSTM Causal Model Loaded on {self.device} (Stats Embedded)")
        else:
            self.net = None
            print("⚠️ Brain Offline: Model Not Found")

    def _load_checkpoint(self, path):
        try:
            return torch.load(path, map_location=self.device, weights_only=False)
        except:
            return None

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        if self.scaler_stats:
            mean = torch.tensor(self.scaler_stats["mean"], dtype=torch.float32).to(self.device)
            scale = torch.tensor(self.scaler_stats["scale"], dtype=torch.float32).to(self.device)
            return (features - mean) / scale
        return features

    def predict(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Inferencia 4D sincronizada."""
        try:
            if not self.net: return {"error": "No Model", "execute_trade": False}
            
            feat_list = context_data.get('features', [0.0] * 4)
            # El LSTM espera (Batch, Seq, Feat). Aquí Seq=1 para inferencia real-time.
            raw_features = torch.tensor([[feat_list]], dtype=torch.float32).to(self.device)
            
            # Normalización causal
            dense_features = self._normalize(raw_features)
            
            with torch.no_grad():
                logits = self.net(dense_features)
                p_win = torch.softmax(logits, dim=1)[:, 1].item()
                
            return {
                "p_win": round(p_win, 4),
                "execute_trade": p_win > 0.55,
                "alpha_confidence": round(p_win, 4) # Simplified for alpha sync
            }
            
        except Exception as e:
            print(f"Inference Error: {e}")
            return {"error": str(e), "execute_trade": False}
