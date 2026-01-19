import torch
import numpy as np
import logging
import pandas as pd
from typing import Dict, List, Any
from app.core.config import settings

logger = logging.getLogger("ML.Inference")

class InferenceEngine:
    """Handles deep neural inference (Monte Carlo Dropout)."""
    
    def __init__(self):
         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict_hybrid_uncertainty(self, model_data: Dict, sequence: List[Dict], feat_names: List[str], symbol: str, seq_len: int) -> Dict:
        """Monte Carlo Dropout Inference (Ensemble) with PVP-FEAT Latent State."""
        if not model_data: return {"p_win": 0.5, "uncertainty": 1.0}
        
        model = model_data["model"]
        model.to(self.device)
        
        from app.ml.feat_processor import feat_processor
        
        # Tensorize sequence
        seq_array = np.array([
            feat_processor.tensorize_snapshot(s, feat_names)
            for s in sequence
        ], dtype=np.float32)
        
        # Padding
        if len(seq_array) < seq_len:
             diff = seq_len - len(seq_array)
             pad = np.zeros((diff, len(feat_names)), dtype=np.float32)
             seq_array = np.concatenate([pad, seq_array])
             
        x = torch.tensor(seq_array).unsqueeze(0).to(self.device)
        
        # Compute Latent Inputs
        latest_state = sequence[-1]
        metrics = feat_processor.compute_latent_vector(pd.Series(latest_state))
        
        # [v4.1] Tensor Grouping for FeatEncoder
        # Mapping aligned with 18D latent structure
        feat_input = {
            "form": torch.tensor([[metrics["skew"], metrics["entropy"], metrics["form"], 0.0]], dtype=torch.float32).to(self.device),
            "space": torch.tensor([[metrics["dist_poc"], metrics["pos_in_va"], metrics["space"]]], dtype=torch.float32).to(self.device),
            "accel": torch.tensor([[metrics["energy"], metrics["accel"], metrics["kalman_score"]]], dtype=torch.float32).to(self.device),
            "time": torch.tensor([[metrics["dist_micro"], metrics["dist_struct"], metrics["dist_macro"], metrics["time"]]], dtype=torch.float32).to(self.device),
            "kinetic": torch.tensor([[metrics["kinetic_pattern_id"], metrics["kinetic_coherence"], metrics["dist_bias"], metrics["layer_alignment"]]], dtype=torch.float32).to(self.device)
        }
        
        # MC Dropout Sampling
        model.train() 
        p_win_arr, alpha_arr, logits_arr, vol_arr = [], [], [], []
        
        with torch.no_grad():
            for _ in range(settings.MC_DROPOUT_SAMPLES):
                outputs = model(x, feat_input=feat_input, force_dropout=True)
                p_win_arr.append(outputs["p_win"].item())
                alpha_arr.append(outputs["alpha"].item())
                vol_arr.append(outputs["volatility"].item())
                probs = torch.softmax(outputs["logits"], dim=-1)[0]
                logits_arr.append((probs[2].item() - probs[0].item() + 1.0) / 2.0)
                
        model.eval()
        
        return {
            "p_win": float(np.mean(p_win_arr)),
            "win_confidence": float(np.mean(p_win_arr)),
            "alpha_multiplier": float(np.mean(alpha_arr)),
            "directional_score": float(np.mean(logits_arr)),
            "volatility_regime": float(np.mean(vol_arr)),
            "uncertainty": float(np.std(p_win_arr))
        }
