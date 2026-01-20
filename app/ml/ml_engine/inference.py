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
        
        # [v5.0-DOCTORAL] Standardized Latent Inputs
        feat_input = {
            "form": torch.tensor([[metrics["physics_entropy"], metrics["physics_viscosity"], metrics["structural_feat_index"], 0.0]], dtype=torch.float32).to(self.device),
            "space": torch.tensor([[metrics["confluence_tensor"], 0.0, 0.0]], dtype=torch.float32).to(self.device), # Adjusted for 3D Space
            "accel": torch.tensor([[metrics["physics_energy"], metrics["physics_force"], metrics["volatility_context"]]], dtype=torch.float32).to(self.device),
            "time": torch.tensor([[metrics["temporal_sin"], metrics["temporal_cos"], metrics["killzone_intensity"], metrics["session_weight"]]], dtype=torch.float32).to(self.device),
            "kinetic": torch.tensor([[metrics["trap_score"], metrics["structural_feat_index"], metrics["physics_force"], metrics["physics_viscosity"]]], dtype=torch.float32).to(self.device)
        }
        
        # [PHASE 13] Physics Tensor for Reconstruction Gating (Cross-Attention)
        p_tensor = torch.stack([
            feat_input["accel"][:, 0], # energy
            feat_input["accel"][:, 1], # force
            feat_input["form"][:, 0],  # entropy
            feat_input["form"][:, 1]   # viscosity
        ], dim=1)

        # MC Dropout Sampling (Bayesian Fusion v5.0)
        model.train() 
        p_win_arr, alpha_arr, logits_arr, aleatoric_arr = [], [], [], []
        
        with torch.no_grad():
            for _ in range(settings.MC_DROPOUT_SAMPLES):
                outputs = model(x, feat_input=feat_input, physics_tensor=p_tensor, force_dropout=True)
                p_win_arr.append(outputs["p_win"].item())
                alpha_arr.append(outputs["alpha"].item())
                # Capture the model's self-predicted uncertainty
                aleatoric_arr.append(outputs["uncertainty"].item())
                
                probs = torch.softmax(outputs["logits"], dim=-1)[0]
                logits_arr.append((probs[2].item() - probs[0].item() + 1.0) / 2.0)
                
        model.eval()
        
        # Calculate Hierarchical Uncertainty
        epistemic_unc = float(np.std(p_win_arr)) # Uncertainty due to model parameters
        aleatoric_unc = float(np.mean(aleatoric_arr)) # Uncertainty due to market noise
        total_uncertainty = (epistemic_unc + aleatoric_unc) / 2.0
        
        return {
            "p_win": float(np.mean(p_win_arr)),
            "win_confidence": float(np.mean(p_win_arr)),
            "alpha_multiplier": float(np.mean(alpha_arr)),
            "directional_score": float(np.mean(logits_arr)),
            "uncertainty": total_uncertainty,
            "epistemic_unc": epistemic_unc,
            "aleatoric_unc": aleatoric_unc
        }
