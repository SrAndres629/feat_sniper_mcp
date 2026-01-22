import os
import torch
import logging
from typing import Optional, Dict, Any
from app.core.config import settings

logger = logging.getLogger("ML.Loader")

class ModelLoader:
    """Loads Pure Hybrid Logic (PyTorch) with Probabilistic Capabilities."""
    
    @staticmethod
    def load_hybrid(symbol: str) -> Optional[Dict[str, Any]]:
        """Loads the HybridProbabilistic (TCN-BiLSTM) model v2."""
        path = os.path.join(settings.MODELS_DIR, f"hybrid_prob_{symbol}_v2.pt")
        
        try:
            from app.ml.models.hybrid_probabilistic import HybridProbabilistic
            
            input_dim = settings.NEURAL_INPUT_DIM
            logger.info(f"🏗️ Initializing Hybrid Model (Dim: {settings.NEURAL_INPUT_DIM} -> {settings.NEURAL_HIDDEN_DIM})")
            model = HybridProbabilistic(
                input_dim=input_dim, 
                hidden_dim=settings.NEURAL_HIDDEN_DIM, 
                num_classes=settings.NEURAL_NUM_CLASSES
            )
            
            if os.path.exists(path):
                data = torch.load(path, map_location="cpu")
                # [TITANIUM AUDIT] Strict loading required. Abort if dimensions mismatch.
                model.load_state_dict(data["state_dict"], strict=True)
                acc = data.get("best_acc", 0.0)
                logger.info(f"Loaded HybridProbabilistic Model for {symbol} (Acc: {acc:.2f})")
            else:
                logger.warning(f"No trained model for {symbol}. Using Untrained Network.")
                
            model.eval()
            return {"model": model, "config": {"seq_len": settings.LSTM_SEQ_LEN}} 

        except Exception as e:
            logger.error(f"Hybrid load failed for {symbol}: {e}")
            return None
