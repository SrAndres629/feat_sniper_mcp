"""
FEAT NEXUS: BRAIN (Model & Risk Engine)
=======================================
Orchestrates inference and exposes dynamic probabilities for C# integration.
"""

import os
import joblib
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("FEAT.Brain")

class NexusBrain:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = None
        self.feature_names = []
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
    def load_model(self, path: str):
        try:
            artifact = joblib.load(path)
            self.model = artifact["model"]
            self.scaler = artifact.get("scaler")
            self.feature_names = artifact.get("feature_names", [])
            logger.info(f"Loaded Nexus Brain model from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def predict_probability(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Returns calibrated probability and execution parameters for cBot.
        """
        if not self.model:
            return {"probability": 0.5, "status": "NO_MODEL"}

        # Prepare vector
        x = np.array([features.get(f, 0.0) for f in self.feature_names]).reshape(1, -1)
        
        if self.scaler:
            x = self.scaler.transform(x)
            
        # Prediction
        prob = self.model.predict_proba(x)[0][1] # Probability of Class 1 (Success)
        
        # Uncertainty estimate (Rudimentary for baseline)
        uncertainty = 1.0 - abs(prob - 0.5) * 2.0
        
        return {
            "symbol": features.get("symbol", "UNKNOWN"),
            "probability": float(prob),
            "uncertainty": float(uncertainty),
            "risk_parameters": self._calculate_risk_params(prob, uncertainty),
            "version": "1.1.0-baseline"
        }

    def _calculate_risk_params(self, prob: float, uncertainty: float) -> Dict[str, float]:
        """
        AI RISK ENGINE: Adapts TP/SL based on model confidence.
        Logic for C# cBot integration.
        """
        # Base multipliers
        tp_mult = 1.0
        sl_mult = 1.0
        
        # Scenario A: High Confidence Expansion
        if prob > 0.8 and uncertainty < 0.3:
            tp_mult = 1.5 # Stretch targets
            sl_mult = 0.8 # Tighten stop
            
        # Scenario B: High Uncertainty / Low Prob
        elif prob < 0.6 or uncertainty > 0.6:
            tp_mult = 0.8 # Conservative targets
            sl_mult = 1.2 # Looser stop to avoid noise
            
        return {
            "tp_multiplier": tp_mult,
            "sl_multiplier": sl_mult,
            "lot_multiplier": (prob - 0.5) * 2.0 if prob > 0.5 else 0.0
        }

# Singleton
nexus_brain = NexusBrain()

def get_brain_inference(features: Dict[str, float]) -> Dict[str, Any]:
    return nexus_brain.predict_probability(features)
