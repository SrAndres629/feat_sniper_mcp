from typing import Dict, Any, Optional
from datetime import datetime
import json

class NeuralService:
    """
    [LEVEL 46] The Neural Cortex Aggregator.
    Centralizes the state of the Brain (AI), Eyes (PVP), and Immune System (Sentinel)
    for real-time visualization.
    """
    def __init__(self):
        self._state = {
            "timestamp": None,
            "predictions": {"buy": 0.0, "sell": 0.0, "hold": 0.0},
            "uncertainty": 0.0,
            "pvp_context": {
                "poc": 0.0, "vah": 0.0, "val": 0.0, 
                "dist_poc": 0.0, "pos_in_va": 0.0, "energy": 0.0
            },
            "immune_system": {
                "status": "NORMAL", "anomaly_score": 0.0, "kill_switch": False
            },
            "symbol": "Waiting...",
            "price": 0.0
        }

    def update_state(self, 
                     symbol: str, 
                     price: float,
                     probs: Dict[str, float], 
                     uncertainty: float, 
                     feature_vector: Dict[str, float],
                     sentinel_status: Dict[str, Any]):
        """
        Updates the global neural state from live inference.
        """
        self._state = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "price": price,
            "predictions": probs,
            "uncertainty": uncertainty,
            "pvp_context": {
                "poc": feature_vector.get("poc_price", 0.0),
                "vah": feature_vector.get("vah_price", 0.0),
                "val": feature_vector.get("val_price", 0.0),
                "dist_poc": feature_vector.get("dist_poc_norm", 0.0),
                "pos_in_va": feature_vector.get("pos_in_va", 0.0),
                "energy": feature_vector.get("energy_score", 0.0)
            },
            "immune_system": {
                "status": "ANOMALY" if sentinel_status.get("kill_switch") else "NORMAL",
                "kill_switch": sentinel_status.get("kill_switch", False),
                "reason": sentinel_status.get("reason", "None")
            },
            # [LEVEL 49] Cognitive Patterns
            "kinetic_context": {
                "pattern_id": feature_vector.get("kinetic_pattern_id", 0),
                "pattern_name": "UNKNOWN", # Helper needed to map ID back to Name if not passed
                "coherence": feature_vector.get("kinetic_coherence", 0.0),
                # Mapping helper
                "label": ["NOISE", "EXPANSION", "COMPRESSION", "FALSE_REVERSAL", "REGIME_CHANGE"][int(feature_vector.get("kinetic_pattern_id", 0))] 
                         if 0 <= int(feature_vector.get("kinetic_pattern_id", 0)) <= 4 else "UNKNOWN"
            }
        }
        
    def get_latest_state(self) -> Dict[str, Any]:
        return self._state

# Singleton Instance
neural_service = NeuralService()
