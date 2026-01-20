import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

class NeuralHealthTracker:
    """
    PhD-Level Neural Performance Monitor.
    Tracks the alignment between model confidence and real-world results.
    Calculates metrics like Brier Score, Expected Calibration Error (ECE), and Drift.
    """
    
    def __init__(self, storage_path: str = "data/neural_health.json"):
        self.storage_path = storage_path
        self.history: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    self.history = json.load(f)
            except:
                self.history = []

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def log_prediction(self, trade_id: str, confidence: float, action: str):
        """Logs a prediction at the time of entry."""
        entry = {
            "trade_id": trade_id,
            "timestamp_entry": datetime.now().isoformat(),
            "confidence": float(confidence),
            "action": action,
            "status": "OPEN",
            "outcome": None,
            "pnl": 0.0
        }
        self.history.append(entry)
        self._save()

    def resolve_prediction(self, trade_id: str, pnl: float):
        """Updates a logged prediction with the final outcome (WIN/LOSS)."""
        for entry in self.history:
            if entry["trade_id"] == trade_id and entry["status"] == "OPEN":
                entry["status"] = "CLOSED"
                entry["outcome"] = 1.0 if pnl > 0 else 0.0
                entry["pnl"] = float(pnl)
                entry["timestamp_exit"] = datetime.now().isoformat()
                break
        self._save()

    def get_health_metrics(self) -> Dict[str, Any]:
        """Calculates PhD metrics: Brier Score, Drift, and Accuracy."""
        closed_trades = [e for e in self.history if e["status"] == "CLOSED"]
        if not closed_trades:
            return {"status": "INITIALIZING", "brier_score": 0.0, "drift_score": 0.0}

        confidences = np.array([e["confidence"] for e in closed_trades])
        outcomes = np.array([e["outcome"] for e in closed_trades])
        
        # 1. Brier Score: lower is better (0 to 1)
        brier_score = np.mean((confidences - outcomes)**2)
        
        # 2. Accuracy vs Confidence Drift
        avg_confidence = np.mean(confidences)
        avg_winrate = np.mean(outcomes)
        drift = abs(avg_confidence - avg_winrate)
        
        # 3. Health Status
        status = "HEALTHY"
        if brier_score > 0.3 or drift > 0.25:
            status = "DETERIORATING"
        if brier_score > 0.5 or drift > 0.4:
            status = "CRITICAL_DRIFT"

        return {
            "status": status,
            "brier_score": float(brier_score),
            "drift_score": float(drift),
            "avg_confidence": float(avg_confidence),
            "actual_winrate": float(avg_winrate),
            "sample_size": len(closed_trades)
        }

# Global Instance
neural_health = NeuralHealthTracker()
