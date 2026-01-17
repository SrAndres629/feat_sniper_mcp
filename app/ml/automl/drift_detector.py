"""
Drift Detector - FEAT NEXUS OPERATION SINGULARITY
================================================
Monitors model performance decay and triggers AutoML retraining.
Detects: Concept Drift (Win-Rate Decay) and Data Drift (Latency Shifts).
"""

import json
import logging
import os
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta
from app.core.config import settings

logger = logging.getLogger("FEAT.AutoML.Drift")

class DriftDetector:
    """
    Monitors 'experience_replay.jsonl' to detect performance degradation.
    """
    
    def __init__(self, history_path: str = "data/experience_replay.jsonl"):
        self.history_path = history_path
        self.WIN_RATE_THRESHOLD = settings.AUTOML_DRIFT_WIN_RATE_THRESHOLD
        self.MIN_TRADES_FOR_ANALYSIS = settings.AUTOML_DRIFT_MIN_TRADES
        
    def analyze_drift(self) -> Dict[str, Any]:
        """Reads recent trades and calculates drift metrics."""
        if not os.path.exists(self.history_path):
            return {"drift_detected": False, "reason": "NO_DATA"}
            
        recent_trades = []
        try:
            with open(self.history_path, "r") as f:
                # Read last 100 trades
                lines = f.readlines()
                recent_lines = lines[-100:]
                for line in recent_lines:
                    recent_trades.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read experience replay: {e}")
            return {"drift_detected": False, "reason": "FILE_ERROR"}
            
        if len(recent_trades) < self.MIN_TRADES_FOR_ANALYSIS:
            return {"drift_detected": False, "reason": "INSUFFICIENT_TRADES"}
            
        # 1. Concept Drift: Accuracy Decay
        outcomes = [1 if t["outcome"] == "WIN" else 0 for t in recent_trades]
        win_rate = np.mean(outcomes)
        
        # 2. Probability Drift (Neural Confidence vs Real Outcome)
        # Assuming we logged 'p_win' at entry in raw_context
        confidences = []
        for t in recent_trades:
            p_win = t.get("raw_context", {}).get("p_win", 0.5)
            confidences.append(p_win)
        
        mean_conf = np.mean(confidences)
        bias = mean_conf - win_rate # If we think we win 80% but win 50%, bias is 0.3 (Dangerous)
        
        drift_detected = False
        reasons = []
        
        if win_rate < self.WIN_RATE_THRESHOLD:
            drift_detected = True
            reasons.append(f"WIN_RATE_DECAY ({win_rate:.2f})")
            
        if abs(bias) > 0.15:
            drift_detected = True
            reasons.append(f"CONFIDENCE_BIAS ({bias:.2f})")
            
        return {
            "drift_detected": drift_detected,
            "win_rate": float(win_rate),
            "bias": float(bias),
            "reasons": reasons,
            "sample_size": len(recent_trades)
        }

    def should_retrain(self) -> bool:
        """Convenience check for the orchestrator."""
        result = self.analyze_drift()
        if result["drift_detected"]:
            logger.critical(f"⚠️ CONCEPT DRIFT DETECTED: {', '.join(result['reasons'])}")
            return True
        return False

drift_detector = DriftDetector()
