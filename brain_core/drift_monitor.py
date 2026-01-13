
# MLOPS GUARDIAN: Concept Drift Monitor
# =====================================
# Detects when the market regime has shifted significantly from the
# training data distribution (Concept Drift).
# If Drift > Threshold (3 Sigma), the system should trigger SAFETY_MODE.

import numpy as np
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger("MLOps.DriftMonitor")

class DriftDetector:
    """
    Monitors statistical distance between Recent Market Data (Live)
    and Training Data (Baseline).
    """
    
    def __init__(self, baseline_stats: Optional[Dict[str, Any]] = None):
        self.baseline_stats = baseline_stats or {}
        self.history_window: List[float] = []
        self.window_size = 100
        self.drift_threshold = 3.0 # Z-Score / Sigma equivalent
        
    def update_baseline(self, training_data: np.ndarray):
        """
        Updates the baseline statistics from new training data.
        """
        if len(training_data) == 0: return
        
        self.baseline_stats = {
            "mean": float(np.mean(training_data)),
            "std": float(np.std(training_data)),
            "min": float(np.min(training_data)),
            "max": float(np.max(training_data)),
            "last_update": datetime.utcnow().isoformat()
        }
        logger.info(f"Drift Baseline Updated: Mean={self.baseline_stats['mean']:.4f}, Std={self.baseline_stats['std']:.4f}")

    def detect_drift(self, current_value: float) -> Dict[str, Any]:
        """
        Checks if the current value (e.g., Volatility, Error Rate, Feature Vector Norm)
        has drifted from the baseline.
        
        Returns:
            Dict with drift_score, is_drift, severity.
        """
        if not self.baseline_stats:
            return {"drift_score": 0.0, "is_drift": False, "status": "NO_BASELINE"}
            
        # Z-Score Calculation (Simple Univariate Drift)
        # In a real heavy tensor system, we'd use KS-Test or KL-Divergence on a window.
        # For High-Frequency, Z-Score of the running metric is fast and effective.
        
        mu = self.baseline_stats.get("mean", 0.0)
        sigma = self.baseline_stats.get("std", 1.0)
        
        if sigma == 0: sigma = 1e-9
        
        z_score = (current_value - mu) / sigma
        abs_score = abs(z_score)
        
        is_drift = abs_score > self.drift_threshold
        
        severity = "NORMAL"
        if abs_score > 5.0: severity = "CRITICAL_DRIFT"
        elif abs_score > 3.0: severity = "DRIFT_DETECTED"
        elif abs_score > 2.0: severity = "WARNING"
        
        # Update rolling window for adaptive checks (future use)
        self.history_window.append(current_value)
        if len(self.history_window) > self.window_size:
            self.history_window.pop(0)

        result = {
            "drift_score": float(abs_score),
            "is_drift": is_drift,
            "severity": severity,
            "baseline_mean": mu,
            "current_val": current_value
        }
        
        if is_drift:
            logger.warning(f"🚨 DRIFT DETECTED! Score: {abs_score:.2f} (Sigma). Value: {current_value} vs Mean: {mu}")
            
        return result

# Singleton Instance
drift_monitor = DriftDetector()
