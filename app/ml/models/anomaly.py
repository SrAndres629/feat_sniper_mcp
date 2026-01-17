import numpy as np
import joblib
import os
import logging
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, Optional

logger = logging.getLogger("FEAT.Anomaly")

class AnomalyDetector:
    """
    [LEVEL 44] The Immune System.
    Uses Isolation Forest to detect 'Alien' data patterns (Flash Crashes, API Errors, Zero-Liquidity).
    
    If Anomaly Score < Threshold, the system initiates a DEFENSIVE LOCKDOWN.
    """
    
    def __init__(self, contamination: float = 0.01):
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination, # Expected % of anomalies (1%)
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
        self.model_path = "models/anomaly_detector.joblib"
        self._load()

    def _load(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                logger.info(f"[SHIELD] Anomaly Detector loaded from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load Anomaly Detector: {e}")

    def train(self, X: np.ndarray):
        """Train on 'Normal' market data."""
        logger.info(f"Training Immune System on {len(X)} samples...")
        self.model.fit(X)
        self.is_trained = True
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info("🛡️ Immune System Validated & Saved.")

    def is_anomaly(self, features: Dict[str, float]) -> bool:
        """
        Returns TRUE if the data point is an outlier.
        """
        if not self.is_trained:
            return False # Default to trust if not trained
            
        try:
            # Convert dict to array (Ensure order matches training!)
            # We use a standard vector for anomaly checks.
            # Ideally consistent with `feat_processor`.
            # For robustness, we check critical physics vars only.
            
            vector = np.array([[
                features.get("close", 0),
                features.get("volume", 0),
                features.get("atr", 0),
                features.get("rsi", 50)
            ]])
            
            # Predict returns -1 for outlier, 1 for inlier
            pred = self.model.predict(vector)
            
            if pred[0] == -1:
                score = self.model.decision_function(vector)[0]
                logger.warning(f"🦠 TOXIC DATA DETECTED! Score: {score:.4f} (Threshold: 0.0)")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Anomaly Check Failed: {e}")
            return False

# Unit Singleton
anomaly_detector = AnomalyDetector()
