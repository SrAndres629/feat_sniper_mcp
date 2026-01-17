"""
MARKET REGIME DETECTOR (Unsupervised Learning)
==============================================
Uses K-Means Clustering to classify market state.
0: CALM (PVP Reversion)
1: TREND (Momentum)
2: CHAOS (Stay Out)
"""
import numpy as np
import joblib
import os
import logging

from sklearn.ensemble import IsolationForest
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from app.core.config import settings

logger = logging.getLogger("feat.market_regime")

class MarketRegime:
    def __init__(self, n_clusters=None):
        self.n_clusters = n_clusters or settings.REGIME_CLUSTERS
        self.model_path = settings.REGIME_MODEL_PATH
        self.scaler_path = settings.REGIME_SCALER_PATH
        self.anomaly_path = settings.ANOMALY_MODEL_PATH
        
        self.model = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, n_init="auto")
        # [LEVEL 32] UNSUPERVISED ANOMALY DETECTION
        self.anomaly_detector = IsolationForest(contamination=settings.ANOMALY_CONTAMINATION, random_state=42)
        self.scaler = StandardScaler()
        
        self.is_fitted = False
        self.labels_map = {} 
        self._load_model()
        
    def _load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                if os.path.exists(self.anomaly_path):
                    self.anomaly_detector = joblib.load(self.anomaly_path)
                    
                self.is_fitted = True
                self._remap_labels()
                logger.info("Loaded Unsupervised Regime & Anomaly Models.")
            except Exception as e:
                logger.warning(f"Failed to load models: {e}")
                
    def _remap_labels(self):
        """
        Dynamically determine which cluster is which based on Centroids.
        """
        if not hasattr(self.model, 'cluster_centers_'):
            return
            
        # We must look at Scaled centers, but mapping logic needs to understand the scale.
        # Feature 0 is ATR.
        centers = self.model.cluster_centers_ 
        volatilities = centers[:, 0] 
        
        sorted_indices = np.argsort(volatilities)
        
        self.labels_map = {
            sorted_indices[0]: "CALM",
            sorted_indices[1]: "TREND",
            sorted_indices[2]: "CHAOS"
        }
        logger.info(f"[REGIME] Clusters Mapped: {self.labels_map}")

    def update(self, atr: float, volume: float):
        """
        Online Learning: Update model with new data point.
        """
        try:
           X = np.array([[float(atr), float(volume)]])
           
           # incremental scaling
           self.scaler.partial_fit(X)
           X_scaled = self.scaler.transform(X)
           
           self.model.partial_fit(X_scaled)
           
           # [LEVEL 32] Fit Anomaly Detector incrementally? 
           # IsoForest doesn't support partial_fit easily. We fit on buffer periodically?
           # specific usage: if we have enough data, we refit it.
           # For now, we assume pre-trained or we fit on small batch?
           # Simpler: We can't partial_fit IsoForest. 
           # We will skip online update for Anomaly Detector to avoid overhead/crash.
           
           self.is_fitted = True
           self._remap_labels()
        except Exception as e:
            logger.error(f"Regime Update Error: {e}")

    def predict(self, atr: float, volume: float) -> str:
        if not self.is_fitted:
            self.update(atr, volume) 
            return "TREND"
            
        try:
            X = np.array([[float(atr), float(volume)]])
            X_scaled = self.scaler.transform(X)
            cluster = self.model.predict(X_scaled)[0]
            return self.labels_map.get(cluster, "UNKNOWN")
        except Exception:
            return "UNKNOWN"

    def is_anomaly(self, atr: float, volume: float) -> bool:
        """
        [LEVEL 32] Unsupervised Outlier Detection.
        Returns True if current market state is an Anomaly (OOD).
        """
        if not hasattr(self.anomaly_detector, "estimators_"):
             # Not fitted yet
             return False
             
        try:
            X = np.array([[float(atr), float(volume)]])
            X_scaled = self.scaler.transform(X)
            pred = self.anomaly_detector.predict(X_scaled)[0]
            # -1 = Anomaly, 1 = Normal
            return pred == -1
        except:
            return False
            
    def train_anomaly_guardian(self, data_matrix):
        """Called periodically to retrain IsoForest on history."""
        try:
            self.anomaly_detector.fit(data_matrix)
            self.save()
            logger.info("🛡️ Anomaly Guardian Retrained.")
        except Exception as e:
            logger.error(f"Guardian Retrain Failed: {e}")

    def save(self):
        try:
            os.makedirs("models", exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            joblib.dump(self.anomaly_detector, self.anomaly_path)
        except Exception as e:
            logger.error(f"Cannot save models: {e}")

# Singleton
market_regime = MarketRegime()
