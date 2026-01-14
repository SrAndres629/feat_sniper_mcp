import numpy as np
import logging
from collections import deque
from typing import Dict, List, Optional, Any

logger = logging.getLogger("feat.ml.drift_monitor")

class WeightDriftMonitor:
    """
    Monitor de Deriva de Pesos (Weight Drift Monitor).
    Detecta si la distribución de los inputs en tiempo real se desvía de los
    stats con los que se entrenó el modelo.
    """
    def __init__(self, training_stats: Optional[Dict[str, List[float]]] = None, window_size: int = 200, drift_threshold: float = 0.8):
        self.training_stats = training_stats
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        
        # Buffers circulares para almacenar los últimos features
        self.feature_window: deque = deque(maxlen=window_size)
        
        # Resultados de drift
        self.last_drift_score = 0.0
        self.is_drifting = False
        
        if training_stats:
            logger.info(f"[DRIFT] Monitor Initialized (Window: {window_size}, Threshold: {drift_threshold})")
            logger.info(f"[DRIFT] Training Means: {training_stats.get('mean')}")
        else:
            logger.warning("[DRIFT] No training stats provided. Monitor will only collect data.")

    def update(self, current_features: List[float]) -> Dict[str, Any]:
        """
        Actualiza el monitor con un nuevo vector de features y calcula el drift score.
        """
        self.feature_window.append(current_features)
        
        # No calculamos drift hasta llenar al menos el 25% de la ventana
        if len(self.feature_window) < self.window_size // 4 or not self.training_stats:
            return {"drift_score": 0.0, "is_drifting": False, "status": "collecting"}

        # Convertir ventana a numpy para cálculo eficiente
        window_np = np.array(self.feature_window)
        current_means = np.mean(window_np, axis=0)
        
        train_means = np.array(self.training_stats["mean"])
        train_scales = np.array(self.training_stats["scale"])
        
        # Manejar discrepancia de dimensiones (ej: Modelo 4D vs Inferencia 5D)
        min_dim = min(len(current_means), len(train_means))
        c_means = current_means[:min_dim]
        t_means = train_means[:min_dim]
        t_scales = train_scales[:min_dim]
        
        # 1. Calcular Desviación Relativa (Z-Score de Medias)
        mean_diff = np.abs(c_means - t_means)
        z_scores = mean_diff / (t_scales + 1e-9)
        
        # 2. Score Agregado (Max Z-Score normalizado)
        avg_z = np.mean(z_scores)
        max_z = np.max(z_scores)
        
        # Normalizamos el drift_score: 3.0 Z-Score -> 1.0 (Critical Drift)
        drift_score = min(max_z / 3.0, 1.0)

        
        self.last_drift_score = drift_score
        self.is_drifting = drift_score >= self.drift_threshold
        
        metrics = {
            "drift_score": round(drift_score, 4),
            "is_drifting": self.is_drifting,
            "avg_z": round(avg_z, 4),
            "max_z": round(max_z, 4),
            "window_fill": len(self.feature_window) / self.window_size
        }
        
        if self.is_drifting:
            logger.warning(f"⚠️ [DRIFT ALERT] High statistical drift detected: {drift_score:.4f} (Max Z: {max_z:.2f})")
            
        return metrics

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_active": self.training_stats is not None,
            "window_size": self.window_size,
            "current_fill": len(self.feature_window),
            "last_drift_score": self.last_drift_score,
            "is_drifting": self.is_drifting
        }
