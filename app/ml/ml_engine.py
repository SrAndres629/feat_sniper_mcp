"""
ML Engine - Quantum Leap Phase 4
================================
Motor de inferencia con Shadow Mode y detección de anomalías.

Features:
- Carga segura de modelos GBM y LSTM
- Shadow Mode por defecto (no ejecuta órdenes)
- IsolationForest para detección de manipulación
- Integración SSE para alertas
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

# Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MODELS_DIR = os.getenv("MODELS_DIR", "models")
EXECUTION_ENABLED = os.getenv("EXECUTION_ENABLED", "False").lower() == "true"
ANOMALY_CONTAMINATION = float(os.getenv("ANOMALY_CONTAMINATION", "0.01"))
SHADOW_LOG_PATH = os.getenv("SHADOW_LOG_PATH", "data/shadow_predictions.jsonl")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("QuantumLeap.MLEngine")

# Feature names (must match data_collector.py)
FEATURE_NAMES = [
    "close", "open", "high", "low", "volume",
    "rsi", "ema_fast", "ema_slow", "ema_spread",
    "feat_score", "fsm_state", "atr", "compression",
    "liquidity_above", "liquidity_below"
]


class ModelLoader:
    """Cargador seguro de modelos con fallback."""
    
    @staticmethod
    def load_gbm(path: str = None):
        """Carga modelo GBM."""
        path = path or os.path.join(MODELS_DIR, "gbm_v1.joblib")
        try:
            import joblib
            data = joblib.load(path)
            logger.info(f"✅ GBM Model loaded: {path}")
            return data
        except Exception as e:
            logger.warning(f"⚠️ GBM load failed: {e}")
            return None
            
    @staticmethod
    def load_lstm(path: str = None):
        """Carga modelo LSTM."""
        path = path or os.path.join(MODELS_DIR, "lstm_v1.pt")
        try:
            import torch
            data = torch.load(path, map_location="cpu")
            
            # Reconstruir modelo
            from app.ml.train_models import LSTMWithAttention
            config = data["model_config"]
            model = LSTMWithAttention(
                input_dim=config["input_dim"],
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"]
            )
            model.load_state_dict(data["model_state"])
            model.eval()
            
            logger.info(f"✅ LSTM Model loaded: {path}")
            return {"model": model, "config": config}
        except Exception as e:
            logger.warning(f"⚠️ LSTM load failed: {e}")
            return None


class AnomalyDetector:
    """
    Detector de anomalías usando IsolationForest.
    
    Detecta manipulación de mercado (volumen absurdo, movimientos erráticos).
    """
    
    def __init__(self, contamination: float = ANOMALY_CONTAMINATION):
        from sklearn.ensemble import IsolationForest
        
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.is_fitted = False
        self.threshold = 0.7  # Score threshold for alert
        
    def fit(self, X: np.ndarray):
        """Entrena detector con datos normales."""
        self.model.fit(X)
        self.is_fitted = True
        logger.info(f"AnomalyDetector fitted on {len(X)} samples")
        
    def score(self, features: Dict[str, float]) -> float:
        """
        Calcula anomaly score.
        
        Returns:
            Score entre 0 (normal) y 1 (muy anómalo)
        """
        if not self.is_fitted:
            return 0.0
            
        x = np.array([[features.get(k, 0) for k in FEATURE_NAMES]])
        
        # decision_function returns negative for outliers
        raw_score = -self.model.decision_function(x)[0]
        
        # Normalizar a [0, 1]
        normalized = 1 / (1 + np.exp(-raw_score))
        
        return float(normalized)
        
    def is_anomaly(self, features: Dict[str, float]) -> bool:
        """Detecta si es anomalía."""
        return self.score(features) > self.threshold


class ShadowLogger:
    """Logger para predicciones en Shadow Mode."""
    
    def __init__(self, path: str = SHADOW_LOG_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
    def log(self, prediction: Dict):
        """Guarda predicción para análisis posterior."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            **prediction
        }
        
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


class MLEngine:
    """
    Motor principal de ML para inferencia.
    
    Combina:
    - GBM para datos tabulares
    - LSTM para patrones secuenciales
    - IsolationForest para anomalías
    - Shadow Mode para testing sin riesgo
    """
    
    def __init__(self):
        # Cargar modelos
        self.gbm_data = ModelLoader.load_gbm()
        self.lstm_data = ModelLoader.load_lstm()
        
        # Componentes
        self.anomaly_detector = AnomalyDetector()
        self.shadow_logger = ShadowLogger()
        
        # Estado
        self.execution_enabled = EXECUTION_ENABLED
        self.sequence_buffer: List[Dict] = []
        self.seq_len = 32
        
        if self.lstm_data:
            self.seq_len = self.lstm_data["config"].get("seq_len", 32)
            
        logger.info(f"MLEngine initialized. Execution: {self.execution_enabled}")
        
    def predict_gbm(self, features: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Predicción con GBM."""
        if self.gbm_data is None:
            return None
            
        model = self.gbm_data["model"]
        scaler = self.gbm_data["scaler"]
        
        x = np.array([[features.get(k, 0) for k in FEATURE_NAMES]])
        x_scaled = scaler.transform(x)
        
        proba = model.predict_proba(x_scaled)[0]
        
        return {
            "p_loss": float(proba[0]),
            "p_win": float(proba[1]),
            "prediction": "WIN" if proba[1] > 0.5 else "LOSS"
        }
        
    def predict_lstm(self, sequence: List[Dict]) -> Optional[Dict[str, float]]:
        """Predicción con LSTM."""
        if self.lstm_data is None:
            return None
            
        import torch
        
        model = self.lstm_data["model"]
        
        # Construir tensor de secuencia
        seq_array = np.array([
            [s.get(k, 0) for k in FEATURE_NAMES]
            for s in sequence
        ], dtype=np.float32)
        
        x = torch.tensor(seq_array).unsqueeze(0)  # (1, T, D)
        
        with torch.no_grad():
            logits = model(x)
            proba = torch.softmax(logits, dim=-1)[0].numpy()
            
        return {
            "p_loss": float(proba[0]),
            "p_win": float(proba[1]),
            "prediction": "WIN" if proba[1] > 0.5 else "LOSS"
        }
        
    def ensemble_predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predicción combinada (ensemble).
        
        Estrategia:
        - Si LSTM disponible con secuencia completa: usar LSTM (mejor en patrones)
        - Si no: usar GBM
        - Combinar con detección de anomalías
        """
        # Añadir a buffer de secuencia
        self.sequence_buffer.append(features)
        if len(self.sequence_buffer) > self.seq_len:
            self.sequence_buffer = self.sequence_buffer[-self.seq_len:]
            
        # Predicciones individuales
        gbm_pred = self.predict_gbm(features)
        
        lstm_pred = None
        if len(self.sequence_buffer) >= self.seq_len:
            lstm_pred = self.predict_lstm(self.sequence_buffer)
            
        # Detección de anomalías
        anomaly_score = self.anomaly_detector.score(features)
        is_anomaly = anomaly_score > 0.7
        
        # Ensemble: priorizar LSTM si disponible
        if lstm_pred:
            primary = lstm_pred
            source = "LSTM"
        elif gbm_pred:
            primary = gbm_pred
            source = "GBM"
        else:
            primary = {"p_win": 0.5, "prediction": "WAIT"}
            source = "NONE"
            
        # Penalizar si hay anomalía
        p_win = primary.get("p_win", 0.5)
        if is_anomaly:
            p_win *= 0.5  # Reducir confianza en manipulación
            
        # Resultado final
        result = {
            "source": source,
            "p_win": p_win,
            "p_loss": 1 - p_win,
            "prediction": "WIN" if p_win > 0.6 else ("LOSS" if p_win < 0.4 else "WAIT"),
            "confidence": abs(p_win - 0.5) * 2,  # 0-1 scale
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "gbm_available": gbm_pred is not None,
            "lstm_available": lstm_pred is not None,
            "execution_enabled": self.execution_enabled,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Shadow logging
        if not self.execution_enabled:
            self.shadow_logger.log(result)
            logger.info(f"🔮 SHADOW: {result['prediction']} (p={p_win:.3f}, src={source})")
        else:
            logger.info(f"⚡ EXECUTE: {result['prediction']} (p={p_win:.3f})")
            
        # Alertas
        if is_anomaly:
            logger.warning("⚠️ WARNING: MANIPULATION DETECTED")
            
        return result
        
    def get_status(self) -> Dict[str, Any]:
        """Retorna estado del engine."""
        return {
            "gbm_loaded": self.gbm_data is not None,
            "lstm_loaded": self.lstm_data is not None,
            "anomaly_fitted": self.anomaly_detector.is_fitted,
            "execution_enabled": self.execution_enabled,
            "sequence_buffer_size": len(self.sequence_buffer),
            "seq_len_required": self.seq_len
        }


# Singleton instance
ml_engine = MLEngine()


# =============================================================================
# MCP-COMPATIBLE ASYNC WRAPPERS
# =============================================================================

async def predict(features: Dict[str, float]) -> Dict[str, Any]:
    """MCP Tool: Genera predicción ML."""
    return ml_engine.ensemble_predict(features)
    
    
async def get_ml_status() -> Dict[str, Any]:
    """MCP Tool: Estado del sistema ML."""
    return ml_engine.get_status()
    
    
async def enable_execution(enable: bool = True) -> Dict[str, str]:
    """MCP Tool: Activa/desactiva ejecución real."""
    ml_engine.execution_enabled = enable
    mode = "EXECUTION" if enable else "SHADOW"
    logger.warning(f"🔧 Mode changed to: {mode}")
    return {"mode": mode, "execution_enabled": enable}
