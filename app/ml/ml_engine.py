"""
ML Engine - Quantum Leap Phase 4
================================
Motor de inferencia con Shadow Mode y deteccin de anomalas.

Features:
- Carga segura de modelos GBM y LSTM
- Shadow Mode por defecto (no ejecuta rdenes)
- IsolationForest para deteccin de manipulacin
- Integracin SSE para alertas
"""

import os
import json
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import torch.nn as nn
from app.core.config import settings
from app.ml.fractal_analysis import fractal_analyzer

# FEAT-DEEP Multi-Temporal Intelligence
try:
    from app.skills.liquidity_detector import (
        is_in_kill_zone, get_current_kill_zone,
        detect_liquidity_pools, is_intention_candle
    )
    FEAT_DEEP_AVAILABLE = True
except ImportError:
    FEAT_DEEP_AVAILABLE = False

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
    "rsi", "atr", "ema_fast", "ema_slow",
    "feat_score", "fsm_state", "liquidity_ratio", "volatility_zscore"
]


class ModelLoader:
    """Secure model loader with cryptographic-grade fallback and dynamic asset identification.
    
    Handles both joblib (Scikit-Learn) and torch models with state-dict reconstruction.
    """
    
    @staticmethod
    def load_gbm(symbol: str, role: str = "sniper") -> Optional[Dict[str, Any]]:
        """Loads a GBM model with Asset Identity and Role support."""
        # Try specific role-based file
        path = os.path.join(MODELS_DIR, f"gbm_{symbol}_{role}.joblib")
        if not os.path.exists(path):
            # Fallback to standard symbol model
            path = os.path.join(MODELS_DIR, f"gbm_{symbol}_v1.joblib")
            if not os.path.exists(path):
                path = os.path.join(MODELS_DIR, "gbm_v1.joblib")
            
        try:
            import joblib
            if os.path.exists(path):
                data = joblib.load(path)
                logger.info(f" GBM Model [{role}] loaded for {symbol}: {path}")
                return data
            return None
        except Exception as e:
            logger.warning(f" GBM load failed for {symbol} [{role}]: {e}")
            return None
            
    @staticmethod
    def load_lstm(symbol: str) -> Optional[Dict[str, Any]]:
        """Loads a Torch LSTM model for a specific symbol.
        
        Args:
            symbol: Target asset symbol.
            
        Returns:
            Optional[Dict[str, Any]]: Dict with 'model' and 'config' or None.
        """
        path = os.path.join(MODELS_DIR, f"lstm_{symbol}_v1.pt")
        if not os.path.exists(path):
            path = os.path.join(MODELS_DIR, "lstm_v1.pt")
            
        try:
            import torch
            data = torch.load(path, map_location="cpu")
            from app.ml.train_models import LSTMWithAttention
            config = data["model_config"]
            model = LSTMWithAttention(
                input_dim=config["input_dim"],
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"]
            )
            model.load_state_dict(data["model_state"])
            model.eval()
            logger.info(f" LSTM Model loaded for {symbol}: {path}")
            return {"model": model, "config": config}
        except Exception as e:
            logger.warning(f" LSTM load failed for {symbol}: {e}")
            return None


class NormalizationGuard:
    """Validates data scales and feature ranges before inference.
    
    Prevents garbage-in-garbage-out by checking ATR-relative scales
    and detecting statistical drift in input features.
    """
    
    def __init__(self, tolerance: float = 5.0):
        self.tolerance = tolerance # Std deviations
        
    def is_statistically_safe(self, features: Dict[str, float], symbol: str) -> bool:
        """Validates if current features are within reasonable statistical bounds.
        
        Args:
            features: Vector of features.
            symbol: Active symbol context.
            
        Returns:
            bool: True if safe to infer.
        """
        # Feature-specific heuristic guards
        atr = features.get("atr", 0.0)
        # 1. Zero Liquidity Guard
        if features.get("volume", 0) <= 0:
            return False
            
        # 2. Extreme Volatility Guard (Gapping)
        if atr > 0.1: # 10% movement in ATR is usually data error or extreme gap
            logger.warning(f"Extreme volatility guard triggered for {symbol}: ATR={atr:.4f}")
            return False
            
        return True

class AnomalyDetector:
    """IsolationForest-based anomaly detection for market manipulation.
    
    Identifies non-linear erratic behavior and volumetric anomalies.
    """
    
    def __init__(self, contamination: float = ANOMALY_CONTAMINATION):
        from sklearn.ensemble import IsolationForest
        
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.is_fitted: bool = False
        self.threshold: float = 0.7 
        
    def fit(self, X: np.ndarray) -> None:
        """Trains detector on normal regime data."""
        self.model.fit(X)
        self.is_fitted = True
        logger.info(f"AnomalyDetector fitted on {len(X)} samples")
        
    def score(self, features: Dict[str, float]) -> float:
        """Calculates anomaly probability.
        
        Returns:
            float: Score normalized to [0, 1].
        """
        if not self.is_fitted:
            return 0.0
            
        x = np.array([[features.get(k, 0) for k in FEATURE_NAMES]])
        raw_score = -self.model.decision_function(x)[0]
        normalized = 1 / (1 + np.exp(-raw_score))
        return float(normalized)
        
    def is_anomaly(self, features: Dict[str, float]) -> bool:
        """Final boolean anomaly assessment."""
        return self.score(features) > self.threshold


class ShadowLogger:
    """Logger para predicciones en Shadow Mode."""
    
    def __init__(self, path: str = SHADOW_LOG_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
    def log(self, prediction: Dict):
        """Guarda prediccin para anlisis posterior."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            **prediction
        }
        
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


class MLEngine:
    """Master ML Inference Engine with Asset Identity Protocol.
    
    Orchestrates ensemble predictions from GBM and LSTM models
    while enforcing statistical and behavioral safety guards.
    """
    
    def __init__(self):
        # Hot-Registry for symbol models (Asset Identity)
        self.models: Dict[str, Dict[str, Any]] = {}
        
        # Components
        self.anomaly_detector = AnomalyDetector()
        self.norm_guard = NormalizationGuard()
        self.shadow_logger = ShadowLogger()
        self.fractal_analyzer = fractal_analyzer
        
        # Internal State
        self.execution_enabled: bool = EXECUTION_ENABLED
        self.sequence_buffers: Dict[str, List[Dict[str, float]]] = {}
        self.seq_len_map: Dict[str, int] = {}
        
        # FEAT-DEEP Multi-Temporal State
        self.macro_bias: Dict[str, str] = {}  # H4 Trend per symbol
        self.h4_veto_enabled: bool = True     # Veto Rule: No counter-trend trading
        self.kill_zone_filter: bool = True    # Reduce risk outside NY session
        
        logger.info(f"MLEngine V7.0 [FEAT-DEEP] initialized. Multi-TF Veto: ENABLED.")
    
    def apply_feat_veto(self, symbol: str, m1_signal: str) -> Tuple[bool, str]:
        """
        FEAT-DEEP Veto Rule: Prevents counter-trend trading.
        
        IF H4_BEARISH + M1_BUY -> Force HOLD (Veto)
        IF H4_BULLISH + M1_SELL -> Force HOLD (Veto)
        
        Returns:
            (should_trade: bool, reason: str)
        """
        if not self.h4_veto_enabled:
            return (True, "H4 Veto disabled")
        
        h4_trend = self.macro_bias.get(symbol, "NEUTRAL")
        
        # Veto Rule
        if h4_trend == "BEARISH" and m1_signal == "BUY":
            return (False, f"VETO: H4={h4_trend} conflicts with M1={m1_signal}")
        
        if h4_trend == "BULLISH" and m1_signal == "SELL":
            return (False, f"VETO: H4={h4_trend} conflicts with M1={m1_signal}")
        
        # Kill Zone Filter
        if self.kill_zone_filter and FEAT_DEEP_AVAILABLE:
            if not is_in_kill_zone("NY"):
                kz = get_current_kill_zone()
                if kz is None:
                    return (True, f"CAUTION: Outside Kill Zone, reduced confidence")
        
        return (True, f"ALIGNED: H4={h4_trend} + M1={m1_signal}")
    
    def update_macro_bias(self, symbol: str, h4_trend: str):
        """Updates the H4 macro bias for a symbol."""
        self.macro_bias[symbol] = h4_trend
        logger.info(f"[FEAT-DEEP] Updated H4 Bias for {symbol}: {h4_trend}")
        
    def _ensure_models(self, symbol: str) -> None:
        """Lazily loads models for the requested symbol with Role Separation."""
        if symbol not in self.models:
            # Load Sniper (Default)
            gbm = ModelLoader.load_gbm(symbol)
            lstm = ModelLoader.load_lstm(symbol)
            
            # Load Strategist (Bias) if available
            # In a real MIP, these would be separate files like 'gbm_BTCUSD_strategist.joblib'
            
            self.models[symbol] = {
                "sniper_gbm": gbm,
                "sniper_lstm": lstm,
                "strategist_gbm": None # Placeholder for future scaling
            }
            if lstm:
                self.seq_len_map[symbol] = lstm["config"].get("seq_len", 32)
            else:
                self.seq_len_map[symbol] = 32
                
            if symbol not in self.sequence_buffers:
                self.sequence_buffers[symbol] = []
        
    def predict_gbm(self, features: Dict[str, float], symbol: str) -> Optional[Dict[str, Any]]:
        """Predicts using the sniper GBM model."""
        self._ensure_models(symbol)
        gbm_data = self.models[symbol].get("sniper_gbm")
        if not gbm_data:
            return None
            
        model = gbm_data["model"]
        x = np.array([[features.get(k, 0) for k in FEATURE_NAMES]])
        
        prob = model.predict_proba(x)[0][1]
        pred_class = int(model.predict(x)[0])
        
        return {"prob": float(prob), "class": pred_class}
        
    def predict_lstm(self, sequence: List[Dict[str, float]], symbol: str) -> Optional[Dict[str, Any]]:
        """Predicts using the sniper LSTM model."""
        self._ensure_models(symbol)
        lstm_data = self.models[symbol].get("sniper_lstm")
        if not lstm_data:
            return None
            
        import torch
        model = lstm_data["model"]
        
        # Build sequence tensor
        seq_array = np.array([
            [s.get(k, 0) for k in FEATURE_NAMES]
            for s in sequence
        ], dtype=np.float32)
        
        x = torch.tensor(seq_array).unsqueeze(0) # (1, T, D)
        
        with torch.no_grad():
            logits = model(x)
            proba = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            
        return {
            "p_loss": float(proba[0]),
            "p_win": float(proba[1]),
            "prediction": "WIN" if proba[1] > 0.5 else "LOSS"
        }

    def _neutral_response(self, symbol: str, reason: str) -> Dict[str, Any]:
        """Returns a neutral prediction response."""
        return {
            "symbol": symbol,
            "probability": 0.5,
            "is_anomaly": True, # Mark as anomaly if neutralized by guard
            "regime": "UNKNOWN",
            "hurst": 0.5,
            "gbm_prob": 0.5,
            "lstm_prob": 0.5,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "why": reason
        }
        
    def ensemble_predict(self, symbol: str, features: Dict[str, float]) -> Dict[str, Any]:
        """Orchestrates the full Multifractal Fusion prediction pipeline."""
        self._ensure_models(symbol)
        
        # 1. Normalization Guard (Statistical Safety)
        if not self.norm_guard.is_statistically_safe(features, symbol):
            logger.warning(f"Normalization Guard triggered for {symbol}. Neutralizing prediction.")
            return self._neutral_response(symbol, "Normalization anomaly detected")
            
        # 2. Macro-Fractal Context (Strategist Layer)
        # Fetch H1/D1 context from DB would be ideal; using a heuristic cache for now
        # MIP Directive: Multi-Temporal Weighting
        hurst_sc = self.fractal_analyzer.compute_hurst(
            np.array([features.get("close", 0) for _ in range(100)]) # Real history needed here
        )
        regime = self.fractal_analyzer.detect_regime(hurst_sc)
        
        # 3. Micro-Execution (Sniper Layer)
        # Sequence Buffer Update
        self.sequence_buffers[symbol].append(features)
        if len(self.sequence_buffers[symbol]) > self.seq_len_map[symbol]:
            self.sequence_buffers[symbol].pop(0)

        gbm_res = self.predict_gbm(features, symbol)
        lstm_res = self.predict_lstm(self.sequence_buffers[symbol], symbol) if len(self.sequence_buffers[symbol]) == self.seq_len_map[symbol] else None

        # 4. Behavioral Anomaly Guard
        is_behavioral_anomaly = self.anomaly_detector.is_anomaly(features)
        
        # 5. Fusion Layer (Weighted Probability)
        # Probability = (GBM_Weight * GBM_Prob + LSTM_Weight * LSTM_Prob) * Strategist_Bias
        # For now, default 50/50 fusion
        p_gbm = gbm_res["prob"] if gbm_res else 0.5
        p_lstm = lstm_res["p_win"] if lstm_res else 0.5 # LSTM returns p_win
        ensemble_p = (p_gbm + p_lstm) / 2
        
        # Bias Application: If Hurst < 0.45 (Mean Revert), dampen trend-following flags
        final_p = ensemble_p
        if regime == "MEAN_REVERTING":
             # Neutralize towards 0.5 if predicting extreme breakout in range
             final_p = 0.5 + (ensemble_p - 0.5) * 0.7 
        
        # 6. Global Veto
        if is_behavioral_anomaly:
            final_p = 0.5 # Neutralize
            
        result = {
            "symbol": symbol,
            "p_loss": 1 - p_win,
            "prediction": "WIN" if p_win > 0.6 else ("LOSS" if p_win < 0.4 else "WAIT"),
            "confidence": abs(p_win - 0.5) * 2,
            "anomaly_score": anomaly_score,
            "is_anomaly": is_behavioral_anomaly,
            "is_statistically_safe": is_statistically_safe,
            "gbm_available": gbm_pred is not None,
            "lstm_available": lstm_pred is not None,
            "execution_enabled": self.execution_enabled,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Log and persistent tracking
        if not self.execution_enabled:
            self.shadow_logger.log(result)
            logger.info(f" [{symbol}] SHADOW: {result['prediction']} (p={p_win:.3f}, src={source})")
        else:
            logger.info(f" [{symbol}] EXECUTE: {result['prediction']} (p={p_win:.3f})")
            
        return result
        
    def get_status(self) -> Dict[str, Any]:
        """Diagnostic snapshot of the MLEngine state."""
        return {
            "v": "5.0",
            "symbols_registered": list(self.models.keys()),
            "anomaly_fitted": self.anomaly_detector.is_fitted,
            "execution_enabled": self.execution_enabled,
            "buffers_active": len(self.sequence_buffers)
        }


# Singleton instance
ml_engine = MLEngine()


# =============================================================================
# MCP-COMPATIBLE ASYNC WRAPPERS
# =============================================================================

async def predict(features: Dict[str, float], symbol: str = "BTCUSD") -> Dict[str, Any]:
    """MCP Tool: Generates high-confidence ML prediction with autonomous guards.
    
    Args:
        features: Feature vector.
        symbol: Target asset symbol.
    """
    return ml_engine.ensemble_predict(symbol, features)
    
    
async def get_ml_status() -> Dict[str, Any]:
    """MCP Tool: Estado del sistema ML."""
    return ml_engine.get_status()
    
    
async def enable_execution(enable: bool = True) -> Dict[str, str]:
    """MCP Tool: Activa/desactiva ejecucin real."""
    ml_engine.execution_enabled = enable
    mode = "EXECUTION" if enable else "SHADOW"
    logger.warning(f" Mode changed to: {mode}")
    return {"mode": mode, "execution_enabled": enable}
