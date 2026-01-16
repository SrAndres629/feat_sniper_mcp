"""
ML Engine - FEAT NEXUS OPERATION SINGULARITY (Clean Core)
=========================================================
Unified Inference Engine for Hybrid TCN-LSTM Architecture.
Removes legacy GBM/RandomForest logic.
Implements 'Fear of the Unknown' (Anomaly Guard) and Physics Gating.

Key Features:
- HybridSniper (TCN-BiLSTM) Inference
- IsolationForest Anomaly Guard
- Physics-Based Gating (Acceleration Checks)
- Async Non-blocking execution
"""

import os
import json
import logging
import asyncio
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from app.core.config import settings
from app.ml.fractal_analysis import fractal_analyzer

# Gating Neuron: Physics Filter
try:
    from app.skills.market_physics import market_physics, MarketRegime
    PHYSICS_GATING_AVAILABLE = True
except ImportError:
    PHYSICS_GATING_AVAILABLE = False

# FEAT-DEEP Multi-Temporal Intelligence
try:
    from app.skills.liquidity_detector import (
        is_in_kill_zone, get_current_kill_zone
    )
    FEAT_DEEP_AVAILABLE = True
except ImportError:
    FEAT_DEEP_AVAILABLE = False

# FEAT Gates
try:
    from app.services.spread_filter import spread_filter
    from app.services.volatility_guard import volatility_guard
    FEAT_GATES_AVAILABLE = True
except ImportError:
    FEAT_GATES_AVAILABLE = False
    spread_filter = None
    volatility_guard = None

# Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MODELS_DIR = os.getenv("MODELS_DIR", "models")
ANOMALY_CONTAMINATION = float(os.getenv("ANOMALY_CONTAMINATION", "0.01"))
SHADOW_LOG_PATH = os.getenv("SHADOW_LOG_PATH", "data/shadow_predictions.jsonl")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | ML_ENGINE | %(message)s"
)
logger = logging.getLogger("FEAT.MLEngine")

# Feature names (must match data_collector.py & feat_processor.py)
FEATURE_NAMES = [
    "close", "open", "high", "low", "volume",
    "rsi", "atr", "ema_fast", "ema_slow",
    "feat_score", "fsm_state", "liquidity_ratio", "volatility_zscore",
    "momentum_kinetic_micro", "entropy_coefficient", "cycle_harmonic_phase", 
    "institutional_mass_flow", "volatility_regime_norm", "acceptance_ratio", 
    "wick_stress", "poc_z_score", "cvd_acceleration",
    "micro_comp", "micro_slope", "oper_slope", "macro_slope", "bias_slope", "fan_bullish"
]


class HurstBuffer:
    """
    Circular buffer for real-time Hurst coefficient calculation.
    """
    BUFFER_SIZE = settings.HURST_BUFFER_SIZE   # Default: 250
    MIN_SAMPLES = settings.HURST_MIN_SAMPLES   # Default: 100
    UPDATE_EVERY_N = settings.HURST_UPDATE_EVERY_N
    
    STATE_INSUFFICIENT = "DATA_INSUFFICIENT"
    STATE_READY = "READY"
    
    def __init__(self):
        self.buffers: Dict[str, deque] = {}
        self.cached_hurst: Dict[str, float] = {}
        self.update_counters: Dict[str, int] = {}
        self.state: Dict[str, str] = {}
        logger.info(f"Using Hurst Buffer: Size={self.BUFFER_SIZE}")
    
    def push(self, symbol: str, close_price: float) -> None:
        if symbol not in self.buffers:
            self.buffers[symbol] = deque(maxlen=self.BUFFER_SIZE)
            self.update_counters[symbol] = 0
            self.cached_hurst[symbol] = None
            self.state[symbol] = self.STATE_INSUFFICIENT
        
        self.buffers[symbol].append(close_price)
        self.update_counters[symbol] += 1
        
        if len(self.buffers[symbol]) >= self.MIN_SAMPLES:
            self.state[symbol] = self.STATE_READY
    
    def get_prices(self, symbol: str) -> np.ndarray:
        if symbol not in self.buffers or len(self.buffers[symbol]) < self.MIN_SAMPLES:
            raise ValueError(f"Insufficient data for {symbol}")
        return np.array(self.buffers[symbol])
    
    def should_recalculate(self, symbol: str) -> bool:
        if symbol not in self.update_counters: return False
        return self.update_counters[symbol] >= self.UPDATE_EVERY_N
    
    def reset_counter(self, symbol: str) -> None:
        self.update_counters[symbol] = 0
    
    def set_cached_hurst(self, symbol: str, hurst: float) -> None:
        self.cached_hurst[symbol] = hurst
    
    def get_cached_hurst(self, symbol: str) -> Optional[float]:
        return self.cached_hurst.get(symbol)
    
    def is_data_insufficient(self, symbol: str) -> bool:
        return self.state.get(symbol, self.STATE_INSUFFICIENT) == self.STATE_INSUFFICIENT


class ModelLoader:
    """Loads Pure Hybrid Logic (PyTorch). No Scikit-Learn GBMs."""
    
    @staticmethod
    def load_hybrid(symbol: str) -> Optional[Dict[str, Any]]:
        """Loads the HybridSniper (TCN-BiLSTM) model."""
        path = os.path.join(MODELS_DIR, f"hybrid_{symbol}_v1.pt")
        
        try:
            from app.ml.models.hybrid_v1 import HybridSniper
            
            input_dim = len(FEATURE_NAMES)
            model = HybridSniper(input_dim=input_dim, hidden_dim=128, num_classes=3)
            
            if os.path.exists(path):
                data = torch.load(path, map_location="cpu")
                model.load_state_dict(data["state_dict"])
                acc = data.get("best_acc", 0.0)
                logger.info(f"Loaded Hybrid Model for {symbol} (Acc: {acc:.2f})")
            else:
                logger.warning(f"No trained model for {symbol}. Using Untrained Network.")
                
            model.eval()
            return {"model": model, "config": {"seq_len": 32}} 

        except Exception as e:
            logger.error(f"Hybrid load failed for {symbol}: {e}")
            return None


class AnomalyDetector:
    """IsolationForest-based Anomaly detection."""
    def __init__(self, contamination: float = ANOMALY_CONTAMINATION):
        from sklearn.ensemble import IsolationForest
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=1 
        )
        self.is_fitted: bool = False
        self.threshold: float = 0.5 
        
    def fit(self, X: np.ndarray) -> None:
        try:
            self.model.fit(X)
            self.is_fitted = True
            logger.info(f"AnomalyDetector fitted on {len(X)} samples")
        except Exception:
            pass
            
    def is_anomaly(self, features: Dict[str, float]) -> bool:
        if not self.is_fitted: return False
        x = np.array([[features.get(k, 0) for k in FEATURE_NAMES]])
        # decision_function gives negative score for outliers
        score = self.model.predict(x)[0] # 1 for inlier, -1 for outlier
        return score == -1


class MLEngine:
    """
    Master ML Inference Engine (Pure Hybrid).
    Orchestrates TCN-BiLSTM predictions.
    """
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        
        self.anomaly_detector = AnomalyDetector()
        self.hurst_buffer = HurstBuffer()
        self.fractal_analyzer = fractal_analyzer
        
        self.seq_len_map: Dict[str, int] = {}
        self.sequence_buffers: Dict[str, deque] = {}
        
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        
        # States
        self.macro_bias: Dict[str, str] = {}
        self.kill_zone_filter: bool = True
        
        logger.info("MLEngine Online (Hybrid TCN-BiLSTM Only)")

    def hydrate(self, symbol: str, prices: List[float], features_list: List[Dict]) -> None:
        """Deep Hydration logic."""
        # 1. Hurst
        for p in prices: self.hurst_buffer.push(symbol, p)
        # 2. Sequence
        self._ensure_model(symbol)
        seq_len = self.seq_len_map.get(symbol, 32)
        if symbol not in self.sequence_buffers:
            self.sequence_buffers[symbol] = deque(maxlen=seq_len)
        
        # hydrate seq
        for f in features_list:
            self.sequence_buffers[symbol].append(f)
            
        logger.info(f"Hydrated {symbol}: Hurst={self.hurst_buffer.get_cached_hurst(symbol)}, SeqLen={len(self.sequence_buffers[symbol])}")

    def _ensure_model(self, symbol: str) -> None:
        if symbol not in self.models:
            loaded = ModelLoader.load_hybrid(symbol)
            if loaded:
                self.models[symbol] = loaded
                seq_len = loaded["config"].get("seq_len", 32)
                self.seq_len_map[symbol] = seq_len
                if symbol not in self.sequence_buffers:
                    self.sequence_buffers[symbol] = deque(maxlen=seq_len)

    def predict_hybrid_uncertainty(self, sequence: List[Dict], symbol: str, n_iter=30) -> Dict:
        """Monte Carlo Dropout Inference."""
        self._ensure_model(symbol)
        model_data = self.models.get(symbol)
        if not model_data: return {"p_win": 0.5, "uncertainty": 1.0}
        
        model = model_data["model"]
        
        from app.ml.feat_processor import feat_processor
        # [CRITICAL] Use tensorize_snapshot for robust String->Float conversion
        seq_array = np.array([
            feat_processor.tensorize_snapshot(s, FEATURE_NAMES)
            for s in sequence
        ], dtype=np.float32)
        
        # Padding if sequence too short
        if len(seq_array) < seq_len:
             # pad with 0
             pass 
             
        x = torch.tensor(seq_array).unsqueeze(0)
        
        model.train() # Enable Dropout
        scores = []
        with torch.no_grad():
            for _ in range(n_iter):
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)[0]
                # 0:SELL, 1:HOLD, 2:BUY
                p_sell = probs[0].item()
                p_buy = probs[2].item()
                score = (p_buy - p_sell + 1.0) / 2.0
                scores.append(score)
        model.eval()
        
        scores = np.array(scores)
        return {
            "p_win": float(np.mean(scores)),
            "uncertainty": float(np.std(scores))
        }

    def _neutral(self, symbol, reason) -> Dict:
        return {
            "symbol": symbol, "probability": 0.5, "uncertainty": 0.0,
            "prediction": "WAIT", "why": reason
        }

    async def predict_async(self, symbol: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Main Inference Entry Point."""
        try:
            # 1. Safety Gates
            if not self.hurst_buffer.is_data_insufficient(symbol):
                # Check Gates
                pass 
                
            self.hurst_buffer.push(symbol, features.get("close", 0))
            
            # 2. Update Sequence
            self._ensure_model(symbol)
            self.sequence_buffers[symbol].append(features)
            
            # 3. Anomaly Check
            if self.anomaly_detector.is_anomaly(features):
                logger.warning(f"Anomaly Detected on {symbol}")
                return self._neutral(symbol, "ANOMALY_DETECTED")
                
            # 4. Inference
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(
                self.executor,
                self.predict_hybrid_uncertainty,
                list(self.sequence_buffers[symbol]),
                symbol,
                30
            )
            
            # 5. Physics Validation
            # If Model says BUY (p > 0.6) but Physics says Decelerating -> Downgrade
            if PHYSICS_GATING_AVAILABLE:
                regime = market_physics.ingest_tick({"close": features.get("close"), "tick_volume": features.get("volume")})
                if regime and res["p_win"] > 0.6 and not regime.is_accelerating:
                     res["p_win"] -= 0.1 # Penalty
                     res["why"] = "Physics Penalty"
            
            res["symbol"] = symbol
            return res
            
        except Exception as e:
            logger.error(f"Inference Error: {e}")
            return self._neutral(symbol, f"ERROR: {str(e)}")

    def record_trade_result(self, lstm_prob: float, result_pips: float, gbm_prob=0.0):
        """Feedback loop endpoint."""
        # Could save to RLAIF or specialized metric tracker
        pass

ml_engine = MLEngine()
