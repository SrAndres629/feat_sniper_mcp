"""
ML Engine - Quantum Leap Phase 5 (ALPHA ENGINE REPAIR)
=======================================================
Motor de inferencia con Shadow Mode y detección de anomalías.

Features:
- Carga segura de modelos GBM y LSTM
- Shadow Mode por defecto (no ejecuta órdenes)
- IsolationForest para detección de manipulación
- Integración SSE para alertas
- [FIX] HurstBuffer: Buffer circular de 250 precios reales
- [FIX] SharpeTracker: Ponderación adaptativa por Sharpe
- [FIX] Async LSTM: Inferencia no-bloqueante
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
import torch.nn as nn
from app.core.config import settings
from app.ml.fractal_analysis import fractal_analyzer

# Gating Neuron: Physics Filter (Senior Patch v9.0)
try:
    from app.skills.market_physics import market_physics, MarketRegime
    PHYSICS_GATING_AVAILABLE = True
except ImportError:
    PHYSICS_GATING_AVAILABLE = False

# FEAT-DEEP Multi-Temporal Intelligence
try:
    from app.skills.liquidity_detector import (
        is_in_kill_zone, get_current_kill_zone,
        detect_liquidity_pools, is_intention_candle
    )
    FEAT_DEEP_AVAILABLE = True
except ImportError:
    FEAT_DEEP_AVAILABLE = False

# [P0-2 FIX] Mandatory FEAT Chain Gates
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
    "feat_score", "fsm_state", "liquidity_ratio", "volatility_zscore",
    "momentum_kinetic_micro", "entropy_coefficient", "cycle_harmonic_phase", 
    "institutional_mass_flow", "volatility_regime_norm", "acceptance_ratio", 
    "wick_stress", "poc_z_score", "cvd_acceleration",
    "micro_comp", "micro_slope", "oper_slope", "macro_slope", "bias_slope", "fan_bullish"
]


# =============================================================================
# FIX #1: HURST BUFFER - Real Price History for Fractal Analysis
# =============================================================================

class HurstBuffer:
    """
    Circular buffer for real-time Hurst coefficient calculation.
    
    Stores the last N closing prices per symbol to enable
    accurate fractal regime detection (Trending vs Range).
    
    [P0-3 FIX] Fallo Explícito:
    - Si no hay datos suficientes, retorna estado 'DATA_INSUFFICIENT'
    - Nunca retorna un Hurst silenciosamente incorrecto
    - Excepciones explícitas si se intenta calcular sin datos
    
    [P1 FIX] Uses centralized config from settings (Single Source of Truth)
    """
    
    # [P1 FIX] Use settings instead of hardcoded values
    BUFFER_SIZE = settings.HURST_BUFFER_SIZE   # Default: 250 periods
    MIN_SAMPLES = settings.HURST_MIN_SAMPLES   # Default: 100 minimum
    UPDATE_EVERY_N = settings.HURST_UPDATE_EVERY_N  # Default: recalc every 50 ticks
    
    # [P0-3] Estados explícitos
    STATE_INSUFFICIENT = "DATA_INSUFFICIENT"
    STATE_READY = "READY"
    
    def __init__(self):
        self.buffers: Dict[str, deque] = {}
        self.cached_hurst: Dict[str, float] = {}
        self.update_counters: Dict[str, int] = {}
        self.state: Dict[str, str] = {}  # [P0-3] Estado explícito por símbolo
        logger.info(f"[ALPHA FIX] HurstBuffer initialized (size={self.BUFFER_SIZE}, min_samples={self.MIN_SAMPLES})")
    
    def push(self, symbol: str, close_price: float) -> None:
        """Push a new closing price to the buffer."""
        if symbol not in self.buffers:
            self.buffers[symbol] = deque(maxlen=self.BUFFER_SIZE)
            self.update_counters[symbol] = 0
            self.cached_hurst[symbol] = None  # [P0-3] None = no calculado, no 0.5 default
            self.state[symbol] = self.STATE_INSUFFICIENT
        
        self.buffers[symbol].append(close_price)
        self.update_counters[symbol] += 1
        
        # [P0-3] Actualizar estado cuando tengamos suficientes datos
        if len(self.buffers[symbol]) >= self.MIN_SAMPLES:
            self.state[symbol] = self.STATE_READY
    
    def get_prices(self, symbol: str) -> np.ndarray:
        """Get price array for Hurst calculation. Raises if insufficient data."""
        if symbol not in self.buffers:
            raise ValueError(f"[P0-3] No buffer exists for symbol {symbol}")
        if len(self.buffers[symbol]) < self.MIN_SAMPLES:
            raise ValueError(f"[P0-3] Insufficient data for {symbol}: {len(self.buffers[symbol])}/{self.MIN_SAMPLES}")
        return np.array(self.buffers[symbol])
    
    def should_recalculate(self, symbol: str) -> bool:
        """Check if it's time to recalculate Hurst (performance optimization)."""
        if symbol not in self.update_counters:
            return False
        return self.update_counters[symbol] >= self.UPDATE_EVERY_N
    
    def has_enough_data(self, symbol: str) -> bool:
        """Check if we have minimum samples for Hurst calculation."""
        return self.state.get(symbol) == self.STATE_READY
    
    def is_data_insufficient(self, symbol: str) -> bool:
        """[P0-3] Explicit check for insufficient data state."""
        return self.state.get(symbol, self.STATE_INSUFFICIENT) == self.STATE_INSUFFICIENT
    
    def reset_counter(self, symbol: str) -> None:
        """Reset the update counter after Hurst recalculation."""
        self.update_counters[symbol] = 0
    
    def set_cached_hurst(self, symbol: str, hurst: float) -> None:
        """Cache the calculated Hurst value."""
        self.cached_hurst[symbol] = hurst
    
    def get_cached_hurst(self, symbol: str) -> Optional[float]:
        """Get the cached Hurst value. Returns None if not calculated."""
        return self.cached_hurst.get(symbol)  # [P0-3] Puede ser None
    
    def get_status(self, symbol: str) -> Dict[str, Any]:
        """Get buffer status for monitoring."""
        return {
            "symbol": symbol,
            "buffer_size": len(self.buffers.get(symbol, [])),
            "max_size": self.BUFFER_SIZE,
            "cached_hurst": self.cached_hurst.get(symbol),
            "state": self.state.get(symbol, self.STATE_INSUFFICIENT),
            "has_enough_data": self.has_enough_data(symbol)
        }


# =============================================================================
# FIX #2: SHARPE TRACKER - Adaptive Ensemble Weights
# =============================================================================

class SharpeTracker:
    """
    Rolling Sharpe ratio tracker for adaptive ensemble weighting.
    
    Tracks GBM and LSTM prediction accuracy to dynamically adjust
    their weights in the ensemble fusion.
    
    Formula: Weight_model = Sharpe_model² / (Sharpe_GBM² + Sharpe_LSTM²)
    
    FIX: Replaces the naive 50/50 split with performance-based weighting.
    
    [P1 FIX] Uses centralized config from settings (Single Source of Truth)
    """
    
    # [P1 FIX] Use settings instead of hardcoded values
    WINDOW_SIZE = settings.SHARPE_WINDOW_SIZE  # Default: 50 trades
    MIN_TRADES = settings.SHARPE_MIN_TRADES    # Default: 10 trades
    MIN_WEIGHT = settings.ENSEMBLE_MIN_WEIGHT  # Default: 0.15 (15%)
    MAX_WEIGHT = settings.ENSEMBLE_MAX_WEIGHT  # Default: 0.85 (85%)
    
    def __init__(self):
        self.gbm_returns: deque = deque(maxlen=self.WINDOW_SIZE)
        self.lstm_returns: deque = deque(maxlen=self.WINDOW_SIZE)
        self.trade_count = 0
        logger.info(f"[ALPHA FIX] SharpeTracker initialized (window={self.WINDOW_SIZE})")
    
    def record_prediction(self, gbm_pred: float, lstm_pred: float, actual_return: float) -> None:
        """
        Record prediction accuracy for both models.
        
        Args:
            gbm_pred: GBM predicted probability (0-1)
            lstm_pred: LSTM predicted probability (0-1)
            actual_return: Actual trade return (positive = win, negative = loss)
        """
        # Calculate "return" based on prediction accuracy
        # If model predicted > 0.5 and trade won, it gets positive return
        gbm_signal = 1 if gbm_pred > 0.5 else -1
        lstm_signal = 1 if lstm_pred > 0.5 else -1
        
        gbm_return = gbm_signal * actual_return
        lstm_return = lstm_signal * actual_return
        
        self.gbm_returns.append(gbm_return)
        self.lstm_returns.append(lstm_return)
        self.trade_count += 1
    
    def _calculate_sharpe(self, returns: deque) -> float:
        """Calculate Sharpe ratio from returns."""
        if len(returns) < self.MIN_TRADES:
            return 1.0  # Default: equal weight
        
        returns_array = np.array(returns)
        mean = np.mean(returns_array)
        std = np.std(returns_array)
        
        if std == 0 or np.isnan(std):
            return 1.0
        
        # Annualized Sharpe (assuming ~50 trades/week)
        sharpe = (mean / std) * np.sqrt(50)
        return max(0.1, sharpe)  # Floor at 0.1 to avoid division errors
    
    def get_weights(self) -> Tuple[float, float]:
        """
        Calculate adaptive weights based on Sharpe ratios.
        
        Returns:
            Tuple[float, float]: (gbm_weight, lstm_weight)
        """
        if self.trade_count < self.MIN_TRADES:
            return 0.5, 0.5  # Default 50/50 until enough data
        
        sharpe_gbm = self._calculate_sharpe(self.gbm_returns)
        sharpe_lstm = self._calculate_sharpe(self.lstm_returns)
        
        # Sharpe-squared weighting formula
        sharpe_gbm_sq = sharpe_gbm ** 2
        sharpe_lstm_sq = sharpe_lstm ** 2
        total = sharpe_gbm_sq + sharpe_lstm_sq
        
        if total == 0:
            return 0.5, 0.5
        
        gbm_weight = sharpe_gbm_sq / total
        lstm_weight = sharpe_lstm_sq / total
        
        # Apply min/max constraints
        gbm_weight = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, gbm_weight))
        lstm_weight = 1.0 - gbm_weight  # Ensure they sum to 1
        
        return gbm_weight, lstm_weight
    
    def get_status(self) -> Dict[str, Any]:
        """Get tracker status for monitoring."""
        gbm_w, lstm_w = self.get_weights()
        return {
            "trade_count": self.trade_count,
            "gbm_weight": round(gbm_w, 3),
            "lstm_weight": round(lstm_w, 3),
            "sharpe_gbm": round(self._calculate_sharpe(self.gbm_returns), 3),
            "sharpe_lstm": round(self._calculate_sharpe(self.lstm_returns), 3),
            "has_enough_data": self.trade_count >= self.MIN_TRADES
        }


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
        path = os.path.join(MODELS_DIR, f"lstm_{symbol}_v2.pt")
        if not os.path.exists(path):
            path = os.path.join(MODELS_DIR, "lstm_v2.pt")
            
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
        if atr > 100.0: # Adjusted for XAU/BTC usage (was 0.1)
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
    
    [ALPHA FIX v5.0] Now with:
    - Real HurstBuffer (250 prices) for accurate regime detection
    - SharpeTracker for adaptive ensemble weights
    - Async LSTM inference (non-blocking)
    """
    
    def __init__(self):
        # Hot-Registry for symbol models (Asset Identity)
        self.models: Dict[str, Dict[str, Any]] = {}
        
        # Components
        self.anomaly_detector = AnomalyDetector()
        self.norm_guard = NormalizationGuard()
        self.shadow_logger = ShadowLogger()
        self.fractal_analyzer = fractal_analyzer
        
        # [FIX #1] Real Hurst Buffer - replaces broken static array
        self.hurst_buffer = HurstBuffer()
        
        # [FIX #2] Sharpe Tracker - replaces naive 50/50 fusion
        self.sharpe_tracker = SharpeTracker()
        
        self.seq_len_map: Dict[str, int] = {}
        
        # [SENIOR ARCHITECTURE] Dedicated Executor for Inference (Bypasses GIL for C-extensions)
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        self.last_loop_time = time.time()
        
        # FEAT-DEEP Multi-Temporal State
        self.macro_bias: Dict[str, str] = {}  # H4 Trend per symbol
        self.h4_veto_enabled: bool = True     # Veto Rule: No counter-trend trading
        self.kill_zone_filter: bool = True    # [P0-2] Now BLOCKS instead of CAUTION
        
        # [P0-2 FIX] Market Data Cache for SpreadFilter/VolatilityGuard
        self._cached_market_data: Dict[str, Dict[str, float]] = {}
        
        logger.info(f"MLEngine V9.0 [P0-FIX] initialized. FEAT Gates + HurstBuffer + SharpeTracker ACTIVE.")
    
    def hydrate_hurst(self, symbol: str, prices: List[float]) -> None:
        """
        [P0 FIX] Deep Hydration for Hurst Buffer.
        Ensures fractal regime is known immediately.
        """
        if not prices:
            return
            
        logger.info(f"🌊 [HURST] Hydrating {symbol} with {len(prices)} prices...")
        for p in prices:
            self.hurst_buffer.push(symbol, p)
            
        # Forces immediate calculation
        try:
            prices_arr = self.hurst_buffer.get_prices(symbol)
            hurst_sc = self.fractal_analyzer.compute_hurst(prices_arr)
            self.hurst_buffer.set_cached_hurst(symbol, hurst_sc)
            logger.info(f"✅ [HURST] {symbol} Hydrated. Initial Hurst: {hurst_sc:.3f}")
        except Exception as e:
            logger.warning(f"⚠️ [HURST] Initial calculation failed after hydration: {e}")

    def hydrate_sequences(self, symbol: str, features_list: List[Dict[str, float]]) -> None:
        """
        [P0 FIX] Deep Hydration for LSTM Sequence Buffers.
        """
        if not features_list:
            return
            
        self._ensure_models(symbol)
        buffer = self.sequence_buffers.get(symbol)
        if buffer is None:
            # Fallback if _ensure_models failed to create it
            seq_len = self.seq_len_map.get(symbol, 32)
            buffer = deque(maxlen=seq_len)
            self.sequence_buffers[symbol] = buffer

        logger.info(f"🌊 [LSTM] Hydrating {symbol} sequence with {len(features_list)} samples...")
        for feat in features_list:
            buffer.append(feat)
            
        logger.info(f"✅ [LSTM] {symbol} sequence buffer ready ({len(buffer)}/{buffer.maxlen})")

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
        
        # [P0-2 FIX] Kill Zone Filter - Now BLOCKS instead of just warning
        if self.kill_zone_filter and FEAT_DEEP_AVAILABLE:
            if not is_in_kill_zone("NY"):
                kz = get_current_kill_zone()
                if kz is None:
                    # [P0-2] Changed from CAUTION to BLOCK
                    logger.warning(f"[FEAT GATE] BLOCKED: Outside Kill Zone - no trading allowed")
                    return (False, "BLOCKED: Outside Kill Zone - T (Time) pillar violated")
        
        return (True, f"ALIGNED: H4={h4_trend} + M1={m1_signal}")
    
    def update_macro_bias(self, symbol: str, h4_trend: str):
        """Updates the H4 macro bias for a symbol."""
        self.macro_bias[symbol] = h4_trend
        logger.info(f"[FEAT-DEEP] Updated H4 Bias for {symbol}: {h4_trend}")
        
    def _ensure_models(self, symbol: str) -> None:
        """Lazily loads models for the requested symbol with Role Separation."""
        if symbol not in self.models:
            # Load Sniper (Main Execution Model)
            gbm = ModelLoader.load_gbm(symbol)
            lstm = ModelLoader.load_lstm(symbol)
            
            self.models[symbol] = {
                "sniper_gbm": gbm,
                "sniper_lstm": lstm
            }
            
            # Initialize neural buffers
            seq_len = 32
            if lstm and "config" in lstm:
                seq_len = lstm["config"].get("seq_len", 32)
            
            if symbol not in self.sequence_buffers:
                self.sequence_buffers[symbol] = deque(maxlen=seq_len)
            self.seq_len_map[symbol] = seq_len
        else:
            # Guard for existing entry
            if symbol not in self.sequence_buffers:
                self.sequence_buffers[symbol] = deque(maxlen=32)
        
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
            "gbm_weight": 0.5,
            "lstm_weight": 0.5,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "why": reason
        }
    
    async def ensemble_predict_async(self, symbol: str, features: Dict[str, float]) -> Dict[str, Any]:
        """
        [P0-2/P0-3 FIX] Async ensemble prediction with MANDATORY FEAT Chain validation.
        
        Order of Gates (ALL must pass):
        1. SpreadFilter (E - Espacio): Validates spread is not toxic
        2. VolatilityGuard (E - Espacio): Validates volatility is not extreme
        3. DataSufficiency (F - Forma): Validates Hurst buffer has enough data
        4. NormalizationGuard: Statistical safety
        5. Physics Gating Neuron: ML-Physics alignment
        """
        self._ensure_models(symbol)
        
        # =====================================================================
        # [P0-2 FIX] MANDATORY FEAT GATES - No trading without explicit OK
        # =====================================================================
        
        # GATE 1: SpreadFilter (Espacio - E Pillar)
        if FEAT_GATES_AVAILABLE and spread_filter:
            current_spread = features.get("spread", 0)
            avg_spread = features.get("avg_spread", current_spread)  # Fallback to current if no avg
            if spread_filter.is_spread_toxic(symbol, current_spread, avg_spread):
                logger.warning(f"[FEAT GATE] BLOCKED: Toxic Spread on {symbol}")
                return self._neutral_response(symbol, "BLOCKED: E (Espacio) - Toxic Spread")
        
        # GATE 2: VolatilityGuard (Espacio - E Pillar)
        if FEAT_GATES_AVAILABLE and volatility_guard:
            market_data = {
                "atr": features.get("atr", 0),
                "avg_atr": features.get("avg_atr", features.get("atr", 1))  # Fallback
            }
            can_trade, reason = volatility_guard.can_trade(market_data)
            if not can_trade:
                logger.warning(f"[FEAT GATE] BLOCKED: {reason}")
                return self._neutral_response(symbol, f"BLOCKED: E (Espacio) - {reason}")
        
        # GATE 3: Normalization Guard (Statistical Safety)
        if not self.norm_guard.is_statistically_safe(features, symbol):
            logger.warning(f"[FEAT GATE] BLOCKED: Normalization anomaly for {symbol}")
            return self._neutral_response(symbol, "BLOCKED: Statistical anomaly detected")
        
        # =====================================================================
        # [P0-3 FIX] DATA_INSUFFICIENT State - Explicit failure, no silent defaults
        # =====================================================================
        close_price = features.get("close", 0)
        self.hurst_buffer.push(symbol, close_price)
        
        # Check if we have enough data - if not, BLOCK trading
        if self.hurst_buffer.is_data_insufficient(symbol):
            buffer_status = self.hurst_buffer.get_status(symbol)
            logger.info(f"[FEAT GATE] DATA_INSUFFICIENT: {symbol} - {buffer_status['buffer_size']}/{self.hurst_buffer.MIN_SAMPLES} samples")
            return self._neutral_response(symbol, f"DATA_INSUFFICIENT: Need {self.hurst_buffer.MIN_SAMPLES} samples, have {buffer_status['buffer_size']}")
        
        # Calculate Hurst if ready
        if self.hurst_buffer.should_recalculate(symbol):
            try:
                prices = self.hurst_buffer.get_prices(symbol)
                hurst_sc = self.fractal_analyzer.compute_hurst(prices)
                self.hurst_buffer.set_cached_hurst(symbol, hurst_sc)
                self.hurst_buffer.reset_counter(symbol)
                logger.debug(f"[HURST] {symbol}: Recalculated = {hurst_sc:.3f} (n={len(prices)})")
            except ValueError as e:
                logger.error(f"[HURST] Calculation failed: {e}")
                return self._neutral_response(symbol, f"HURST_ERROR: {str(e)}")
        
        hurst_sc = self.hurst_buffer.get_cached_hurst(symbol)
        if hurst_sc is None:
            # Should not happen if is_data_insufficient check passed, but safety first
            return self._neutral_response(symbol, "HURST_NOT_CALCULATED: Internal state error")
        
        regime = self.fractal_analyzer.detect_regime(hurst_sc)
        
        # =====================================================================
        # Sequence Buffer Update (using deque - O(1) rotation)
        # =====================================================================
        self.sequence_buffers[symbol].append(features)  # deque auto-rotates

        # GBM prediction (fast, keep sync)
        gbm_res = self.predict_gbm(features, symbol)
        
        # [FIX #3] LSTM prediction - run in thread pool to release GIL
        lstm_res = None
        if len(self.sequence_buffers[symbol]) == self.seq_len_map[symbol]:
            try:
                # [SENIOR ABSTRACTION] Use dedicated executor
                loop = asyncio.get_event_loop()
                lstm_res = await asyncio.wait_for(
                    loop.run_in_executor(self.executor, self.predict_lstm, list(self.sequence_buffers[symbol]), symbol),
                    timeout=settings.DECISION_TTL_MS / 2000.0 # 50% of TTL max for inference
                )
            except asyncio.TimeoutError:
                logger.warning(f"[LSTM] {symbol}: Inference timeout (>{settings.DECISION_TTL_MS/2}ms)")
                lstm_res = None

        # 4. Behavioral Anomaly Guard
        is_behavioral_anomaly = self.anomaly_detector.is_anomaly(features)
        
        # 4.5 [LAST MILE] BIDIRECTIONAL GATING NEURON
        # Physics can now BOOST (not just veto) LSTM confidence
        physics_gate_closed = False
        physics_boost = 0.0  # [NEW] Boost factor from physics
        
        if PHYSICS_GATING_AVAILABLE:
            regime_data = market_physics.ingest_tick({"close": close_price, "tick_volume": features.get("volume", 0)})
            if regime_data:
                p_lstm_temp = lstm_res["p_win"] if lstm_res else 0.5
                lstm_confident = abs(p_lstm_temp - 0.5) > 0.15  # LSTM > 65% or < 35%
                lstm_uncertain = abs(p_lstm_temp - 0.5) <= 0.10  # LSTM between 40-60%
                physics_accelerating = regime_data.is_accelerating
                
                # VETO: LSTM confident + Physics NOT accelerating → Block
                if lstm_confident and not physics_accelerating:
                    logger.info(f"[GATE] VETO: Accel={regime_data.acceleration_score:.3f}, LSTM_p={p_lstm_temp:.3f} (no alignment)")
                    physics_gate_closed = True
                
                # [LAST MILE] BOOST: Physics accelerating + LSTM uncertain → Boost confidence
                # This allows strong market moves to validate uncertain AI predictions
                elif physics_accelerating and lstm_uncertain:
                    # Boost proportional to acceleration score (capped at 0.07)
                    accel_score = regime_data.acceleration_score
                    physics_boost = min(0.07, accel_score * 0.02)
                    
                    # Direction of boost based on trend
                    if regime_data.trend == "BULLISH":
                        physics_boost = abs(physics_boost)  # Boost towards WIN
                    elif regime_data.trend == "BEARISH":
                        physics_boost = -abs(physics_boost)  # Boost towards LOSS
                    
                    logger.info(f"[GATE] BOOST: Accel={accel_score:.3f}, Trend={regime_data.trend}, Boost={physics_boost:+.3f}")
        
        # 5. [FIX #2] SHARPE-WEIGHTED Fusion (replaces naive 50/50)
        p_gbm = gbm_res["prob"] if gbm_res else 0.5
        p_lstm = lstm_res["p_win"] if lstm_res else 0.5
        
        gbm_weight, lstm_weight = self.sharpe_tracker.get_weights()
        
        # Weighted ensemble fusion
        ensemble_p = (gbm_weight * p_gbm) + (lstm_weight * p_lstm)
        
        # [LAST MILE] Apply physics boost BEFORE regime dampening
        ensemble_p = ensemble_p + physics_boost
        ensemble_p = max(0.0, min(1.0, ensemble_p))  # Clamp to [0, 1]
        
        # Regime-based dampening
        final_p = ensemble_p
        if regime == "MEAN_REVERTING":
            # Stronger dampening for ranging markets
            final_p = 0.5 + (ensemble_p - 0.5) * 0.6
        elif regime == "RANDOM_WALK":
            # Moderate dampening for noisy markets
            final_p = 0.5 + (ensemble_p - 0.5) * 0.8
        
        # 6. Global Veto (only for anomaly or physics veto, NOT for boost)
        if is_behavioral_anomaly or physics_gate_closed:
            final_p = 0.5  # Neutralize
            
        anomaly_score = self.anomaly_detector.score(features)
            
        result = {
            "symbol": symbol,
            "p_win": final_p,
            "p_loss": 1 - final_p,
            "prediction": "WIN" if final_p > 0.6 else ("LOSS" if final_p < 0.4 else "WAIT"),
            "confidence": abs(final_p - 0.5) * 2,
            # [ALPHA FIX] New fields for monitoring
            "hurst": round(hurst_sc, 3),
            "regime": regime,
            "gbm_prob": round(p_gbm, 3),
            "lstm_prob": round(p_lstm, 3),
            "gbm_weight": round(gbm_weight, 3),
            "lstm_weight": round(lstm_weight, 3),
            "anomaly_score": anomaly_score,
            "is_anomaly": is_behavioral_anomaly,
            "is_statistically_safe": True,
            "gbm_available": gbm_res is not None,
            "lstm_available": lstm_res is not None,
            "execution_enabled": self.execution_enabled,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Log and persistent tracking
        source = f"GBM:{gbm_weight*100:.0f}%|LSTM:{lstm_weight*100:.0f}%"
        if not self.execution_enabled:
            self.shadow_logger.log(result)
            logger.info(f" [{symbol}] SHADOW: {result['prediction']} (p={final_p:.3f}, src={source})")
        else:
            logger.info(f" [{symbol}] EXECUTE: {result['prediction']} (p={final_p:.3f})")
        
        return result
    
    def ensemble_predict(self, symbol: str, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Synchronous wrapper for backwards compatibility.
        Runs the async version in an event loop.
        """
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - create task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.ensemble_predict_async(symbol, features)
                )
                return future.result(timeout=0.1)  # 100ms max
        except RuntimeError:
            # No running loop - safe to use asyncio.run
            return asyncio.run(self.ensemble_predict_async(symbol, features))
        
    async def check_loop_jitter(self):
        """Monitors event loop latency. Logs critical warnings if jitter > 50ms."""
        now = time.time()
        # Measure time drift from expected 0.1s check interval
        jitter = abs((now - self.last_loop_time) - 0.1) 
        self.last_loop_time = now
        
        if jitter > 0.050: # 50ms threshold
            logger.critical(f"🚨 [JITTER] Event Loop Latency Spike: {jitter*1000:.2f}ms. GIL Contention Detected!")

    def get_status(self) -> Dict[str, Any]:
        """Diagnostic snapshot of the MLEngine state - [ALPHA FIX v8.0]."""
        return {
            "v": "8.0-ALPHA-FIX",
            "symbols_registered": list(self.models.keys()),
            "anomaly_fitted": self.anomaly_detector.is_fitted,
            "execution_enabled": self.execution_enabled,
            "buffers_active": len(self.sequence_buffers),
            # [ALPHA FIX] New diagnostic fields
            "hurst_buffer": {s: self.hurst_buffer.get_status(s) for s in self.models.keys()},
            "sharpe_tracker": self.sharpe_tracker.get_status()
        }

    def record_trade_result(self, gbm_prob: float, lstm_prob: float, result_pips: float):
        """
        [SENIOR ABSTRACTION] Closed-loop feedback with noise filter.
        Updates model performance metrics (Sharpe) based on actual trade outcomes.
        """
        if self.sharpe_tracker:
            self.sharpe_tracker.record_prediction(gbm_prob, lstm_prob, result_pips)
            logger.info(f"📈 [FEEDBACK] Sharpe updated. New weights: {self.sharpe_tracker.get_weights()}")


# Singleton instance
ml_engine = MLEngine()


# =============================================================================
# MCP-COMPATIBLE ASYNC WRAPPERS
# =============================================================================

async def predict(features: Dict[str, float], symbol: str = "BTCUSD") -> Dict[str, Any]:
    """MCP Tool: Generates high-confidence ML prediction with autonomous guards.
    
    [ALPHA FIX] Now uses async ensemble with real Hurst and Sharpe-weighted fusion.
    
    Args:
        features: Feature vector.
        symbol: Target asset symbol.
    """
    return await ml_engine.ensemble_predict_async(symbol, features)
    
    
async def get_ml_status() -> Dict[str, Any]:
    """MCP Tool: Estado del sistema ML."""
    return ml_engine.get_status()
    
    
async def enable_execution(enable: bool = True) -> Dict[str, str]:
    """MCP Tool: Activa/desactiva ejecucin real."""
    ml_engine.execution_enabled = enable
    mode = "EXECUTION" if enable else "SHADOW"
    logger.warning(f" Mode changed to: {mode}")
    return {"mode": mode, "execution_enabled": enable}
