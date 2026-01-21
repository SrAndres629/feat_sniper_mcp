import os
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings
from app.ml.models.anomaly import anomaly_detector
from app.ml.fractal_analysis import fractal_analyzer

from .fractal import HurstBuffer
from .loader import ModelLoader
from .inference import InferenceEngine
from .rlaif import ExperienceReplay
from .validators import validate_24ch_integrity, DataIntegrityError  # V6 Circuit Breaker

logger = logging.getLogger("FEAT.MLEngine")

try:
    from app.skills.market_physics import market_physics
    PHYSICS_AVAILABLE = True
except ImportError: PHYSICS_AVAILABLE = False

class MLEngine:
    """Master ML Inference Engine (Pure Hybrid TCN-BiLSTM)."""
    
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.seq_len_map: Dict[str, int] = {}
        self.sequence_buffers: Dict[str, deque] = {}
        
        self.hurst_buffer = HurstBuffer()
        self.inference_engine = InferenceEngine()
        self.replay = ExperienceReplay()
        
        # Singletons
        self.anomaly_detector = anomaly_detector
        self.fractal_analyzer = fractal_analyzer
        
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        self.feature_names = list(settings.NEURAL_FEATURE_NAMES)
        
        logger.info("MLEngine Online (Hybrid TCN-BiLSTM Only)")

    def _ensure_model(self, symbol: str) -> None:
        if symbol not in self.models:
            loaded = ModelLoader.load_hybrid(symbol)
            if loaded:
                self.models[symbol] = loaded
                seq_len = loaded["config"].get("seq_len", 32)
                self.seq_len_map[symbol] = seq_len
                if symbol not in self.sequence_buffers:
                    self.sequence_buffers[symbol] = deque(maxlen=seq_len)

    def hydrate(self, symbol: str, prices: List[float], features_list: List[Dict]) -> None:
        for p in prices: self.hurst_buffer.push(symbol, p)
        self._ensure_model(symbol)
        
        if symbol not in self.sequence_buffers:
             self.sequence_buffers[symbol] = deque(maxlen=self.seq_len_map.get(symbol, 32))
             
        for f in features_list:
            self.sequence_buffers[symbol].append(f)
        logger.info(f"Hydrated {symbol}: Hurst={self.hurst_buffer.get_cached_hurst(symbol)}, Seq={len(self.sequence_buffers[symbol])}")

    @validate_24ch_integrity  # V6 CIRCUIT BREAKER - Enforces 24-channel integrity
    async def predict_async(self, symbol: str, features: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.hurst_buffer.push(symbol, features.get("close", 0))
            self._ensure_model(symbol)
            self.sequence_buffers[symbol].append(features)
            
            if self.anomaly_detector.is_anomaly(features):
                logger.critical(f"🦠 TOXIC DATA DETECTED on {symbol}.")
                from app.core.system_guard import system_sentinel
                system_sentinel.trigger_kill_switch(f"ML Anomaly detected on {symbol}")
                return self._neutral(symbol, "ANOMALY_DETECTED")
            
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(
                self.executor,
                self.inference_engine.predict_hybrid_uncertainty,
                self.models.get(symbol),
                list(self.sequence_buffers[symbol]),
                self.feature_names,
                symbol,
                self.seq_len_map.get(symbol, 32)
            )
            
            # [PHASE 13 - PHYSICS VETO & HUMILITY]
            # 1. Epistemic Gating (Rule 6: Epistemic Humility)
            if res["uncertainty"] > settings.CONVERGENCE_MAX_UNCERTAINTY:
                logger.warning(f"⚠️ HIGH UNCERTAINTY on {symbol} ({res['uncertainty']:.2f}). Trade Vetoed.")
                return self._neutral(symbol, "HIGH_UNCERTAINTY")

            # 2. Physics Validation (Rule 1: Physics Veto)
            if PHYSICS_AVAILABLE:
                regime = market_physics.ingest_tick({"close": features.get("close"), "tick_volume": features.get("volume")})
                # Check for Directional Divergence
                # (Neural says Long, but Logic/Structure says Down)
                is_long = res["directional_score"] > 0.6
                is_short = res["directional_score"] < 0.4
                
                # Simple Physics Veto Logic: If Energy is too low, or regime is toxic
                if regime and res["p_win"] > 0.6:
                    if not regime.is_accelerating and regime.velocity < 2.0:
                         logger.info(f"🛡️ PHYSICS VETO: Insufficient Momentum on {symbol}.")
                         return self._neutral(symbol, "PHYSICS_MOMENTUM_VETO")
            
            res["symbol"] = symbol
            return res
            
        except Exception as e:
            logger.error(f"Inference Error: {e}")
            return self._neutral(symbol, str(e))

    def _neutral(self, symbol, reason) -> Dict:
        return {
            "symbol": symbol, "buy": 0.0, "sell": 0.0, "hold": 1.0, "p_win": 0.5,
            "alpha_multiplier": 1.0, "volatility_regime": 0.5, "uncertainty": 1.0,
            "prediction": "WAIT", "why": reason, "execute_trade": False
        }
    
    async def reload_weights(self):
        """Clears the model cache to force a reload from disk on next inference."""
        logger.info("♻️ MLEngine: Reloading weights for all symbols...")
        # Clear model cache
        self.models.clear()
        self.sequence_buffers.clear()
        # Optionally perform explicit reload for already hydrated symbols
        logger.info("✅ MLEngine: Model weights flushed. Will reload on demand.")

    def get_status(self): 
        return {"v": "2.0", "symbols_registered": list(self.models.keys()), "anomaly_fitted": True}

    # Backward Compatibility & Sentinel Support
    def hydrate_hurst(self, s, p): self.hydrate(s, p, [])
    def hydrate_sequences(self, s, f): self.hydrate(s, [], f)
    async def ensemble_predict_async(self, s, c): return await self.predict_async(s, c)
    async def record_trade_result(self, t, p, s, c): self.replay.record_trade_result(t, p, s, c)
    async def check_loop_jitter(self):
         """Mock method for JitterSentinel health check compatibility."""
         pass
