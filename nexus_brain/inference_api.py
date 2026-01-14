import logging
import torch
import numpy as np
import time
import pandas as pd
from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional
from nexus_brain.hybrid_model import HybridModel
from app.services.rag_memory import rag_memory
from app.skills.indicators import calculate_feat_layers, detect_divergence
from app.core.config import settings
from app.ml.drift_monitor import WeightDriftMonitor
from app.services.recalibration import recalibration_service

logger = logging.getLogger("feat.brain.inference")

class NeuralInferenceAPI:
    """
    Módulo 4: The Neural Link.
    Puente de inferencia entre MCP Server y HybridModel (PyTorch).
    """
    def __init__(self, model_path: str = "models/lstm_XAUUSD_v2.pt"):
        self.brain = HybridModel(model_path)
        self.max_history = settings.LAYER_BIAS_PERIOD + 200
        self.history_np = np.zeros((self.max_history, 3)) # close, high, low
        self.history_len = 0
        
        # Phase 12: Stateful SMMA Cache for O(1) Updates
        # We track state for all 21 micro/operative/bias layers
        self.smma_states = {
            "micro": np.zeros(len(settings.LAYER_MICRO_PERIODS)),
            "operative": np.zeros(len(settings.LAYER_OPERATIVE_PERIODS)),
            "bias": 0.0,
            "bias_prev": 0.0,
            # Module 5: Volume Z-Score State
            "vol_mean": 0.0,
            "vol_std": 1.0,
            "vol_ema_alpha": 0.01  # EMA for rolling stats
        }
        self.initialized = False
        
        # Weight Drift Monitor Initialization
        self.drift_monitor = WeightDriftMonitor(
            training_stats=self.brain.scaler_stats if self.brain else None,
            window_size=200,
            drift_threshold=0.8
        )
        
        logger.info(f"[BRAIN] Neural Link Initialized (5D Vector: L1, L1W, L4S, Div, Vol_Z)")

    async def predict_next_candle(self, market_data: Dict[str, Any], physics_regime: Any = None) -> Dict[str, Any]:
        """
        Genera predicción de probabilidad para la siguiente vela.
        """
        if not self.brain or not self.brain.net:
            return {"error": "Brain Offline", "p_win": 0.0}

        try:
            # 0. Context
            price = float(market_data.get('bid') or market_data.get('close') or 0.0)
            
            # 1. Update Persistent Buffer
            self.history_np = np.roll(self.history_np, -1, axis=0)
            self.history_np[-1] = [price, float(market_data.get('high') or 0.0), float(market_data.get('low') or 0.0)]
            self.history_len = min(self.history_len + 1, self.max_history)
            
            # Warmup Check
            if self.history_len < settings.LAYER_BIAS_PERIOD:
                return {"status": "Warming up", "p_win": 0.5}
            
            # 2. O(1) SMMA Update Logic (Institutional Grade)
            if not self.initialized:
                # Cold Start: Initialize states with full window calculation once
                df_init = pd.DataFrame(self.history_np[-(settings.LAYER_BIAS_PERIOD):], columns=['close', 'high', 'low'])
                for i, p in enumerate(settings.LAYER_MICRO_PERIODS):
                    self.smma_states["micro"][i] = df_init['close'].ewm(alpha=1.0/p, adjust=False).mean().iloc[-1]
                for i, p in enumerate(settings.LAYER_OPERATIVE_PERIODS):
                    self.smma_states["operative"][i] = df_init['close'].ewm(alpha=1.0/p, adjust=False).mean().iloc[-1]
                self.smma_states["bias"] = df_init['close'].ewm(alpha=1.0/settings.LAYER_BIAS_PERIOD, adjust=False).mean().iloc[-1]
                self.smma_states["bias_prev"] = df_init['close'].ewm(alpha=1.0/settings.LAYER_BIAS_PERIOD, adjust=False).mean().iloc[-4]
                self.initialized = True
            else:
                # Hot Update: Recursive formula S[t] = (1-a)*S[t-1] + a*X[t]
                for i, p in enumerate(settings.LAYER_MICRO_PERIODS):
                    a = 1.0 / p
                    self.smma_states["micro"][i] = (1-a) * self.smma_states["micro"][i] + a * price
                for i, p in enumerate(settings.LAYER_OPERATIVE_PERIODS):
                    a = 1.0 / p
                    self.smma_states["operative"][i] = (1-a) * self.smma_states["operative"][i] + a * price
                
                a_bias = 1.0 / settings.LAYER_BIAS_PERIOD
                self.smma_states["bias_prev"] = self.smma_states["bias"] # Approximated slope
                self.smma_states["bias"] = (1-a_bias) * self.smma_states["bias"] + a_bias * price

            # 3. Assemble Feature Vector (Zero Overhead)
            l1_mean = np.mean(self.smma_states["micro"])
            l1_width = np.std(self.smma_states["micro"])
            l2_mean = np.mean(self.smma_states["operative"])
            l4_slope = (self.smma_states["bias"] / self.smma_states["bias_prev"] - 1) * 100 if self.smma_states["bias_prev"] != 0 else 0.0
            
            # Module 5: Volume Z-Score (5th Dimension)
            volume = float(market_data.get('volume') or market_data.get('tick_volume') or 0)
            if volume > 0:
                # Update rolling mean/std with EMA
                alpha = self.smma_states["vol_ema_alpha"]
                self.smma_states["vol_mean"] = (1-alpha) * self.smma_states["vol_mean"] + alpha * volume
                vol_diff = abs(volume - self.smma_states["vol_mean"])
                self.smma_states["vol_std"] = (1-alpha) * self.smma_states["vol_std"] + alpha * vol_diff
                
                vol_zscore = (volume - self.smma_states["vol_mean"]) / max(self.smma_states["vol_std"], 1e-6)
            else:
                vol_zscore = 0.0
            
            features = [l1_mean, l1_width, l4_slope, l1_mean / l2_mean if l2_mean != 0 else 1.0, vol_zscore]

            # Update Drift Monitor
            drift_metrics = self.drift_monitor.update(features)

            # PvP Alpha Check (Needs recent history of price/slope, simplified for O(1))
            # [Note: Divergence detection usually needs a window, we can use a small circular buffer for scores]
            pvp_regime = "NEUTRAL" 

            context = {
                "features": features,
                "raw_data": market_data,
                "hour": datetime.now().hour,
                "pvp_divergence": pvp_regime,
                "drift_metrics": drift_metrics
            }
            
            # Inferencia
            start_inf = time.time()
            result = self.brain.predict(context)
            latency_ms = (time.time() - start_inf) * 1000
            
            # Module 7: RAG Recalibration (Feedback Loop)
            recal = await recalibration_service.get_confidence_multiplier()
            p_win_raw = result.get("p_win", 0.0)
            p_win_adj = p_win_raw * recal["multiplier"]
            result["p_win_raw"] = p_win_raw
            result["p_win"] = round(p_win_adj, 4)
            result["recalibration"] = recal
            
            if recal["drawdown_mode"]:
                logger.info(f"📉 [RECAL] Confidence penalized ({recal['multiplier']}x): {recal['reason']}")

            # Module 6: Consenso de Desconfianza (Alpha Refinement)

            # Physics vs Neural Veto
            p_win = result.get("p_win", 0.0)
            execute_trade = result.get("execute_trade", False)
            
            if execute_trade:
                # Si queremos COMPRAR (p_win > threshold) pero la pendiente macro es negativa (Bajista)
                if p_win > 0.55 and l4_slope < -0.01:
                    result["execute_trade"] = False
                    result["veto_reason"] = "Physics Conflict: Neural BUY vs Macro BEARISH Slope"
                    logger.info(f"🚫 VETO [Consenso]: Neural BUY contradicted by L4 Slope ({l4_slope:.4f})")
                
                # Si queremos VENDER (p_win < threshold_sell) pero la pendiente macro es positiva (Alcista)
                # Nota: El modelo actual parece ser binario para p_win, asumimos p_win alto = compra.
                # Si el modelo soporta señales de venta (p_win < 0.45 por ejemplo), aplicaríamos el veto inverso.
            
            # Auditoria de Latencia (Fase 12 Target: < 5ms)
            if latency_ms > 5:
                logger.warning(f"⚠️ LATENCY ALERT: {latency_ms:.2f}ms")
            
            result["latency_ms"] = latency_ms
            return result


        except Exception as e:
            logger.error(f"Inference Failure: {e}")
            return {"error": str(e), "p_win": 0.0}

# Singleton Global
neural_api = NeuralInferenceAPI()