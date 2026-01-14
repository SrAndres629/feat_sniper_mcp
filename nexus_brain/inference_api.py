import logging
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from nexus_brain.hybrid_model import HybridModel
from app.services.rag_memory import rag_memory

logger = logging.getLogger("feat.brain.inference")

class NeuralInferenceAPI:
    """
    Módulo 4: The Neural Link.
    Puente de inferencia entre MCP Server y HybridModel (PyTorch).
    """
    def __init__(self, model_path: str = "models/lstm_XAUUSD_v2.pt"):
        self.brain = HybridModel(model_path)
        logger.info("[BRAIN] Neural Link Initialized")

    async def predict_next_candle(self, market_data: Dict[str, Any], physics_regime: Any = None) -> Dict[str, Any]:
        """
        Genera predicción de probabilidad para la siguiente vela.
        TEMPORAL CONTEXT: Incluye hora, sesión y rendimiento histórico (RAG).
        """
        if not self.brain or not self.brain.net:
            return {"error": "Brain Offline", "p_win": 0.0}

        try:
            # 0. Temporal Context Retrieval
            now = datetime.now()
            hour = now.hour
            
            # Determine Session (0: Asia, 1: London, 2: NY)
            # Simplified: Asia (0-7), London (8-13), NY (14-23) UTC/Local approximation
            if 0 <= hour <= 7: session = 0.0
            elif 8 <= hour <= 13: session = 1.0
            else: session = 2.0
            
            # Historical Benchmark (RAG)
            performance = await rag_memory.get_hourly_performance(hour)
            hist_winrate = performance.get("win_rate", 0.5)
            sample_count = min(performance.get("sample_size", 0) / 100, 1.0) # Normalized to [0,1]

            # 1. Feature Extraction (Basic V1.1 - Temporal Aware & Robust)
            def safe_float(val, default=0.0):
                try:
                    if val is None: return default
                    f_val = float(val)
                    return f_val if not np.isnan(f_val) and not np.isinf(f_val) else default
                except: return default

            price = safe_float(market_data.get('bid') or market_data.get('close'))
            volume = safe_float(market_data.get('tick_volume') or market_data.get('volume'))
            spread = 0.0
            if 'ask' in market_data and 'bid' in market_data:
                spread = safe_float(market_data['ask']) - safe_float(market_data['bid'])

            # Construct Feature Vector (10 dims)
            # [Accel, VolZ, Price, Volume, Hour, Session, Hist_Winrate, Sample, RSI, Spread]
            features = [
                safe_float(physics_regime.acceleration_score if physics_regime else 0.0),
                safe_float(physics_regime.vol_z_score if physics_regime else 0.0),
                price,
                volume,
                float(hour) / 24.0, # Normalizada
                session / 2.0,      # Normalizada
                safe_float(hist_winrate),
                safe_float(sample_count),
                0.5,                # RSI Placeholder
                spread
            ]
            
            context = {
                "features": features,
                "raw_data": market_data,
                "hour": hour,
                "session": session,
                "hist_winrate": hist_winrate
            }
            
            # 2. Inferencia
            result = self.brain.predict(context)
            
            # 3. Decision Refinement (Senior Bias: Small Profit > Any Loss)
            # Si el winrate historico de esta hora es < 45%, ser mas conservador.
            if hist_winrate < 0.45:
                result["p_win"] *= 0.8
                result["alpha_confidence"] *= 0.8
                if result["p_win"] < 0.6: result["execute_trade"] = False

            return result

        except Exception as e:
            logger.error(f"Inference Failure: {e}")
            return {"error": str(e), "p_win": 0.0}

# Singleton Global
neural_api = NeuralInferenceAPI()
