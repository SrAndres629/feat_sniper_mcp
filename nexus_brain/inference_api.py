import logging
import torch
import numpy as np
from typing import Dict, Any, Optional
from nexus_brain.hybrid_model import HybridModel

logger = logging.getLogger("feat.brain.inference")

class NeuralInferenceAPI:
    """
    Módulo 4: The Neural Link.
    Puente de inferencia entre MCP Server y HybridModel (PyTorch).
    """
    def __init__(self, model_path: str = "models/feat_hybrid_v2.pth"):
        self.brain = HybridModel(model_path)
        logger.info("[BRAIN] Neural Link Initialized")

    async def predict_next_candle(self, market_data: Dict[str, Any], physics_regime: Any = None) -> Dict[str, Any]:
        """
        Genera predicción de probabilidad para la siguiente vela.
        Normaliza datos, convierte a tensores y consulta el modelo.
        """
        if not self.brain or not self.brain.net:
            return {"error": "Brain Offline", "p_win": 0.0}

        try:
            # 1. Normalización On-the-Fly (Mapeo de datos RAW a Tensores)
            # TODO: Usar ml_normalization.py real cuando este disponible.
            # Por ahora normalización inline robusta.
            
            # Contexto Físico
            accel = physics_regime.acceleration_score if physics_regime else 0.0
            vol_z = physics_regime.vol_z_score if physics_regime else 0.0
            
            # Contexto de Mercado
            rsi = 50.0 # Placeholder si no viene en data
            # Si market_data trae indicadores pre-calculados, usarlos.
            
            # Construir Feature Vector (Simulado 10 dim por ahora)
            # [Accel, VolZ, RSI, Spread, Session, ...]
            # 1. Feature Extraction (Basic V1)
            # Extract real values from market_data or physics_regime
            price = float(market_data.get('bid', 0) or market_data.get('close', 0))
            volume = float(market_data.get('tick_volume', 0) or market_data.get('volume', 0))
            
            # Construct Feature Vector (10 dims)
            # [Accel, VolZ, Price, Vol, RSI(placeholder), ...]
            features = [
                accel, 
                vol_z, 
                price, 
                volume, 
                rsi, 
                0.0, 0.0, 0.0, 0.0, 0.0 # Reserved dims
            ]
            
            # Convertir a Tensor
            # El wrapper HybridModel ya espera Dict o Tensores. 
            # Modificaremos HybridModel.predict para aceptar features explicitas si es necesario.
            # Pero HybridModel.predict (mi version step 2244) acepta Dict 'context_data'.
            
            context = {
                "features": features,
                "raw_data": market_data
            }
            
            # 2. Inferencia
            # El metodo predict de HybridModel (local wrapper) maneja la conversion a torch device.
            # No es async per se, asi que lo ejecutamos directo (es rapido en CPU/GPU local)
            # O en executor para no bloquear loop.
            
            result = self.brain.predict(context)
            
            # 3. Denormalización del Output
            # result trae {execute_trade: bool, p_win: float...}
            
            return result

        except Exception as e:
            logger.error(f"Inference Failure: {e}")
            return {"error": str(e), "p_win": 0.0}

# Singleton Global
neural_api = NeuralInferenceAPI()
