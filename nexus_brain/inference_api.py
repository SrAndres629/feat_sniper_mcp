No puedo modificar los archivos directamente porque las herramientas de escritura (`write_file` o `run_shell_command`) no están disponibles en este entorno. Sin embargo, he refactorizado el código de `nexus_brain/inference_api.py` según tus instrucciones.

Aquí tienes el código completo refactorizado. Por favor, reemplaza el contenido de `nexus_brain/inference_api.py` con lo siguiente:

### Cambios realizados:
1.  **Buffer de Historia**: Se añadió `self.history = deque(...)` para acumular ticks y permitir el cálculo de indicadores (SMMA requiere historia).
2.  **Cálculo FEAT**: Se importó y utilizó `calculate_feat_layers` para procesar el DataFrame generado desde el buffer.
3.  **Vector de Características**: Se extrajeron exactamente las 4 características solicitadas: `[L1_Mean, L1_Width, L4_Slope, Div_L1_L2]`.
4.  **Limpieza**: Se eliminó la lógica de construcción manual del vector de 10 dimensiones y la dependencia lógica de `physics_regime`.

import logging
import torch
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional
from nexus_brain.hybrid_model import HybridModel
from app.services.rag_memory import rag_memory
from app.skills.indicators import calculate_feat_layers
from app.core.config import settings

logger = logging.getLogger("feat.brain.inference")

class NeuralInferenceAPI:
    """
    Módulo 4: The Neural Link.
    Puente de inferencia entre MCP Server y HybridModel (PyTorch).
    """
    def __init__(self, model_path: str = "models/lstm_XAUUSD_v2.pt"):
        self.brain = HybridModel(model_path)
        # Buffer circular para datos históricos necesarios para indicadores FEAT
        # Necesitamos al menos settings.LAYER_BIAS_PERIOD (2048) + margen
        self.history = deque(maxlen=settings.LAYER_BIAS_PERIOD + 100)
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

            # 1. Feature Extraction (FEAT Layers)
            def safe_float(val, default=0.0):
                try:
                    if val is None: return default
                    f_val = float(val)
                    return f_val if not np.isnan(f_val) and not np.isinf(f_val) else default
                except: return default

            price = safe_float(market_data.get('bid') or market_data.get('close'))
            
            # Update History Buffer
            self.history.append({'close': price})
            
            # Convert buffer to DataFrame for calculation
            df = pd.DataFrame(self.history)
            
            # Calculate FEAT Layers
            feat_df = calculate_feat_layers(df)
            
            # Extract exactly 4 features: [L1_Mean, L1_Width, L4_Slope, Div_L1_L2]
            if not feat_df.empty:
                latest = feat_df.iloc[-1]
                features = [
                    safe_float(latest.get('L1_Mean')),
                    safe_float(latest.get('L1_Width')),
                    safe_float(latest.get('L4_Slope')),
                    safe_float(latest.get('Div_L1_L2'))
                ]
            else:
                # Fallback vectors if insufficient history
                # Div_L1_L2 ratio default 1.0 (neutral), others 0.0
                features = [0.0, 0.0, 0.0, 1.0]

            # Construct Context for HybridModel
            # Result passed as vector (1, 4) implicit via list
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
                if "alpha_confidence" in result:
                    result["alpha_confidence"] *= 0.8
                if result["p_win"] < 0.6: result["execute_trade"] = False

            return result

        except Exception as e:
            logger.error(f"Inference Failure: {e}")
            return {"error": str(e), "p_win": 0.0}

# Singleton Global
neural_api = NeuralInferenceAPI()