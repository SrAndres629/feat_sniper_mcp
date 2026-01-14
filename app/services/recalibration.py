import logging
from typing import Dict, Any
from app.services.rag_memory import rag_memory
from datetime import datetime

logger = logging.getLogger("feat.ml.recalibration")

class RecalibrationService:
    """
    Hueco #1: Bucle de Retroalimentación RAG.
    Ajusta la confianza de la IA basado en el rendimiento reciente de la sesión.
    """
    def __init__(self, win_rate_threshold: float = 0.45, penalty_factor: float = 0.8):
        self.win_rate_threshold = win_rate_threshold
        self.penalty_factor = penalty_factor
        self.session_drawdown_mode = False

    async def get_confidence_multiplier(self) -> Dict[str, Any]:
        """
        Calcula el multiplicador de confianza basado en los últimos trades.
        """
        try:
            current_hour = datetime.now().hour
            perf = await rag_memory.get_hourly_performance(current_hour)
            
            win_rate = perf.get("win_rate", 1.0)
            sample_size = perf.get("sample_size", 0)
            
            # Solo penalizamos si hay suficiente muestra (ej. > 2 trades en esa hora históricamente)
            if sample_size >= 2 and win_rate < self.win_rate_threshold:
                self.session_drawdown_mode = True
                multiplier = self.penalty_factor
                reason = f"Low Hourly WinRate: {win_rate:.2f} ({sample_size} trades)"
            else:
                self.session_drawdown_mode = False
                multiplier = 1.0
                reason = "Performance OK"

            return {
                "multiplier": multiplier,
                "drawdown_mode": self.session_drawdown_mode,
                "reason": reason,
                "stats": perf
            }
            
        except Exception as e:
            logger.error(f"Recalibration Error: {e}")
            return {"multiplier": 1.0, "drawdown_mode": False, "reason": "Error", "stats": {}}

# Instancia Global
recalibration_service = RecalibrationService()
