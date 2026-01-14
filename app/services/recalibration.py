import logging
from typing import Dict, Any
from app.services.rag_memory import rag_memory
from datetime import datetime, timedelta

logger = logging.getLogger("feat.ml.recalibration")

TURBULENT_REGIME = "Turbulento"

class RecalibrationService:
    """
    Módulo 7: Bucle de Retroalimentación (RAG).
    Ajusta la confianza de la IA basado en el rendimiento reciente (últimos 10 trades)
    e implementa vetos de seguridad por régimen de mercado.
    """
    def __init__(self, win_rate_threshold: float = 0.40, penalty_factor: float = 0.8, n_trades: int = 10, veto_duration_hours: int = 2):
        self.win_rate_threshold = win_rate_threshold
        self.penalty_factor = penalty_factor
        self.n_trades = n_trades
        self.veto_duration = timedelta(hours=veto_duration_hours)
        self.vetoes = {}  # Almacena: {"regime": expiry_timestamp}

    def is_vetoed(self, regime: str) -> bool:
        """Verifica si un régimen de mercado está actualmente vetado."""
        if regime in self.vetoes:
            if datetime.now() < self.vetoes[regime]:
                return True
            else:
                # El veto ha expirado
                del self.vetoes[regime]
        return False

    def apply_veto(self, regime: str):
        """Aplica un veto de seguridad a un régimen de mercado."""
        expiry = datetime.now() + self.veto_duration
        self.vetoes[regime] = expiry
        logger.warning(f"🛡️  SECURITY VETO applied to '{regime}' regime for {self.veto_duration.seconds / 3600} hours.")

    async def get_confidence_multiplier(self, current_regime: str = "UNKNOWN") -> Dict[str, Any]:
        """
        Calcula el multiplicador de confianza y gestiona los vetos de seguridad.
        """
        # 1. Chequeo de Veto de Seguridad
        if self.is_vetoed(current_regime):
            return {
                "multiplier": 0.0,
                "drawdown_mode": True,
                "reason": f"Active Security Veto for '{current_regime}'",
                "stats": {"veto_expiry": self.vetoes.get(current_regime)}
            }
            
        try:
            # 2. Análisis de Rendimiento Reciente (Últimos N Trades)
            last_trades = await rag_memory.get_last_n_trades(self.n_trades)
            if not last_trades:
                return {"multiplier": 1.0, "drawdown_mode": False, "reason": "No recent trades", "stats": {}}

            wins = sum(1 for trade in last_trades if trade.get("outcome") == "WIN")
            total = len(last_trades)
            win_rate = wins / total if total > 0 else 0.0

            # 3. Detección de Racha de Pérdidas en Régimen Turbulento
            if current_regime == TURBULENT_REGIME and len(last_trades) >= 2:
                last_two_trades = last_trades[:2]
                if all(t.get("regime") == TURBULENT_REGIME and t.get("outcome") == "LOSS" for t in last_two_trades):
                    self.apply_veto(TURBULENT_REGIME)
                    return {
                        "multiplier": 0.0,
                        "drawdown_mode": True,
                        "reason": f"Veto triggered: 2 consecutive losses in {TURBULENT_REGIME} regime.",
                        "stats": {"last_trades": last_trades}
                    }

            # 4. Penalización por Bajo Win-Rate General
            if total >= self.n_trades / 2 and win_rate < self.win_rate_threshold:
                return {
                    "multiplier": self.penalty_factor,
                    "drawdown_mode": True,
                    "reason": f"Low Win-Rate ({win_rate:.2f}) in last {total} trades.",
                    "stats": {"win_rate": win_rate, "total_trades": total}
                }

            return {
                "multiplier": 1.0,
                "drawdown_mode": False,
                "reason": "Performance OK",
                "stats": {"win_rate": win_rate, "total_trades": total}
            }

        except Exception as e:
            logger.error(f"Recalibration Error: {e}")
            return {"multiplier": 1.0, "drawdown_mode": False, "reason": "Error", "stats": {}}

# Instancia Global
recalibration_service = RecalibrationService()
