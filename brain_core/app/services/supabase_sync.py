import logging
from typing import Dict, Any, Optional
from supabase import create_client, Client
from app.core.config import settings

logger = logging.getLogger("MT5_Bridge.SupabaseSync")

class SupabaseSync:
    """
    Gestor de sincronizacin con Supabase para el ecosistema NEXUS.
    Centraliza el logging de seales y trades en la nube.
    """
    _client: Optional[Client] = None

    def __init__(self):
        self._initialize_client()

    def _initialize_client(self):
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            logger.warning("Supabase no configurado. El logging en la nube estar desactivado.")
            return

        try:
            self._client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            logger.info(" Cliente Supabase NEXUS inicializado.")
        except Exception as e:
            logger.error(f"Error inicializando Supabase: {e}")

    async def log_signal(self, data: Dict[str, Any]):
        """Registra una seal del Sniper en Supabase."""
        if not self._client:
            return

        try:
            # Adaptar datos al esquema de la tabla sniper_signals
            payload = {
                "symbol": data.get("symbol", "UNKNOWN"),
                "timeframe": data.get("timeframe", "UNKNOWN"),
                "action": data.get("action", "WAIT"),
                "price": data.get("price", 0),
                "confidence": data.get("confidence", 0),
                "engineer_diagnosis": data.get("reason", ""),
                "metadata": data
            }
            
            # Operacin sncrona enviada a hilo para no bloquear el loop
            import anyio
            await anyio.to_thread.run_sync(
                lambda: self._client.table("sniper_signals").insert(payload).execute()
            )
            logger.info(f" Seal sincronizada con Supabase: {payload['symbol']} {payload['action']}")
        except Exception as e:
            logger.error(f"Error sincronizando seal con Supabase: {e}")

    async def update_performance(self, model_id: str, metrics: Dict[str, Any]):
        """Actualiza mtricas de performance del modelo ML en la nube."""
        if not self._client:
            return
            
        try:
            import anyio
            await anyio.to_thread.run_sync(
                lambda: self._client.table("model_performance").insert({
                    "model_id": model_id,
                    "symbol": metrics.get("symbol", "ALL"),
                    "win_rate": metrics.get("win_rate", 0),
                    "profit_factor": metrics.get("profit_factor", 0),
                    "sharpe_ratio": metrics.get("sharpe", 0),
                    "hyperparameters": metrics
                }).execute()
            )
        except Exception as e:
            logger.error(f"Error actualizando performance en Supabase: {e}")

# Instancia global
supabase_sync = SupabaseSync()
