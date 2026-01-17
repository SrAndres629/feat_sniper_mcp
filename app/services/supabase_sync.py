import logging
from typing import Dict, Any, Optional
from supabase import create_client, Client
from app.core.config import settings

from app.core.observability import resilient

logger = logging.getLogger("MT5_Bridge.SupabaseSync")

class SupabaseSync:
    """Master Cloud Persistence Manager for the NEXUS Ecosystem.
    
    Orchestrates signal logging, institutional tick auditing, and performance
    tracking with built-in circuit breakers for cloud-latency resilience.
    
    Attributes:
        _client: Supabase client instance.
    """
    _client: Optional[Client] = None

    @property
    def client(self) -> Optional[Client]:
        """Provides access to the raw Supabase client."""
        return self._client

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

    @resilient(max_retries=2, failure_threshold=5, recovery_timeout=60)
    async def log_signal(self, data: Dict[str, Any]) -> None:
        """Synchronizes a Sniper signal to the cloud.
        
        Args:
            data: Signal raw data and engineered reason.
        """
        if not self._client:
            return

        # Institutional Normalization: 0-100 to 0.0-1.0
        conf = data.get("confidence", 0)
        if conf > 1: conf = conf / 100.0
        
        p_win = data.get("p_win", 0.5)
        if p_win > 1: p_win = p_win / 100.0

        payload = {
            "symbol": data.get("symbol", "UNKNOWN"),
            "timeframe": data.get("timeframe", "UNKNOWN"),
            "direction": data.get("action", "WAIT"),
            "entry_price": data.get("price", 0),
            "confidence": float(conf),
            "explanation": data.get("reason", ""),
            "ml_source": data.get("source", "GBM+LSTM"),
            "p_win": float(p_win),
            "top_drivers": data.get("top_drivers", []),
            "execution_enabled": data.get("execution_enabled", False)
        }
        
        import anyio
        await anyio.to_thread.run_sync(
            lambda: self._client.table("feat_signals").insert(payload).execute()
        )
        logger.info(f" Signal synced with Supabase: {payload['symbol']} {payload['action']}")

    @resilient(max_retries=1, failure_threshold=10)
    async def log_tick(self, data: Dict[str, Any]) -> None:
        """Asynchronously logs an institutional market tick.
        
        Optimized for high frequency with loose circuit breaker thresholds.
        """
        if not self._client:
            return

        payload = {
            "symbol": data.get("symbol", "UNKNOWN"),
            "bid": float(data.get("bid", data.get("close", 0))),
            "ask": float(data.get("ask", data.get("close", 0))),
            "volume": float(data.get("volume", 0)),
            "tick_time": data.get("tick_time") or data.get("time")
        }
        
        payload = {k: v for k, v in payload.items() if v is not None}
        
        import anyio
        await anyio.to_thread.run_sync(
            lambda: self._client.table("market_ticks").insert(payload).execute()
        )

    @resilient(max_retries=3)
    async def update_performance(self, model_id: str, metrics: Dict[str, Any]) -> None:
        """Updates ML model performance track-record."""
        if not self._client:
            return
            
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

    @resilient(max_retries=2)
    async def log_activity(self, event: str, level: str = "INFO", details: Optional[Dict] = None) -> None:
        """Logs high-level bot activity events.
        
        Args:
            event: Event description.
            level: INFO, WARNING, ERROR, CRITICAL.
            details: Structured JSON metadata.
        """
        if not self._client:
            return
            
        payload = {
            "event": event,
            "level": level,
            "details": details or {},
            "phase": "AUTONOMOUS_EVOLUTION_V5"
        }
        
        import anyio
        await anyio.to_thread.run_sync(
            lambda: self._client.table("bot_activity_log").insert(payload).execute()
        )

    @resilient(max_retries=1, failure_threshold=20)
    async def log_neural_state(self, state: Dict[str, Any]) -> None:
        """
        [LEVEL 46] High-Frequency sync of Neural Cortex State for Real-time Dashboard.
        Pushes Probabilities + PVP Context + Immune Status.
        """
        if not self._client:
            return

        # Flatten the nested state for SQL table
        # state = neural_service.get_latest_state()
        probs = state.get("predictions", {})
        pvp = state.get("pvp_context", {})
        immune = state.get("immune_system", {})
        
        payload = {
            "symbol": state.get("symbol", "WAIT"),
            "price": state.get("price", 0.0),
            "alpha_confidence": float(probs.get("buy", 0.0) if probs.get("buy") > 0.5 else probs.get("sell", 0.0)),
            "acceleration": 0.0, # Placeholder or need to pass explicitly
            "hurst": 0.5, # Placeholder
            
            # PVP
            "poc_price": float(pvp.get("poc", 0.0)),
            "vah_price": float(pvp.get("vah", 0.0)),
            "val_price": float(pvp.get("val", 0.0)),
            "energy_score": float(pvp.get("energy", 0.0)),
            
            "metadata": {
                "probs": probs,
                "immune": immune,
                "kinetic": state.get("kinetic_context", {}), # [LEVEL 49]
                "uncertainty": state.get("uncertainty", 0.0)
            }
        }
        
        import anyio
        await anyio.to_thread.run_sync(
            lambda: self._client.table("neural_signals").insert(payload).execute()
        )

# Instancia global
supabase_sync = SupabaseSync()
