import asyncio
import logging
import time
from typing import Optional, Any

logger = logging.getLogger("feat.sre")

class CircuitBreaker:
    """
    Módulo 9: The Circuit Breaker.
    Monitores de seguridad SRE.
    """
    def __init__(self):
        self.last_tick_time = time.time()
        self.is_tripped = False
        self.current_level = 0  # 0: Safe, 1: Defensive, 2: Survival, 3: Shutdown
        # Tolerance: 15 seconds of silence = DEATH
        self.max_latency = 15.0 
        self.trade_manager: Optional[Any] = None 
        self._daily_opening_balance: Optional[float] = None

    def set_trade_manager(self, manager):
        self.trade_manager = manager

    async def get_drawdown(self) -> float:
        """Calcula el drawdown diario actual."""
        from app.core.config import settings
        from app.core.mt5_conn import mt5_conn
        import MetaTrader5 as mt5
        
        # Sincronizar balance inicial si es necesario
        if not self._daily_opening_balance:
            account = await mt5_conn.execute(mt5.account_info)
            if account:
                self._daily_opening_balance = account.balance
        
        if not self._daily_opening_balance:
            return 0.0
            
        account = await mt5_conn.execute(mt5.account_info)
        if not account:
            return 0.0
            
        real_loss = self._daily_opening_balance - account.equity
        return (real_loss / self._daily_opening_balance) * 100 if self._daily_opening_balance > 0 else 0.0

    async def get_lot_multiplier(self) -> float:
        """Retorna el multiplicador de lotaje basado en el nivel de riesgo."""
        from app.core.config import settings
        
        dd = await self.get_drawdown()
        
        if dd >= settings.CB_LEVEL_3_DD:
            self.current_level = 3
            if not self.is_tripped:
                await self.emergency_shutdown()
            return 0.0
        elif dd >= settings.CB_LEVEL_2_DD:
            self.current_level = 2
            return 0.25  # 75% Reduction
        elif dd >= settings.CB_LEVEL_1_DD:
            self.current_level = 1
            return 0.75  # 25% Reduction
        
        self.current_level = 0
        return 1.0

    def can_execute(self) -> bool:
        """Veredicto final para la ejecucin de órdenes."""
        return not self.is_tripped and self.current_level < 3

    def record_failure(self):
        """Registra un fallo técnico (broker error, timeout)."""
        # Futura implementación: Escalamiento si hay N fallos seguidos
        pass

    def record_success(self):
        """Limpia contadores de fallos temporales."""
        pass

    def heartbeat(self):
        """Llamado por cada tick ZMQ recibido."""
        self.last_tick_time = time.time()

    async def monitor_heartbeat(self):
        """Tarea de fondo: Monitorea latencia y resets diarios."""
        logger.info(f"🛡️ Multi-Level Circuit Breaker ARMED (POM Protocol Active)")
        try:
            while True:
                # 1. Monitoreo de Latencia (Heartbeat)
                latency = time.time() - self.last_tick_time
                if latency > self.max_latency and not self.is_tripped:
                    logger.critical(f"🚨 CIRCUIT TRIP: Dead Silence ({latency:.1f}s > {self.max_latency}s)")
                    await self.emergency_shutdown()
                
                # 2. Monitoreo de Drawdown (Status Check)
                multiplier = await self.get_lot_multiplier()
                if self.current_level > 0:
                     logger.warning(f"⚠️ CB STATUS: Level {self.current_level} (Multiplier: {multiplier})")

                await asyncio.sleep(5.0)  # Check every 5s
        except asyncio.CancelledError:
            logger.info("🛡️ Circuit Breaker Disarmed")

    async def emergency_shutdown(self):
        self.is_tripped = True
        self.current_level = 3
        logger.critical("⛔ SRE PROTOCOL: EXECUTE ORDER 66 (Emergency Shutdown)")
        if self.trade_manager:
            try:
                # Cerramos todo
                await self.trade_manager.close_all_positions()
            except Exception as e:
                logger.error(f"Emergency Execution Failed: {e}")
        else:
            logger.error("SRE Error: TradeManager not linked! Cannot close positions automatically.")

# Singleton Global
circuit_breaker = CircuitBreaker()
