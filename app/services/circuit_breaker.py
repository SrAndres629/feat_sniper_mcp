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
        # Tolerance: 15 seconds of silence = DEATH
        self.max_latency = 15.0 
        self.trade_manager: Optional[Any] = None 

    def set_trade_manager(self, manager):
        self.trade_manager = manager

    def heartbeat(self):
        """Llamado por cada tick ZMQ recibido."""
        self.last_tick_time = time.time()
        if self.is_tripped:
            # Auto-Reset if data flows again? 
            # Risk policy: Manual reset preferred. But for now auto-recover logic.
            # self.is_tripped = False
            pass

    async def monitor_heartbeat(self):
        """Tarea de fondo (Background Task)."""
        logger.info("🛡️ Circuit Breaker ARMED (Latency Limit: 15s)")
        try:
            while True:
                if not self.is_tripped:
                    latency = time.time() - self.last_tick_time
                    
                    if latency > self.max_latency:
                        # TRIGGER EMERGENCY
                        logger.critical(f"🚨 CIRCUIT TRIP: Dead Silence ({latency:.1f}s > {self.max_latency}s)")
                        await self.emergency_shutdown()
                        # Wait before checking again/resetting
                        await asyncio.sleep(10)
                        
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            logger.info("🛡️ Circuit Breaker Disarmed")

    async def emergency_shutdown(self):
        self.is_tripped = True
        logger.critical("⛔ SRE PROTOCOL: EXECUTE ORDER 66 (Close All)")
        if self.trade_manager:
            try:
                # Assuming TradeManager has close_all_positions
                await self.trade_manager.close_all_positions()
            except Exception as e:
                logger.error(f"SRE Execution Failed: {e}")
        else:
            logger.error("SRE Error: TradeManager not linked!")

# Singleton Global
circuit_breaker = CircuitBreaker()
