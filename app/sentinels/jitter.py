import asyncio
import logging
import time
from app.core.config import settings

logger = logging.getLogger("feat.sentinel.jitter")

class JitterSentinel:
    """
    Monitors the health of the ZMQ processing loop and MT5 connectivity.
    """
    def __init__(self, ml_engine=None):
        self.ml_engine = ml_engine
        self.last_heartbeat = time.time()
        self.running = True

    async def run_loop(self):
        logger.info("🛡️ JITTER SENTINEL: Online (Sub-millisecond monitoring)")
        from app.core.mt5_conn import mt5_conn
        
        while self.running:
            try:
                if self.ml_engine:
                    await self.ml_engine.check_loop_jitter()
                
                # Check for "Dead Silence" (MT5 disconnection or loop hang)
                now = time.time()
                since_heartbeat = now - self.last_heartbeat
                
                if since_heartbeat > 30.0:
                    # Check if actually disconnected from MT5 terminal
                    is_connected = mt5_conn.connected
                    
                    if not hasattr(self, '_last_warn') or now - self._last_warn > 60.0:
                        import psutil
                        terminal_process_running = any("terminal64.exe" in p.name().lower() for p in psutil.process_iter(['name']))
                        
                        if not terminal_process_running:
                            logger.error("🛑 CRITICAL: MetaTrader 5 process (terminal64.exe) is NOT running.")
                        elif not is_connected:
                            logger.error("🛑 CRITICAL: MT5 Terminal process is running but session is DISCONNECTED.")
                        else:
                            # If connected but no ticks, might be market closed or idle
                            logger.warning(f"🕒 IDLE: {since_heartbeat:.0f}s since last tick. (Market closed or low liquidity)")
                        self._last_warn = now
                
            except Exception as e:
                logger.error(f"Jitter Sentinel Error: {e}")
            
            await asyncio.sleep(1.0) # Reduced check frequency to save CPU

    def heartbeat(self):
        self.last_heartbeat = time.time()

    def stop(self):
        self.running = False
