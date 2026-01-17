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
        while self.running:
            try:
                if self.ml_engine:
                    await self.ml_engine.check_loop_jitter()
                
                # Check for "Dead Silence" (MT5 disconnection or loop hang)
                if time.time() - self.last_heartbeat > 30.0:
                    logger.warning("🕒 WARNING: 30s Jitter Detected. Potential Radio Silence.")
                
            except Exception as e:
                logger.error(f"Jitter Sentinel Error: {e}")
            
            await asyncio.sleep(0.1)

    def heartbeat(self):
        self.last_heartbeat = time.time()

    def stop(self):
        self.running = False
