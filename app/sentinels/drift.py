import asyncio
import logging
from app.core.config import settings

logger = logging.getLogger("feat.sentinel.drift")

class DriftSentinel:
    """
    Monitors model performance drift and triggers AutoML evolution.
    """
    def __init__(self, automl_orchestrator=None):
        self.automl_orchestrator = automl_orchestrator
        self.running = True

    async def run_loop(self):
        if not self.automl_orchestrator or not settings.AUTOML_ENABLED:
            logger.info("🛡️ DRIFT SENTINEL: Standby (AutoML Disabled)")
            return
            
        logger.info(f"🛡️ DRIFT SENTINEL: Active (Cycle: {settings.AUTOML_CHECK_INTERVAL_MINUTES}m)")
        while self.running:
            try:
                await self.automl_orchestrator.check_and_evolve()
            except Exception as e:
                logger.error(f"Drift Sentinel Error: {e}")
            
            await asyncio.sleep(settings.AUTOML_CHECK_INTERVAL_MINUTES * 60)

    def stop(self):
        self.running = False
