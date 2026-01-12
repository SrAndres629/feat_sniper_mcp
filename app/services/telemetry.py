import logging
import httpx
import asyncio
from datetime import datetime
from app.core.config import settings

logger = logging.getLogger("MT5_Bridge.Services.Telemetry")

class TelemetryEngine:
    """
    High-fidelity data pipeline to n8n for AI analysis.
    Dispatches market state and bot performance.
    """
    
    def __init__(self):
        self.webhook_url = settings.N8N_WEBHOOK_URL

    async def send_to_n8n(self, payload: dict):
        """Dispatches structured JSON to n8n webhook."""
        if not self.webhook_url:
            logger.debug("Telemetry: N8N_WEBHOOK_URL not set. Skipping.")
            return

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=payload,
                    timeout=5.0
                )
                if response.status_code == 200:
                    logger.info(f"Telemetry: Payload dispatched to n8n for asset {payload.get('asset')}")
                else:
                    logger.warning(f"Telemetry: n8n returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Telemetry: Failed to send data to n8n: {e}")

telemetry = TelemetryEngine()
