
# =============================================================================
# ZMQ PROJECTOR: Python-to-MT5 HUD Bridge
# =============================================================================
# Broadcasts system state, pilot status, and neural metrics to MT5 visualizer.
# Uses ZMQ_PUB to avoid blocking.

import zmq
import json
import logging
import asyncio
from typing import Dict, Any, Optional

logger = logging.getLogger("MT5_Bridge.Projector")

class ZMQProjector:
    """
    Projector sends 'HUD' updates to MT5 using ZMQ Publish pattern.
    MT5 acts as a Subscriber to this feed.
    """
    def __init__(self, host: str = "0.0.0.0", port: int = 5556):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.addr = f"tcp://{host}:{port}"
        self._is_bound = False

    def start(self):
        """Bind the PUB socket."""
        try:
            self.socket.bind(self.addr)
            self._is_bound = True
            logger.info(f"ZMQ Projector (HUD) BINDING on {self.addr}")
        except Exception as e:
            logger.error(f"ZMQ Projector BIND error: {e}")

    def project_hud(self, data: Dict[str, Any]):
        """
        Sends HUD snapshot to MT5.
        
        Args:
            data: {
                "fsm_state": "autonomous",
                "pilot": "Neural Net",
                "confidence": 0.87,
                "vault": {"balance": 150.0, "status": "LOCKED"},
                "regime": "TRENDING",
                "ema_colors": {"wind": "#00FF00", ...}
            }
        """
        if not self._is_bound:
            self.start()
            
        try:
            payload = json.dumps(data)
            self.socket.send_string(payload, zmq.NOBLOCK)
        except Exception as e:
            logger.debug(f"HUD Projection failed (normal if no sub): {e}")

    def stop(self):
        self.socket.close()
        self.context.term()

# Global Instance
zmq_projector = ZMQProjector()
