import logging
import json
from app.core.zmq_bridge import zmq_bridge

logger = logging.getLogger("feat.hud")

class ZMQProjector:
    """
    Módulo 8: The HUD.
    Proyección visual en MT5 via ZMQ.
    """
    def __init__(self):
        self.bridge = None

    def set_bridge(self, bridge):
        self.bridge = bridge

    async def draw_zone(self, symbol: str, high: float, low: float, color: str = "clrGreen", name: str = "zone"):
        """Dibuja rectángulo (FVG/Block)."""
        await self._send({
            "sub_action": "DRAW_RECT",
            "symbol": symbol,
            "name": name,
            "price1": high,
            "price2": low,
            "color": color,
            "fill": True
        })

    async def draw_arrow(self, symbol: str, price: float, direction: str, color: str = None):
        """Dibuja flecha de señal."""
        arrow_code = 233 if direction == "BUY" else 234 
        if not color: color = "clrLime" if direction == "BUY" else "clrRed"
        
        await self._send({
            "sub_action": "DRAW_ARROW",
            "symbol": symbol,
            "price": price,
            "code": arrow_code,
            "color": color
        })
    
    async def clear_all(self):
        await self._send({"sub_action": "CLEAR_ALL"})

    async def _send(self, payload: dict):
        if self.bridge and self.bridge.running:
             # Wrapper para send_command action="HUD_UPDATE"
             await self.bridge.send_command("HUD_UPDATE", **payload)
        else:
            pass # Silent fail if offline

# Singleton
hud_projector = ZMQProjector()
