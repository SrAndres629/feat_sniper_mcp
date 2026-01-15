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

    async def project_telemetry(self, symbol: str, feat_index: float, regime: str, ai_confidence: float, projections: dict = None, vault_active: bool = False):
        """
        Broadcasting de telemetría completa a MT5.
        DEPRECATED: Use broadcast_system_state.
        """
        await self.broadcast_system_state(regime, ai_confidence, feat_index, vault_active, 
                                          projections.get("high", 0) if projections else 0.0,
                                          projections.get("low", 0) if projections else 0.0)

    async def broadcast_system_state(self, regime: str, confidence: float, feat_score: float, vault_active: bool, p_high: float = 0.0, p_low: float = 0.0):
        """Envía el latido del sistema al HUD de MT5."""
        # Normalizar confianza a 0-100 para visualización
        conf_int = int(confidence * 100) if confidence <= 1.0 else int(confidence)
        
        payload = {
            "sub_action": "STATE_UPDATE", # Tag para debug
            "regime": regime,             # Trend, Ranging, etc.
            "ai_confidence": str(conf_int),
            "feat_index": f"{feat_score:.2f}",
            "vault_active": "true" if vault_active else "false",
            "proj_high": f"{p_high:.2f}",
            "proj_low": f"{p_low:.2f}"
        }
        # Enviar al bridge
        await self._send(payload)

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
