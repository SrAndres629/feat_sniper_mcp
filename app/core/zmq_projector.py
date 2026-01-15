import logging
import json
import time
from datetime import datetime
from typing import List, Dict, Any
from app.core.zmq_bridge import zmq_bridge

logger = logging.getLogger("feat.hud")

class ZMQProjector:
    """
    Módulo 8: The HUD (FEAT Institutional V2).
    Proyección visual avanzada en MT5 via ZMQ.
    """
    def __init__(self):
        self.bridge = zmq_bridge

    async def broadcast_full_narrative(self, 
                                     regime: str, 
                                     confidence: float, 
                                     feat_score: float, 
                                     vault_active: bool,
                                     pvp_level: float,       # El "Imán" del precio (Feat Score Price)
                                     structure_map: dict,    # {'last_bos': 2040.5, 'type': 'BULLISH'}
                                     active_zones: list,     # [{'top': 2045, 'bottom': 2043, 'type': 'SUPPLY'}]
                                     session_state: str      # "NY_KILLZONE_OPEN"
                                     ):
        """
        Proyecta la narrativa completa institucional al HUD de MT5.
        """
        payload = {
            "sub_action": "NARRATIVE_UPDATE",
            "timestamp": int(time.time()),
            
            # 1. Dashboard Data
            "regime": regime,
            "ai_confidence": f"{int(confidence * 100)}",
            "vault_active": "true" if vault_active else "false",
            
            # 2. PVP / FEAT Logic (La evolución del indicador)
            "feat_pvp_price": f"{pvp_level:.2f}", 
            "feat_score_val": f"{feat_score:.2f}",
            
            # 3. Estructura y Espacio
            "struct_level": f"{structure_map.get('last_bos', 0):.2f}",
            "struct_type": structure_map.get('type', 'NONE'),
            
            # 4. Tiempo (Killzones)
            "session": session_state
        }
        
        # Serializamos zonas como string separado (MQL5 lo parseará)
        # Formato: "top|bottom|type#top|bottom|type"
        zones_str = ""
        for z in active_zones:
            zones_str += f"{z['top']}|{z['bottom']}|{z['type']}#"
        
        payload["zones_packed"] = zones_str

        await self._send(payload)

    async def broadcast_system_state(self, regime: str, confidence: float, feat_score: float, vault_active: bool, p_high: float = 0.0, p_low: float = 0.0):
        """Envía el latido del sistema al HUD de MT5 (Legacy support)."""
        conf_int = int(confidence * 100) if confidence <= 1.0 else int(confidence)
        payload = {
            "sub_action": "STATE_UPDATE",
            "regime": regime,
            "ai_confidence": str(conf_int),
            "feat_index": f"{feat_score:.2f}",
            "vault_active": "true" if vault_active else "false",
            "proj_high": f"{p_high:.2f}",
            "proj_low": f"{p_low:.2f}"
        }
        await self._send(payload)

    async def draw_zone(self, symbol: str, high: float, low: float, color: str = "clrGreen", name: str = "zone"):
        """Dibuja rectángulo (Legacy support)."""
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
             await self.bridge.send_raw(payload)

# Singleton
hud_projector = ZMQProjector()
