"""
FEAT Module T: TIEMPO (Sincronización Operativa)
=================================================
El filtro de Market Timing que alinea la operación con las ventanas de mayor liquidez.

Este módulo es un KILL SWITCH:
- Si retorna CLOSED, los demás módulos NO se ejecutan.
- Si retorna OPEN, el flujo continúa a Módulo F (Forma).

Kill Zones (GMT):
- London: 02:00-05:00 (Manipulación)
- New York: 12:00-15:00 (Expansión principal)
- London Close: 16:00-17:00 (Retrocesos)
- Asia: 20:00-04:00 (Solo observar, NO operar)
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Literal
from enum import Enum

logger = logging.getLogger("FEAT.Tiempo")


class KillZone(Enum):
    LONDON_OPEN = "LONDON_OPEN"
    NEW_YORK = "NEW_YORK"
    LONDON_CLOSE = "LONDON_CLOSE"
    ASIA = "ASIA"
    OFF_HOURS = "OFF_HOURS"


class MarketStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PAUSE = "PAUSE"  # Near high-impact news


# Kill Zone definitions in GMT
KILL_ZONES_GMT = {
    KillZone.LONDON_OPEN: {"start": 2, "end": 5},
    KillZone.NEW_YORK: {"start": 12, "end": 15},
    KillZone.LONDON_CLOSE: {"start": 16, "end": 17},
    KillZone.ASIA: {"start": 20, "end": 4},  # Wraps midnight
}

# Trading permissions per Kill Zone
ZONE_PERMISSIONS = {
    KillZone.LONDON_OPEN: {"can_trade": True, "risk_level": "MEDIUM", "note": "Buscar manipulación"},
    KillZone.NEW_YORK: {"can_trade": True, "risk_level": "HIGH", "note": "Máxima liquidez"},
    KillZone.LONDON_CLOSE: {"can_trade": True, "risk_level": "LOW", "note": "Solo cerrar posiciones"},
    KillZone.ASIA: {"can_trade": False, "risk_level": "NONE", "note": "Solo observar"},
    KillZone.OFF_HOURS: {"can_trade": False, "risk_level": "NONE", "note": "Mercado dormido"},
}


def get_current_killzone(server_time_gmt: datetime = None) -> KillZone:
    """
    Determina la Kill Zone activa basándose en la hora GMT.
    
    Returns:
        KillZone enum value
    """
    if server_time_gmt is None:
        server_time_gmt = datetime.now(timezone.utc)
    
    hour = server_time_gmt.hour
    
    # Check each zone
    for zone, times in KILL_ZONES_GMT.items():
        start, end = times["start"], times["end"]
        
        # Handle Asia (wraps midnight)
        if start > end:
            if hour >= start or hour < end:
                return zone
        else:
            if start <= hour < end:
                return zone
    
    return KillZone.OFF_HOURS


def check_h4_alignment(h4_candle_direction: str, proposed_direction: str) -> Dict[str, Any]:
    """
    Regla de Oro: NUNCA operar en contra de la vela H4.
    
    Args:
        h4_candle_direction: "BULLISH", "BEARISH", or "NEUTRAL"
        proposed_direction: "BUY" or "SELL"
        
    Returns:
        Dict with alignment status and allowed actions
    """
    h4_candle_direction = h4_candle_direction.upper()
    proposed_direction = proposed_direction.upper()
    
    # Alignment matrix
    if h4_candle_direction == "BULLISH":
        if proposed_direction == "BUY":
            return {"aligned": True, "constraint": "LONGS_ONLY", "reason": "H4 alcista, compras permitidas"}
        else:
            return {"aligned": False, "constraint": "NO_SHORTS", "reason": "VETO: H4 alcista, ventas prohibidas"}
    
    elif h4_candle_direction == "BEARISH":
        if proposed_direction == "SELL":
            return {"aligned": True, "constraint": "SHORTS_ONLY", "reason": "H4 bajista, ventas permitidas"}
        else:
            return {"aligned": False, "constraint": "NO_LONGS", "reason": "VETO: H4 bajista, compras prohibidas"}
    
    else:  # NEUTRAL
        return {"aligned": True, "constraint": "BOTH", "reason": "H4 neutral, ambas direcciones posibles"}


def check_news_filter(news_in_minutes: int) -> Dict[str, Any]:
    """
    Filtro de noticias de alto impacto.
    
    Args:
        news_in_minutes: Minutos hasta la próxima noticia de alto impacto
        
    Returns:
        Dict with news filter status
    """
    if news_in_minutes <= 30:
        return {
            "status": "PAUSE",
            "reason": f"Noticia de alto impacto en {news_in_minutes} minutos",
            "action": "WAIT_30_MIN_AFTER"
        }
    elif news_in_minutes <= 60:
        return {
            "status": "CAUTION",
            "reason": f"Noticia en {news_in_minutes} minutos, reducir riesgo",
            "action": "REDUCE_POSITION_SIZE"
        }
    else:
        return {
            "status": "CLEAR",
            "reason": "Sin noticias próximas",
            "action": "NORMAL_RISK"
        }


def analyze_tiempo(
    server_time_gmt: str = None,
    h4_candle: str = "NEUTRAL",
    news_in_minutes: int = 999,
    proposed_direction: str = None
) -> Dict[str, Any]:
    """
    🕐 FEAT MODULE T: Análisis completo de Timing.
    
    Este es el primer filtro de la cadena FEAT.
    Si retorna status="CLOSED", NO ejecutar los demás módulos.
    
    Args:
        server_time_gmt: Hora del servidor en formato ISO o None para usar actual
        h4_candle: Dirección de la vela H4 actual ("BULLISH", "BEARISH", "NEUTRAL")
        news_in_minutes: Minutos hasta la próxima noticia de alto impacto
        proposed_direction: "BUY" o "SELL" si ya hay una dirección propuesta
        
    Returns:
        Dict con análisis completo de tiempo
    """
    # Parse time
    if server_time_gmt:
        try:
            if isinstance(server_time_gmt, str):
                # Handle various formats
                if "T" in server_time_gmt:
                    time_obj = datetime.fromisoformat(server_time_gmt.replace("Z", "+00:00"))
                else:
                    time_obj = datetime.strptime(server_time_gmt, "%H:%M:%S").replace(tzinfo=timezone.utc)
            else:
                time_obj = server_time_gmt
        except:
            time_obj = datetime.now(timezone.utc)
    else:
        time_obj = datetime.now(timezone.utc)
    
    # Get current Kill Zone
    current_kz = get_current_killzone(time_obj)
    zone_info = ZONE_PERMISSIONS[current_kz]
    
    # Check news filter
    news_check = check_news_filter(news_in_minutes)
    
    # Determine market status
    if news_check["status"] == "PAUSE":
        market_status = MarketStatus.PAUSE
    elif zone_info["can_trade"]:
        market_status = MarketStatus.OPEN
    else:
        market_status = MarketStatus.CLOSED
    
    # H4 alignment (if direction proposed)
    h4_alignment = None
    if proposed_direction:
        h4_alignment = check_h4_alignment(h4_candle, proposed_direction)
        if not h4_alignment["aligned"]:
            market_status = MarketStatus.CLOSED  # Veto by H4
    
    # Build response
    result = {
        "module": "FEAT_Tiempo",
        "status": market_status.value,
        "session": current_kz.value,
        "server_time_gmt": time_obj.strftime("%H:%M:%S"),
        "analysis": {
            "kill_zone": {
                "name": current_kz.value,
                "can_trade": zone_info["can_trade"],
                "risk_level": zone_info["risk_level"],
                "note": zone_info["note"]
            },
            "h4_candle": h4_candle,
            "h4_alignment": h4_alignment,
            "news_filter": news_check
        }
    }
    
    # Bias constraint
    if h4_alignment:
        result["bias_constraint"] = h4_alignment["constraint"]
    elif h4_candle == "BULLISH":
        result["bias_constraint"] = "LONGS_PREFERRED"
    elif h4_candle == "BEARISH":
        result["bias_constraint"] = "SHORTS_PREFERRED"
    else:
        result["bias_constraint"] = "NEUTRAL"
    
    # Action instruction
    if market_status == MarketStatus.OPEN:
        result["instruction"] = "PROCEED_TO_MODULE_F"
    elif market_status == MarketStatus.PAUSE:
        result["instruction"] = "WAIT_FOR_NEWS_CLEARANCE"
    else:
        result["instruction"] = "STOP_CHAIN"
    
    logger.info(f"[FEAT-T] Session={current_kz.value}, Status={market_status.value}, H4={h4_candle}")
    
    return result


# =============================================================================
# Async wrapper for MCP
# =============================================================================

async def feat_check_tiempo(
    server_time_gmt: str = None,
    h4_candle: str = "NEUTRAL",
    news_in_minutes: int = 999,
    proposed_direction: str = None
) -> Dict[str, Any]:
    """
    MCP Tool: FEAT Module T - Tiempo analysis.
    """
    return analyze_tiempo(server_time_gmt, h4_candle, news_in_minutes, proposed_direction)
