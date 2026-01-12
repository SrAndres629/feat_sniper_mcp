"""
FEAT Module T: TIEMPO ADVANCED (Ciclo Diario de Liquidez)
==========================================================
Sistema avanzado de Market Timing basado en ciclos de liquidez NY.

Fases del Ciclo:
- ASIA_OVERNIGHT: 00:00-04:00 NY - Intensidad BAJA
- PRE_LONDON: 04:00-06:00 NY - Intensidad MEDIA
- LONDON_KILLZONE: 06:00-09:00 NY - Intensidad ALTA
- LONDON_LUNCH: 09:00-11:00 NY - Intensidad MEDIA
- NY_KILLZONE: 11:30-13:00 NY - Intensidad MUY ALTA
- NY_MIDDAY: 13:00-15:00 NY - Intensidad ALTA
- NY_CLOSE: 15:00-17:00 NY - Intensidad MEDIA (decae)
- POST_MARKET: 17:00+ NY - Intensidad BAJA

Cierres H4 (NY): 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Literal
from enum import Enum

logger = logging.getLogger("FEAT.TiempoAdvanced")


# =============================================================================
# ENUMS Y CONSTANTES
# =============================================================================

class LiquidityPhase(Enum):
    ASIA_OVERNIGHT = "ASIA_OVERNIGHT"
    ASIA_DEEP = "ASIA_DEEP"
    PRE_LONDON = "PRE_LONDON"
    LONDON_KILLZONE = "LONDON_KILLZONE"
    LONDON_LUNCH = "LONDON_LUNCH"
    PRE_NY = "PRE_NY"
    NY_KILLZONE = "NY_KILLZONE"
    NY_MIDDAY = "NY_MIDDAY"
    NY_CLOSE = "NY_CLOSE"
    POST_MARKET = "POST_MARKET"


class LiquidityIntensity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class OperationalMode(Enum):
    WAIT = "WAIT"           # No operar, dormir
    PREPARE = "PREPARE"     # Marcar niveles, preparar órdenes
    EXECUTE = "EXECUTE"     # Ejecutar con escalonamiento
    MANAGE_ONLY = "MANAGE"  # Solo gestionar posiciones abiertas


# Mapeo de fases con horarios NY y características
LIQUIDITY_PHASES = {
    # (hora_inicio, hora_fin): (fase, intensidad, modo, descripción)
    (0, 1): (LiquidityPhase.ASIA_OVERNIGHT, LiquidityIntensity.LOW, OperationalMode.WAIT, 
             "Rango estrecho, spreads amplios"),
    (1, 4): (LiquidityPhase.ASIA_DEEP, LiquidityIntensity.LOW, OperationalMode.WAIT,
             "Liquidez mínima; evitar tamaño alto"),
    (4, 6): (LiquidityPhase.PRE_LONDON, LiquidityIntensity.MEDIUM, OperationalMode.PREPARE,
             "Primeros indicios de posicionamiento; marcar POCs"),
    (6, 9): (LiquidityPhase.LONDON_KILLZONE, LiquidityIntensity.HIGH, OperationalMode.EXECUTE,
             "Gran volumen; barridos de stops; ejecutar escalonado"),
    (9, 11): (LiquidityPhase.LONDON_LUNCH, LiquidityIntensity.MEDIUM, OperationalMode.MANAGE_ONLY,
              "Transición; momentum puede consolidar"),
    (11, 13): (LiquidityPhase.NY_KILLZONE, LiquidityIntensity.VERY_HIGH, OperationalMode.EXECUTE,
               "PICO de liquidez y volatilidad; ejecutar con slices"),
    (13, 15): (LiquidityPhase.NY_MIDDAY, LiquidityIntensity.HIGH, OperationalMode.MANAGE_ONLY,
               "Continuación o consolidación; trailing stops"),
    (15, 17): (LiquidityPhase.NY_CLOSE, LiquidityIntensity.MEDIUM, OperationalMode.MANAGE_ONLY,
               "Recolocación de stops; reducir tamaño"),
    (17, 20): (LiquidityPhase.POST_MARKET, LiquidityIntensity.LOW, OperationalMode.WAIT,
               "Liquidez decrece; preparar macro-niveles"),
    (20, 24): (LiquidityPhase.ASIA_OVERNIGHT, LiquidityIntensity.LOW, OperationalMode.WAIT,
               "Rango, acumulación ligera")
}

# Cierres H4 importantes (hora NY)
H4_CLOSES_NY = [0, 4, 8, 12, 16, 20]


# =============================================================================
# FUNCIONES DE CONVERSIÓN TEMPORAL
# =============================================================================

def utc_to_ny(utc_time: datetime) -> datetime:
    """
    Convierte hora UTC a hora de Nueva York (Eastern Time).
    Considera DST automáticamente.
    """
    # Offset estándar NY: UTC-5 (EST) o UTC-4 (EDT)
    # Simplificación: usamos -5 por defecto (ajustar según DST)
    ny_offset = timedelta(hours=-5)
    return utc_time + ny_offset


def get_current_ny_time() -> datetime:
    """Obtiene la hora actual en Nueva York."""
    return utc_to_ny(datetime.now(timezone.utc))


def parse_time_input(time_input: str = None) -> datetime:
    """
    Parsea una entrada de tiempo (string o None).
    Retorna datetime en UTC.
    """
    if time_input is None:
        return datetime.now(timezone.utc)
    
    try:
        if "T" in time_input:
            return datetime.fromisoformat(time_input.replace("Z", "+00:00"))
        elif ":" in time_input:
            # Solo hora HH:MM
            parts = time_input.split(":")
            now = datetime.now(timezone.utc)
            return now.replace(hour=int(parts[0]), minute=int(parts[1]), second=0)
        else:
            return datetime.now(timezone.utc)
    except:
        return datetime.now(timezone.utc)


# =============================================================================
# FUNCIONES DE ANÁLISIS DE FASE
# =============================================================================

def get_liquidity_phase(ny_hour: int) -> Dict[str, Any]:
    """
    Determina la fase de liquidez basada en la hora NY.
    """
    for (start, end), (phase, intensity, mode, desc) in LIQUIDITY_PHASES.items():
        if start <= ny_hour < end:
            return {
                "phase": phase,
                "intensity": intensity,
                "operational_mode": mode,
                "description": desc
            }
    
    # Default
    return {
        "phase": LiquidityPhase.ASIA_OVERNIGHT,
        "intensity": LiquidityIntensity.LOW,
        "operational_mode": OperationalMode.WAIT,
        "description": "Fuera de horario principal"
    }


def is_near_h4_close(ny_hour: int, ny_minute: int, margin_minutes: int = 15) -> bool:
    """
    Verifica si estamos cerca de un cierre H4 (±margin minutos).
    Los cierres H4 son momentos de posible barrido institucional.
    """
    for h4_close in H4_CLOSES_NY:
        if ny_hour == h4_close and ny_minute < margin_minutes:
            return True
        if ny_hour == (h4_close - 1) % 24 and ny_minute >= (60 - margin_minutes):
            return True
    return False


def calculate_intensity_multiplier(
    base_intensity: LiquidityIntensity,
    news_upcoming: bool = False,
    is_h4_close: bool = False
) -> Dict[str, Any]:
    """
    Ajusta la intensidad basada en factores adicionales.
    """
    multiplier = 1.0
    warnings = []
    
    if news_upcoming:
        multiplier *= 1.5  # Aumenta volatilidad esperada
        warnings.append("⚠️ Noticia próxima - Ampliar spreads y reducir tamaño")
    
    if is_h4_close:
        multiplier *= 1.2  # Posibles barridos
        warnings.append("🕐 Cierre H4 próximo - Proteger posiciones")
    
    # Ajustar intensidad
    intensity_order = [LiquidityIntensity.LOW, LiquidityIntensity.MEDIUM, 
                       LiquidityIntensity.HIGH, LiquidityIntensity.VERY_HIGH]
    
    current_idx = intensity_order.index(base_intensity)
    adjusted_idx = min(len(intensity_order) - 1, int(current_idx * multiplier))
    
    return {
        "adjusted_intensity": intensity_order[adjusted_idx],
        "multiplier": multiplier,
        "warnings": warnings
    }


# =============================================================================
# FUNCIÓN PRINCIPAL DE ANÁLISIS
# =============================================================================

def analyze_tiempo_advanced(
    server_time_utc: str = None,
    news_event_upcoming: bool = False,
    h4_direction: str = "NEUTRAL",
    h1_direction: str = "NEUTRAL"
) -> Dict[str, Any]:
    """
    🕐 FEAT MODULE T ADVANCED: Ciclo Diario de Liquidez XAU/USD
    
    Analiza la hora actual y determina:
    - Fase del ciclo de liquidez
    - Intensidad esperada (M15/H1/H4)
    - Modo operativo (WAIT/PREPARE/EXECUTE/MANAGE)
    - Checklist de confirmación multitemporal
    
    Args:
        server_time_utc: Hora del servidor en formato ISO o HH:MM (UTC)
        news_event_upcoming: Si hay noticias de alto impacto próximas
        h4_direction: Dirección de H4 para validación
        h1_direction: Dirección de H1 para validación
        
    Returns:
        Dict con análisis completo de timing institucional
    """
    # Parse y convertir tiempo
    utc_time = parse_time_input(server_time_utc)
    ny_time = utc_to_ny(utc_time)
    ny_hour = ny_time.hour
    ny_minute = ny_time.minute
    
    # Obtener fase de liquidez
    phase_info = get_liquidity_phase(ny_hour)
    phase = phase_info["phase"]
    base_intensity = phase_info["intensity"]
    base_mode = phase_info["operational_mode"]
    
    # Verificar cercanía a cierre H4
    near_h4_close = is_near_h4_close(ny_hour, ny_minute)
    
    # Ajustar intensidad por factores externos
    intensity_adj = calculate_intensity_multiplier(base_intensity, news_event_upcoming, near_h4_close)
    
    # Determinar modo operativo final
    final_mode = base_mode
    if news_event_upcoming and base_mode == OperationalMode.EXECUTE:
        final_mode = OperationalMode.PREPARE  # Reducir a preparación si hay noticias
    
    # Validación multitemporal
    multitemporal_check = {
        "h4_aligned": h4_direction != "NEUTRAL",
        "h1_aligned": h1_direction != "NEUTRAL",
        "h4_h1_match": h4_direction == h1_direction,
        "ready_for_m15": h4_direction == h1_direction and h4_direction != "NEUTRAL"
    }
    
    # Generar instrucción de acción
    if final_mode == OperationalMode.WAIT:
        instruction = "STOP_CHAIN"
        action_text = "Mercado dormido. No operar hasta próxima Kill Zone."
    elif final_mode == OperationalMode.PREPARE:
        instruction = "ALERT_ONLY"
        action_text = "Preparar niveles y ordenes limit. Esperar confirmación."
    elif final_mode == OperationalMode.EXECUTE:
        if multitemporal_check["ready_for_m15"]:
            instruction = "PROCEED_TO_MODULE_F"
            action_text = "Kill Zone activa + Alineación HTF. Buscar entrada en M15."
        else:
            instruction = "WAIT_FOR_ALIGNMENT"
            action_text = "Kill Zone activa pero sin alineación HTF. Esperar."
    else:  # MANAGE
        instruction = "MANAGE_POSITIONS"
        action_text = "Solo gestionar posiciones. No abrir nuevas."
    
    # Risk warning
    risk_warnings = intensity_adj["warnings"].copy()
    if base_intensity == LiquidityIntensity.VERY_HIGH:
        risk_warnings.append("⚡ Volatilidad extrema - Usar slicing y stops amplios")
    
    result = {
        "module": "FEAT_TIEMPO_ADVANCED",
        "status": "ANALYZED",
        "timestamp_utc": utc_time.isoformat(),
        
        # Tiempo
        "ny_time": ny_time.strftime("%H:%M"),
        "ny_hour": ny_hour,
        "utc_time": utc_time.strftime("%H:%M"),
        
        # Fase de liquidez
        "phase_name": phase.value,
        "phase_description": phase_info["description"],
        "liquidity_intensity": intensity_adj["adjusted_intensity"].value,
        "base_intensity": base_intensity.value,
        
        # Modo operativo
        "operational_mode": final_mode.value,
        "instruction": instruction,
        "action_text": action_text,
        
        # Alertas
        "near_h4_close": near_h4_close,
        "news_upcoming": news_event_upcoming,
        "risk_warnings": risk_warnings,
        
        # Checklist multitemporal
        "multitemporal_check": multitemporal_check,
        "checklist": {
            "h4_check": "✅ PASS" if multitemporal_check["h4_aligned"] else "❌ REQUIRED",
            "h1_check": "✅ PASS" if multitemporal_check["h1_aligned"] else "❌ REQUIRED",
            "htf_alignment": "✅ ALIGNED" if multitemporal_check["h4_h1_match"] else "⚠️ DIVERGENT",
            "m15_trigger": "🔍 SEARCH" if multitemporal_check["ready_for_m15"] else "⏳ WAIT"
        }
    }
    
    logger.info(f"[FEAT-T] NY={ny_time.strftime('%H:%M')}, Phase={phase.value}, Mode={final_mode.value}")
    
    return result


# =============================================================================
# FUNCIÓN SIMPLIFICADA PARA RETROCOMPATIBILIDAD
# =============================================================================

def analyze_tiempo(
    server_time_gmt: str = None,
    h4_candle: str = "NEUTRAL",
    news_in_minutes: int = 999,
    proposed_direction: str = None
) -> Dict[str, Any]:
    """
    Wrapper de retrocompatibilidad para analyze_tiempo_advanced.
    """
    news_upcoming = news_in_minutes <= 30
    
    result = analyze_tiempo_advanced(
        server_time_utc=server_time_gmt,
        news_event_upcoming=news_upcoming,
        h4_direction=h4_candle,
        h1_direction=h4_candle  # Usar H4 como proxy si no tenemos H1
    )
    
    # Convertir a formato legacy
    legacy_result = {
        "module": "FEAT_Tiempo",
        "status": "OPEN" if result["operational_mode"] in ["EXECUTE", "PREPARE"] else "CLOSED",
        "session": result["phase_name"],
        "server_time_gmt": result["utc_time"],
        "instruction": result["instruction"],
        "analysis": {
            "kill_zone": {
                "name": result["phase_name"],
                "can_trade": result["operational_mode"] in ["EXECUTE", "PREPARE"],
                "risk_level": result["liquidity_intensity"],
                "note": result["phase_description"]
            },
            "h4_candle": h4_candle,
            "news_filter": {"status": "PAUSE" if news_upcoming else "CLEAR"}
        },
        "bias_constraint": "LONGS_PREFERRED" if h4_candle == "BULLISH" else (
            "SHORTS_PREFERRED" if h4_candle == "BEARISH" else "NEUTRAL"
        ),
        
        # Nuevo: datos avanzados
        "advanced": {
            "ny_time": result["ny_time"],
            "liquidity_intensity": result["liquidity_intensity"],
            "operational_mode": result["operational_mode"],
            "checklist": result["checklist"],
            "risk_warnings": result["risk_warnings"]
        }
    }
    
    return legacy_result


# =============================================================================
# ASYNC WRAPPERS PARA MCP
# =============================================================================

async def feat_check_tiempo(
    server_time_gmt: str = None,
    h4_candle: str = "NEUTRAL",
    news_in_minutes: int = 999
) -> Dict[str, Any]:
    """MCP Tool: FEAT Module T - Tiempo (retrocompatible)."""
    return analyze_tiempo(server_time_gmt, h4_candle, news_in_minutes)


async def feat_check_tiempo_advanced(
    server_time_utc: str = None,
    news_event_upcoming: bool = False,
    h4_direction: str = "NEUTRAL",
    h1_direction: str = "NEUTRAL"
) -> Dict[str, Any]:
    """MCP Tool: FEAT Module T ADVANCED - Ciclo de Liquidez."""
    return analyze_tiempo_advanced(server_time_utc, news_event_upcoming, h4_direction, h1_direction)
