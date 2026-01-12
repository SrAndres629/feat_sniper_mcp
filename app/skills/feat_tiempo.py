"""
FEAT Module T: TIEMPO INSTITUCIONAL COMPLETO
==============================================
Sistema de timing basado en flujos institucionales XAU/USD (GC Futures).

FUENTES:
- CME Gold Futures market hours
- LBMA Gold Price benchmarks (AM/PM fixes)
- Shanghai Gold Exchange (SGE) benchmarks
- Academic literature on intraday seasonality

TIMEZONE BASE: Bolivia (UTC-4) / NY (EST/EDT)

FIXES INSTITUCIONALES (Puntos de Anclaje):
- LBMA AM Fix: ~10:30 UTC → 06:30 Bolivia
- LBMA PM Fix: ~15:00 UTC → 11:00 Bolivia
- SGE Morning: ~02:15 UTC → 22:15 Bolivia (día anterior)
- SGE Afternoon: ~06:15 UTC → 02:15 Bolivia
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Tuple
from enum import Enum
import pytz

logger = logging.getLogger("FEAT.TiempoInstitucional")


# =============================================================================
# TIMEZONES
# =============================================================================

try:
    NY_TZ = pytz.timezone('America/New_York')
    BOLIVIA_TZ = pytz.timezone('America/La_Paz')
    UTC_TZ = pytz.UTC
except:
    NY_TZ = None
    BOLIVIA_TZ = None
    UTC_TZ = None


# =============================================================================
# ENUMS
# =============================================================================

class SessionPhase(Enum):
    GLOBEX_OPEN = "GLOBEX_OPEN"           # 19:00-22:00 Bolivia
    ASIA_SGE = "ASIA_SGE"                  # 22:00-02:30 Bolivia
    PRE_LONDON = "PRE_LONDON"              # 02:30-04:00 Bolivia
    LONDON_OPEN = "LONDON_OPEN"            # 04:00-06:30 Bolivia
    LBMA_AM_FIX = "LBMA_AM_FIX"           # 06:30 Bolivia (±20min)
    LONDON_MORNING = "LONDON_MORNING"      # 06:50-09:00 Bolivia
    KILLZONE_OVERLAP = "KILLZONE_OVERLAP"  # 09:00-13:00 Bolivia ⭐
    LBMA_PM_FIX = "LBMA_PM_FIX"           # 11:00 Bolivia (±20min)
    NY_AFTERNOON = "NY_AFTERNOON"          # 13:00-18:00 Bolivia
    DAILY_PAUSE = "DAILY_PAUSE"            # 18:00-19:00 Bolivia


class WeeklyCycle(Enum):
    INDUCTION = "INDUCTION"
    DIRECTION = "DIRECTION"
    MIDWEEK = "MIDWEEK"
    EXPANSION = "EXPANSION"
    CLOSING = "CLOSING"


class EntryTemplate(Enum):
    TEMPLATE_A = "CONFIRMACION_FUERTE"    # D1+H4 aligned, overlap, break+retest
    TEMPLATE_B = "CONTINUATION_POST_FIX"  # Post LBMA/SGE continuation
    TEMPLATE_C = "SWEEP_REVERSAL"         # Sweep + rejection


# =============================================================================
# INSTITUTIONAL FIXES (Bolivia Time - UTC-4)
# =============================================================================

INSTITUTIONAL_FIXES = {
    "LBMA_AM": {"hour": 6, "minute": 30, "importance": "HIGH", "margin_min": 20},
    "LBMA_PM": {"hour": 11, "minute": 0, "importance": "HIGH", "margin_min": 20},
    "SGE_MORNING": {"hour": 22, "minute": 15, "importance": "MEDIUM", "margin_min": 15},  # prev day
    "SGE_AFTERNOON": {"hour": 2, "minute": 15, "importance": "MEDIUM", "margin_min": 15},
}


# =============================================================================
# SESSION MATRIX (Bolivia Time - UTC-4)
# =============================================================================

SESSION_MATRIX = {
    # (start, end): {phase, liquidity, expansion_prob, risk_mult, action}
    (19, 22): {
        "phase": SessionPhase.GLOBEX_OPEN,
        "liquidity": 0.25,
        "expansion_prob": 0.20,
        "risk_mult": 0.30,
        "action": "MARK_RANGE",
        "description": "Inicio Globex. Baja liquidez. Marcar rango nocturno."
    },
    (22, 2): {  # Wraps midnight
        "phase": SessionPhase.ASIA_SGE,
        "liquidity": 0.40,
        "expansion_prob": 0.30,
        "risk_mult": 0.40,
        "action": "PULLBACK_ENTRY",
        "description": "Asia/SGE. Micro-sweeps. Entrar en pullbacks a soporte asiático."
    },
    (2, 4): {
        "phase": SessionPhase.PRE_LONDON,
        "liquidity": 0.35,
        "expansion_prob": 0.25,
        "risk_mult": 0.35,
        "action": "PREPARE",
        "description": "Pre-London. Marcar niveles para expansión europea."
    },
    (4, 6): {
        "phase": SessionPhase.LONDON_OPEN,
        "liquidity": 0.60,
        "expansion_prob": 0.55,
        "risk_mult": 0.65,
        "action": "BREAK_RETEST",
        "description": "Apertura Londres. Breakouts del rango asiático."
    },
    (6, 9): {
        "phase": SessionPhase.LONDON_MORNING,
        "liquidity": 0.70,
        "expansion_prob": 0.60,
        "risk_mult": 0.75,
        "action": "CONTINUATION",
        "description": "Londres mañana. Post-LBMA AM. Continuaciones."
    },
    (9, 13): {
        "phase": SessionPhase.KILLZONE_OVERLAP,
        "liquidity": 0.95,
        "expansion_prob": 0.85,
        "risk_mult": 1.00,
        "action": "EXECUTE_FULL",
        "description": "⭐ KILLZONE London-NY overlap. Máxima liquidez. Entradas principales."
    },
    (13, 18): {
        "phase": SessionPhase.NY_AFTERNOON,
        "liquidity": 0.50,
        "expansion_prob": 0.35,
        "risk_mult": 0.50,
        "action": "MANAGE_ONLY",
        "description": "NY tarde. Decrece liquidez. Solo gestión y re-tests."
    },
    (18, 19): {
        "phase": SessionPhase.DAILY_PAUSE,
        "liquidity": 0.10,
        "expansion_prob": 0.05,
        "risk_mult": 0.10,
        "action": "NO_TRADE",
        "description": "Pausa diaria. Spreads amplios. EVITAR."
    },
}


# =============================================================================
# WEEKLY CYCLE
# =============================================================================

WEEKLY_CYCLE = {
    0: {"phase": WeeklyCycle.INDUCTION, "risk_mult": 0.50, "trap_probability": 0.70},
    1: {"phase": WeeklyCycle.DIRECTION, "risk_mult": 1.00, "trap_probability": 0.20},
    2: {"phase": WeeklyCycle.MIDWEEK, "risk_mult": 0.85, "trap_probability": 0.35},
    3: {"phase": WeeklyCycle.EXPANSION, "risk_mult": 1.20, "trap_probability": 0.15},
    4: {"phase": WeeklyCycle.CLOSING, "risk_mult": 0.40, "trap_probability": 0.50},
    5: {"phase": WeeklyCycle.CLOSING, "risk_mult": 0.10, "trap_probability": 0.90},
    6: {"phase": WeeklyCycle.CLOSING, "risk_mult": 0.10, "trap_probability": 0.90},
}


# =============================================================================
# TIMEZONE FUNCTIONS
# =============================================================================

def get_bolivia_time(utc_time: datetime = None) -> datetime:
    """Convierte UTC a Bolivia (UTC-4)."""
    if utc_time is None:
        utc_time = datetime.now(timezone.utc)
    
    if BOLIVIA_TZ:
        try:
            if utc_time.tzinfo is None:
                utc_time = utc_time.replace(tzinfo=timezone.utc)
            return utc_time.astimezone(BOLIVIA_TZ)
        except:
            pass
    return utc_time - timedelta(hours=4)


def get_ny_time(utc_time: datetime = None) -> datetime:
    """Convierte UTC a NY."""
    if utc_time is None:
        utc_time = datetime.now(timezone.utc)
    
    if NY_TZ:
        try:
            if utc_time.tzinfo is None:
                utc_time = utc_time.replace(tzinfo=timezone.utc)
            return utc_time.astimezone(NY_TZ)
        except:
            pass
    return utc_time - timedelta(hours=5)


# =============================================================================
# SESSION DETECTION
# =============================================================================

def get_session_info(bolivia_hour: int) -> Dict[str, Any]:
    """Obtiene info de sesión basada en hora Bolivia."""
    for (start, end), info in SESSION_MATRIX.items():
        # Handle wrap around midnight
        if start > end:
            if bolivia_hour >= start or bolivia_hour < end:
                return info.copy()
        else:
            if start <= bolivia_hour < end:
                return info.copy()
    
    return SESSION_MATRIX[(18, 19)].copy()  # Default: pause


def check_near_fix(bolivia_hour: int, bolivia_minute: int) -> Dict[str, Any]:
    """Verifica proximidad a LBMA/SGE fixes."""
    current_total_min = bolivia_hour * 60 + bolivia_minute
    
    for fix_name, fix_info in INSTITUTIONAL_FIXES.items():
        fix_min = fix_info["hour"] * 60 + fix_info["minute"]
        margin = fix_info["margin_min"]
        
        distance = abs(current_total_min - fix_min)
        if distance <= margin:
            return {
                "near_fix": True,
                "fix_name": fix_name,
                "distance_minutes": distance,
                "importance": fix_info["importance"],
                "action": "CAUTION" if distance <= 10 else "AWARE"
            }
    
    return {"near_fix": False}


# =============================================================================
# ALIGNMENT SCORING
# =============================================================================

def calculate_alignment_score(
    d1_direction: str,
    h4_direction: str,
    h1_direction: str = None
) -> Dict[str, Any]:
    """
    Calcula score de alineación multi-timeframe.
    
    Reglas:
    - D1 + H4 aligned = 100% size
    - H4 aligned, D1 neutral = 50% size
    - D1 contrary = AVOID or minimal size
    """
    d1_dir = d1_direction.upper() if d1_direction else "NEUTRAL"
    h4_dir = h4_direction.upper() if h4_direction else "NEUTRAL"
    h1_dir = h1_direction.upper() if h1_direction else "NEUTRAL"
    
    # Full alignment
    if d1_dir == h4_dir and d1_dir != "NEUTRAL":
        alignment = "FULL_ALIGNMENT"
        size_mult = 1.0
        confidence = 0.90
        extra_confirmation = False
    
    # H4 aligned, D1 neutral
    elif h4_dir != "NEUTRAL" and d1_dir == "NEUTRAL":
        alignment = "PARTIAL_ALIGNMENT"
        size_mult = 0.50
        confidence = 0.65
        extra_confirmation = True
    
    # H4 aligned, D1 contrary
    elif h4_dir != "NEUTRAL" and d1_dir != "NEUTRAL" and h4_dir != d1_dir:
        alignment = "CONFLICT"
        size_mult = 0.20
        confidence = 0.35
        extra_confirmation = True
    
    # Both neutral
    else:
        alignment = "NO_TREND"
        size_mult = 0.30
        confidence = 0.40
        extra_confirmation = True
    
    # H1 confirmation bonus
    h1_bonus = 0
    if h1_dir == h4_dir and h1_dir != "NEUTRAL":
        h1_bonus = 0.10
        confidence = min(1.0, confidence + 0.10)
    
    return {
        "alignment_type": alignment,
        "size_multiplier": round(size_mult, 2),
        "confidence": round(confidence + h1_bonus, 2),
        "requires_extra_confirmation": extra_confirmation,
        "direction": h4_dir if h4_dir != "NEUTRAL" else d1_dir,
        "d1": d1_dir,
        "h4": h4_dir,
        "h1": h1_dir
    }


# =============================================================================
# ENTRY TEMPLATE MATCHING
# =============================================================================

def match_entry_template(
    session_phase: SessionPhase,
    alignment: Dict,
    near_fix: bool = False,
    has_sweep: bool = False
) -> Dict[str, Any]:
    """
    Determina qué plantilla de entrada aplica.
    """
    templates = []
    
    # Template A: Confirmación Fuerte
    if alignment["alignment_type"] == "FULL_ALIGNMENT" and \
       session_phase == SessionPhase.KILLZONE_OVERLAP:
        templates.append({
            "template": EntryTemplate.TEMPLATE_A.value,
            "name": "Confirmación Fuerte",
            "probability": 0.85,
            "rules": [
                "D1/H4 alineados ✅",
                "Overlap 09:00-13:00 ✅",
                "Esperar break+retest H1",
                "Tick-volume ≥ 1.5x media",
                "Stop bajo low del retest",
                "TP: 1.5x riesgo mínimo"
            ]
        })
    
    # Template B: Continuation Post-Fix
    if near_fix and alignment["size_multiplier"] >= 0.50:
        templates.append({
            "template": EntryTemplate.TEMPLATE_B.value,
            "name": "Continuación Post-Fix",
            "probability": 0.70,
            "rules": [
                f"Post-fix detectado ✅",
                "Confirmar con 2 velas H1 en dirección",
                "Tick-volume en aumento",
                "Entrada en retest del nivel del fix",
                "Tamaño: 50% si D1 neutral"
            ]
        })
    
    # Template C: Sweep Reversal
    if has_sweep:
        templates.append({
            "template": EntryTemplate.TEMPLATE_C.value,
            "name": "Sweep & Reversal",
            "probability": 0.75,
            "rules": [
                "Mecha barró stops ✅",
                "Fuerte vela de rechazo",
                "Entrar en 1ª vela de confirmación",
                "Stop bajo mínimo del sweep",
                "Funciona incluso sin alineación HTF clara"
            ]
        })
    
    if not templates:
        templates.append({
            "template": "NO_TEMPLATE",
            "name": "Sin plantilla clara",
            "probability": 0.30,
            "rules": ["Esperar mejor setup o killzone"]
        })
    
    return {"applicable_templates": templates, "best_template": templates[0]}


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_tiempo_institucional(
    server_time_utc: str = None,
    d1_direction: str = "NEUTRAL",
    h4_direction: str = "NEUTRAL",
    h1_direction: str = "NEUTRAL",
    has_sweep: bool = False,
    news_upcoming: bool = False
) -> Dict[str, Any]:
    """
    🕐 FEAT TIEMPO INSTITUCIONAL: Análisis completo para GC/XAU.
    
    Integra:
    - Sesiones globales (Globex, Asia, London, NY)
    - LBMA/SGE fixes
    - Ciclo semanal
    - Alineación multi-timeframe
    - Plantillas de entrada
    """
    # Parse time
    if server_time_utc:
        try:
            if isinstance(server_time_utc, str):
                if "T" in server_time_utc:
                    utc_time = datetime.fromisoformat(server_time_utc.replace("Z", "+00:00"))
                else:
                    parts = server_time_utc.split(":")
                    utc_time = datetime.now(timezone.utc).replace(hour=int(parts[0]), minute=int(parts[1]))
            else:
                utc_time = server_time_utc
        except:
            utc_time = datetime.now(timezone.utc)
    else:
        utc_time = datetime.now(timezone.utc)
    
    # Convert timezones
    bolivia_time = get_bolivia_time(utc_time)
    ny_time = get_ny_time(utc_time)
    
    bolivia_hour = bolivia_time.hour
    bolivia_minute = bolivia_time.minute
    weekday = bolivia_time.weekday()
    
    # Get session info
    session_info = get_session_info(bolivia_hour)
    session_phase = session_info["phase"]
    
    # Check fix proximity
    fix_check = check_near_fix(bolivia_hour, bolivia_minute)
    
    # Weekly cycle
    weekly_info = WEEKLY_CYCLE.get(weekday, WEEKLY_CYCLE[6])
    
    # Alignment scoring
    alignment = calculate_alignment_score(d1_direction, h4_direction, h1_direction)
    
    # Entry template matching
    templates = match_entry_template(
        session_phase, alignment, fix_check.get("near_fix", False), has_sweep
    )
    
    # Calculate final risk multiplier
    session_mult = session_info["risk_mult"]
    weekly_mult = weekly_info["risk_mult"]
    alignment_mult = alignment["size_multiplier"]
    
    # News penalty
    news_penalty = 0.5 if news_upcoming else 1.0
    
    # Fix warning penalty
    fix_penalty = 0.7 if fix_check.get("near_fix") and fix_check.get("distance_minutes", 20) <= 10 else 1.0
    
    combined_risk_mult = round(session_mult * weekly_mult * alignment_mult * news_penalty * fix_penalty, 3)
    combined_risk_mult = max(0.05, min(1.5, combined_risk_mult))
    
    # Build result
    result = {
        "module": "FEAT_TIEMPO_INSTITUCIONAL",
        "timestamp_utc": utc_time.isoformat() if hasattr(utc_time, 'isoformat') else str(utc_time),
        
        "time": {
            "bolivia": bolivia_time.strftime("%H:%M") if hasattr(bolivia_time, 'strftime') else str(bolivia_time),
            "ny": ny_time.strftime("%H:%M") if hasattr(ny_time, 'strftime') else str(ny_time),
            "day_of_week": bolivia_time.strftime("%A") if hasattr(bolivia_time, 'strftime') else str(weekday)
        },
        
        "session": {
            "phase": session_phase.value,
            "liquidity": session_info["liquidity"],
            "expansion_probability": session_info["expansion_prob"],
            "recommended_action": session_info["action"],
            "description": session_info["description"]
        },
        
        "weekly": {
            "phase": weekly_info["phase"].value,
            "trap_probability": weekly_info["trap_probability"],
            "risk_mult": weekly_info["risk_mult"]
        },
        
        "fixes": fix_check,
        
        "alignment": alignment,
        
        "templates": templates,
        
        "risk": {
            "session_mult": session_mult,
            "weekly_mult": weekly_mult,
            "alignment_mult": alignment_mult,
            "news_penalty": news_penalty,
            "fix_penalty": fix_penalty,
            "combined_risk_multiplier": combined_risk_mult,
            "recommended_size": "FULL" if combined_risk_mult > 0.8 else ("HALF" if combined_risk_mult > 0.4 else "QUARTER")
        },
        
        "ml_features": {
            # Session
            "bolivia_hour_normalized": bolivia_hour / 23.0,
            "is_killzone": 1 if session_phase == SessionPhase.KILLZONE_OVERLAP else 0,
            "is_london": 1 if session_phase in [SessionPhase.LONDON_OPEN, SessionPhase.LONDON_MORNING] else 0,
            "is_ny_afternoon": 1 if session_phase == SessionPhase.NY_AFTERNOON else 0,
            "liquidity_expected": session_info["liquidity"],
            "expansion_probability": session_info["expansion_prob"],
            
            # Weekly
            "is_induction": 1 if weekly_info["phase"] == WeeklyCycle.INDUCTION else 0,
            "is_expansion_day": 1 if weekly_info["phase"] == WeeklyCycle.EXPANSION else 0,
            "trap_probability": weekly_info["trap_probability"],
            
            # Fixes
            "near_fix": 1 if fix_check.get("near_fix") else 0,
            
            # Alignment
            "full_alignment": 1 if alignment["alignment_type"] == "FULL_ALIGNMENT" else 0,
            "alignment_confidence": alignment["confidence"],
            "size_multiplier": alignment["size_multiplier"],
            
            # Combined
            "combined_risk_multiplier": combined_risk_mult
        },
        
        "checklist": {
            "htf_aligned": "✅" if alignment["alignment_type"] == "FULL_ALIGNMENT" else "⚠️",
            "in_killzone": "✅" if session_phase == SessionPhase.KILLZONE_OVERLAP else "⏳",
            "near_fix_warning": "⚠️ CAUTION" if fix_check.get("near_fix") else "✅ CLEAR",
            "weekly_trap_risk": "⚠️ HIGH" if weekly_info["trap_probability"] > 0.5 else "✅ LOW",
            "recommended_action": templates["best_template"]["template"]
        },
        
        "guidance": {
            "can_trade": combined_risk_mult > 0.3,
            "best_action": session_info["action"],
            "size": "FULL" if combined_risk_mult > 0.8 else ("HALF" if combined_risk_mult > 0.4 else "QUARTER"),
            "cautions": []
        }
    }
    
    # Add cautions
    if weekly_info["trap_probability"] > 0.5:
        result["guidance"]["cautions"].append(f"📅 {weekly_info['phase'].value}: Alto riesgo de trampas")
    if fix_check.get("near_fix"):
        result["guidance"]["cautions"].append(f"⏰ Cerca de {fix_check.get('fix_name')}: posible slippage")
    if alignment["alignment_type"] == "CONFLICT":
        result["guidance"]["cautions"].append("⚠️ D1/H4 en conflicto: reducir tamaño o evitar")
    if session_info["action"] == "NO_TRADE":
        result["guidance"]["cautions"].append("🚫 Pausa diaria: EVITAR operaciones")
    
    logger.info(f"[FEAT-T] Bolivia={bolivia_time.strftime('%H:%M')}, Phase={session_phase.value}, RiskMult={combined_risk_mult}")
    
    return result


# =============================================================================
# LEGACY WRAPPERS
# =============================================================================

def analyze_tiempo(server_time_gmt=None, h4_candle="NEUTRAL", news_in_minutes=999, proposed_direction=None):
    """Wrapper legacy."""
    result = analyze_tiempo_institucional(server_time_gmt, "NEUTRAL", h4_candle, "NEUTRAL", False, news_in_minutes <= 30)
    return {
        "module": "FEAT_Tiempo",
        "status": "OPEN" if result["risk"]["combined_risk_multiplier"] > 0.3 else "CLOSED",
        "session": result["session"]["phase"],
        "instruction": "PROCEED_TO_MODULE_F" if result["guidance"]["can_trade"] else "WAIT",
        "risk_multiplier": result["risk"]["combined_risk_multiplier"],
        "advanced": result
    }


def generate_chrono_features(server_time_utc=None, news_upcoming=False, current_spread_pips=None):
    """Wrapper para ML features."""
    result = analyze_tiempo_institucional(server_time_utc, news_upcoming=news_upcoming)
    return {
        "module": "FEAT_Tiempo_ML",
        "ml_features": result.get("ml_features", {}),
        "session": result.get("session", {}),
        "weekly": result.get("weekly", {}),
        "risk": result.get("risk", {})
    }


# =============================================================================
# ASYNC MCP WRAPPERS
# =============================================================================

async def feat_check_tiempo(server_time_gmt=None, h4_candle="NEUTRAL", news_in_minutes=999):
    return analyze_tiempo(server_time_gmt, h4_candle, news_in_minutes)

async def feat_check_tiempo_advanced(server_time_utc=None, news_event_upcoming=False, h4_direction="NEUTRAL", h1_direction="NEUTRAL"):
    return analyze_tiempo_institucional(server_time_utc, "NEUTRAL", h4_direction, h1_direction, False, news_event_upcoming)

async def feat_generate_chrono_features(server_time_utc=None, news_upcoming=False, current_spread_pips=None):
    return generate_chrono_features(server_time_utc, news_upcoming, current_spread_pips)

async def feat_analyze_tiempo_institucional(
    server_time_utc=None,
    d1_direction="NEUTRAL",
    h4_direction="NEUTRAL",
    h1_direction="NEUTRAL",
    has_sweep=False,
    news_upcoming=False
):
    """MCP Tool: Análisis institucional completo."""
    return analyze_tiempo_institucional(server_time_utc, d1_direction, h4_direction, h1_direction, has_sweep, news_upcoming)
