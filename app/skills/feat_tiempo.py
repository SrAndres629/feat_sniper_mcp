"""
FEAT Module T: TIEMPO PROBABILÍSTICO (Chrono-Analyst Master)
==============================================================
Sistema de timing probabilístico para XAU/USD.

FILOSOFÍA: La IA SIEMPRE puede operar, pero ajusta su política según:
- Liquidez esperada (0.0-1.0)
- Volatilidad esperada (0.0-1.0)  
- Risk multiplier (0.1-2.0)
- Ciclo semanal (Induction/Direction/Expansion/Closing)

NO hay "Kill Switches" - solo PRIOR PROBABILITIES y RISK ADJUSTMENTS.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
from enum import Enum
import pytz

logger = logging.getLogger("FEAT.ChronoAnalyst")


# =============================================================================
# TIMEZONES
# =============================================================================

try:
    NY_TZ = pytz.timezone('America/New_York')
    BOLIVIA_TZ = pytz.timezone('America/La_Paz')  # Santa Cruz = La Paz timezone
except:
    NY_TZ = None
    BOLIVIA_TZ = None


# =============================================================================
# ENUMS
# =============================================================================

class WeeklyCycle(Enum):
    INDUCTION = "INDUCTION"      # Lunes: Falsos breakouts, crear liquidez
    DIRECTION = "DIRECTION"       # Martes: Se establece dirección real
    MIDWEEK = "MIDWEEK"          # Miércoles: Corrección o continuación
    EXPANSION = "EXPANSION"       # Jueves: Máxima expansión
    CLOSING = "CLOSING"          # Viernes: Cierre de ciclos


class SessionPhase(Enum):
    ASIA_OVERNIGHT = "ASIA_OVERNIGHT"
    ASIA_DEEP = "ASIA_DEEP"
    PRE_LONDON = "PRE_LONDON"
    LONDON_KILLZONE = "LONDON_KILLZONE"
    LONDON_CONTINUATION = "LONDON_CONTINUATION"
    PRE_NY = "PRE_NY"
    NY_OPEN_KILLZONE = "NY_OPEN_KILLZONE"
    NY_NEWS = "NY_NEWS"
    NY_LUNCH = "NY_LUNCH"
    NY_AFTERNOON = "NY_AFTERNOON"
    NY_CLOSE = "NY_CLOSE"
    POST_MARKET = "POST_MARKET"


# =============================================================================
# MATRIZ DE LIQUIDEZ: FEATURES NUMÉRICAS (0.0-1.0)
# =============================================================================

# (hora_inicio, hora_fin): {features}
LIQUIDITY_MATRIX = {
    (0, 1):   {"phase": "ASIA_OVERNIGHT", "liquidity": 0.15, "volatility": 0.10, "risk_mult": 0.25, "spread_risk": 0.80},
    (1, 3):   {"phase": "ASIA_DEEP", "liquidity": 0.10, "volatility": 0.08, "risk_mult": 0.20, "spread_risk": 0.90},
    (3, 4):   {"phase": "ASIA_LATE", "liquidity": 0.20, "volatility": 0.15, "risk_mult": 0.30, "spread_risk": 0.70},
    (4, 5):   {"phase": "PRE_LONDON", "liquidity": 0.35, "volatility": 0.30, "risk_mult": 0.50, "spread_risk": 0.50},
    (5, 6):   {"phase": "LONDON_EARLY", "liquidity": 0.55, "volatility": 0.50, "risk_mult": 0.70, "spread_risk": 0.35},
    (6, 8):   {"phase": "LONDON_KILLZONE", "liquidity": 0.80, "volatility": 0.75, "risk_mult": 1.00, "spread_risk": 0.20},
    (8, 9):   {"phase": "LONDON_CONTINUATION", "liquidity": 0.75, "volatility": 0.65, "risk_mult": 0.90, "spread_risk": 0.25},
    (9, 10):  {"phase": "PRE_NY", "liquidity": 0.50, "volatility": 0.45, "risk_mult": 0.60, "spread_risk": 0.40},
    (10, 11): {"phase": "NY_OPEN_KILLZONE", "liquidity": 0.95, "volatility": 0.90, "risk_mult": 1.00, "spread_risk": 0.15},
    (11, 12): {"phase": "NY_NEWS", "liquidity": 0.90, "volatility": 0.95, "risk_mult": 0.80, "spread_risk": 0.20},
    (12, 13): {"phase": "NY_LUNCH", "liquidity": 0.60, "volatility": 0.50, "risk_mult": 0.65, "spread_risk": 0.30},
    (13, 15): {"phase": "NY_AFTERNOON", "liquidity": 0.55, "volatility": 0.50, "risk_mult": 0.60, "spread_risk": 0.35},
    (15, 16): {"phase": "NY_CLOSE", "liquidity": 0.40, "volatility": 0.35, "risk_mult": 0.40, "spread_risk": 0.50},
    (16, 17): {"phase": "POST_MARKET_EARLY", "liquidity": 0.25, "volatility": 0.20, "risk_mult": 0.25, "spread_risk": 0.70},
    (17, 20): {"phase": "POST_MARKET", "liquidity": 0.15, "volatility": 0.12, "risk_mult": 0.20, "spread_risk": 0.80},
    (20, 24): {"phase": "ASIA_OVERNIGHT", "liquidity": 0.15, "volatility": 0.10, "risk_mult": 0.25, "spread_risk": 0.80},
}

# Cierres H4 importantes (hora NY) - momentos de posible barrido
H4_CLOSES = [0, 4, 8, 12, 16, 20]


# =============================================================================
# CICLO SEMANAL
# =============================================================================

WEEKLY_CYCLE_MAP = {
    0: {"phase": WeeklyCycle.INDUCTION, "risk_mult": 0.50, "bias": "TRAP_EXPECTED", 
        "note": "Lunes: Falsos breakouts. Movimientos suelen ser trampas."},
    1: {"phase": WeeklyCycle.DIRECTION, "risk_mult": 1.00, "bias": "DIRECTION_DISCOVERY",
        "note": "Martes: Se establece la dirección real de la semana."},
    2: {"phase": WeeklyCycle.MIDWEEK, "risk_mult": 0.85, "bias": "CONTINUATION_OR_CORRECTION",
        "note": "Miércoles: Posible corrección o continuación."},
    3: {"phase": WeeklyCycle.EXPANSION, "risk_mult": 1.20, "bias": "MAX_EXPANSION",
        "note": "Jueves: Día de máxima expansión. Riesgo agresivo."},
    4: {"phase": WeeklyCycle.CLOSING, "risk_mult": 0.40, "bias": "WEEKLY_CLOSE",
        "note": "Viernes: Cierre de ciclos. Solo operar NY AM (10-12)."},
    5: {"phase": WeeklyCycle.CLOSING, "risk_mult": 0.10, "bias": "WEEKEND",
        "note": "Sábado: Mercado cerrado."},
    6: {"phase": WeeklyCycle.CLOSING, "risk_mult": 0.10, "bias": "WEEKEND",
        "note": "Domingo: Pre-apertura Asia (tarde)."},
}


# =============================================================================
# FUNCIONES DE TIMEZONE
# =============================================================================

def get_ny_time(utc_time: datetime = None) -> datetime:
    """Convierte UTC a hora de Nueva York."""
    if utc_time is None:
        utc_time = datetime.now(timezone.utc)
    
    if NY_TZ:
        try:
            if utc_time.tzinfo is None:
                utc_time = utc_time.replace(tzinfo=timezone.utc)
            return utc_time.astimezone(NY_TZ)
        except:
            pass
    
    # Fallback: UTC-5 (EST)
    return utc_time - timedelta(hours=5)


def get_bolivia_time(utc_time: datetime = None) -> datetime:
    """Convierte UTC a hora de Bolivia (Santa Cruz = UTC-4)."""
    if utc_time is None:
        utc_time = datetime.now(timezone.utc)
    
    if BOLIVIA_TZ:
        try:
            if utc_time.tzinfo is None:
                utc_time = utc_time.replace(tzinfo=timezone.utc)
            return utc_time.astimezone(BOLIVIA_TZ)
        except:
            pass
    
    # Fallback: UTC-4
    return utc_time - timedelta(hours=4)


# =============================================================================
# FUNCIONES DE ANÁLISIS
# =============================================================================

def get_liquidity_features(ny_hour: int) -> Dict[str, Any]:
    """
    Obtiene features de liquidez para la hora NY dada.
    Retorna valores numéricos para ML.
    """
    for (start, end), features in LIQUIDITY_MATRIX.items():
        if start <= ny_hour < end:
            return features.copy()
    
    # Default: baja liquidez
    return {"phase": "UNKNOWN", "liquidity": 0.1, "volatility": 0.1, "risk_mult": 0.2, "spread_risk": 0.9}


def get_weekly_features(weekday: int) -> Dict[str, Any]:
    """
    Obtiene features del ciclo semanal.
    weekday: 0=Lunes, 1=Martes, ..., 6=Domingo
    """
    return WEEKLY_CYCLE_MAP.get(weekday, WEEKLY_CYCLE_MAP[6]).copy()


def calculate_h4_proximity(ny_hour: int, ny_minute: int) -> Dict[str, Any]:
    """
    Calcula proximidad al cierre H4 más cercano.
    """
    # Encontrar cierre H4 más cercano
    current_minutes = ny_hour * 60 + ny_minute
    
    min_distance = 240  # 4 horas máximo
    nearest_close = 0
    
    for h4_close in H4_CLOSES:
        close_minutes = h4_close * 60
        
        # Distancia hacia adelante
        dist_forward = (close_minutes - current_minutes) % 1440
        # Distancia hacia atrás
        dist_backward = (current_minutes - close_minutes) % 1440
        
        if dist_forward < min_distance:
            min_distance = dist_forward
            nearest_close = h4_close
        if dist_backward < min_distance:
            min_distance = dist_backward
            nearest_close = h4_close
    
    return {
        "nearest_h4_close": nearest_close,
        "minutes_to_h4_close": min_distance,
        "is_near_h4_close": min_distance <= 15,
        "h4_close_warning": min_distance <= 5
    }


def calculate_combined_risk_multiplier(
    liquidity_features: Dict,
    weekly_features: Dict,
    h4_proximity: Dict,
    news_upcoming: bool = False,
    current_spread_pips: float = None,
    avg_spread_pips: float = None
) -> Dict[str, Any]:
    """
    Calcula el multiplicador de riesgo combinado.
    Este es el output más importante para el ML y la ejecución.
    """
    # Base: multiplicador de la hora
    hourly_mult = liquidity_features.get("risk_mult", 0.5)
    
    # Ajuste semanal
    weekly_mult = weekly_features.get("risk_mult", 1.0)
    
    # Penalización por proximidad H4
    h4_penalty = 0.7 if h4_proximity.get("is_near_h4_close") else 1.0
    
    # Penalización por noticias
    news_penalty = 0.5 if news_upcoming else 1.0
    
    # Penalización por spread alto
    spread_penalty = 1.0
    if current_spread_pips and avg_spread_pips and avg_spread_pips > 0:
        spread_ratio = current_spread_pips / avg_spread_pips
        if spread_ratio > 2.0:
            spread_penalty = 0.3  # Spread muy alto
        elif spread_ratio > 1.5:
            spread_penalty = 0.6  # Spread elevado
    
    # Combinar todos los factores
    combined = hourly_mult * weekly_mult * h4_penalty * news_penalty * spread_penalty
    
    # Limitar entre 0.1 y 2.0
    final_mult = max(0.1, min(2.0, combined))
    
    return {
        "hourly_mult": hourly_mult,
        "weekly_mult": weekly_mult,
        "h4_penalty": h4_penalty,
        "news_penalty": news_penalty,
        "spread_penalty": spread_penalty,
        "combined_risk_multiplier": round(final_mult, 3),
        "recommended_size": "MICRO" if final_mult < 0.3 else ("SMALL" if final_mult < 0.5 else ("NORMAL" if final_mult < 0.8 else "FULL"))
    }


# =============================================================================
# FUNCIÓN PRINCIPAL: GENERATE ML FEATURES
# =============================================================================

def generate_chrono_features(
    server_time_utc: str = None,
    news_upcoming: bool = False,
    current_spread_pips: float = None,
    avg_spread_pips: float = None
) -> Dict[str, Any]:
    """
    🕐 FEAT CHRONO-ANALYST: Genera features numéricas para ML.
    
    CRÍTICO: Este output es para alimentar a la red neuronal.
    NO hay "gates" o "kill switches" - solo PROBABILIDADES y MULTIPLICADORES.
    
    Returns:
        Dict con features numéricas listas para ML
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
    ny_time = get_ny_time(utc_time)
    bolivia_time = get_bolivia_time(utc_time)
    
    ny_hour = ny_time.hour
    ny_minute = ny_time.minute
    weekday = ny_time.weekday()  # 0=Monday
    
    # Get features
    liquidity_features = get_liquidity_features(ny_hour)
    weekly_features = get_weekly_features(weekday)
    h4_proximity = calculate_h4_proximity(ny_hour, ny_minute)
    
    # Calculate combined risk
    risk_analysis = calculate_combined_risk_multiplier(
        liquidity_features, weekly_features, h4_proximity,
        news_upcoming, current_spread_pips, avg_spread_pips
    )
    
    # Build ML feature vector
    ml_features = {
        # Time features (normalized)
        "hour_ny": ny_hour,
        "hour_ny_normalized": ny_hour / 23.0,
        "minute_normalized": ny_minute / 59.0,
        "weekday": weekday,
        "weekday_normalized": weekday / 6.0,
        
        # Session flags (one-hot)
        "is_asia": 1 if ny_hour in range(0, 5) or ny_hour in range(19, 24) else 0,
        "is_london": 1 if ny_hour in range(5, 12) else 0,
        "is_ny": 1 if ny_hour in range(10, 17) else 0,
        "is_overlap": 1 if ny_hour in range(10, 12) else 0,
        
        # Killzone flags
        "is_london_killzone": 1 if ny_hour in range(6, 9) else 0,
        "is_ny_killzone": 1 if ny_hour in range(10, 12) else 0,
        "is_any_killzone": 1 if ny_hour in range(6, 9) or ny_hour in range(10, 12) else 0,
        
        # Continuous features (0-1)
        "liquidity_expected": liquidity_features["liquidity"],
        "volatility_expected": liquidity_features["volatility"],
        "spread_risk": liquidity_features["spread_risk"],
        
        # Weekly cycle
        "weekly_phase": weekly_features["phase"].value,
        "weekly_risk_mult": weekly_features["risk_mult"],
        "is_induction_day": 1 if weekday == 0 else 0,
        "is_expansion_day": 1 if weekday == 3 else 0,
        "is_closing_day": 1 if weekday == 4 else 0,
        
        # H4 proximity
        "minutes_to_h4_close": h4_proximity["minutes_to_h4_close"],
        "is_near_h4_close": 1 if h4_proximity["is_near_h4_close"] else 0,
        
        # Combined risk
        "combined_risk_multiplier": risk_analysis["combined_risk_multiplier"],
        
        # News
        "news_upcoming": 1 if news_upcoming else 0
    }
    
    # Build human-readable result
    result = {
        "module": "FEAT_CHRONO_ANALYST",
        "timestamp_utc": utc_time.isoformat() if hasattr(utc_time, 'isoformat') else str(utc_time),
        
        # Human readable times
        "time": {
            "utc": utc_time.strftime("%H:%M") if hasattr(utc_time, 'strftime') else str(utc_time),
            "ny": ny_time.strftime("%H:%M") if hasattr(ny_time, 'strftime') else str(ny_time),
            "bolivia": bolivia_time.strftime("%H:%M") if hasattr(bolivia_time, 'strftime') else str(bolivia_time),
            "day_of_week": ny_time.strftime("%A") if hasattr(ny_time, 'strftime') else str(weekday)
        },
        
        # Session analysis
        "session": {
            "phase": liquidity_features["phase"],
            "liquidity_expected": liquidity_features["liquidity"],
            "volatility_expected": liquidity_features["volatility"],
            "spread_risk": liquidity_features["spread_risk"]
        },
        
        # Weekly cycle
        "weekly": {
            "phase": weekly_features["phase"].value,
            "bias": weekly_features["bias"],
            "note": weekly_features["note"]
        },
        
        # H4 analysis
        "h4": h4_proximity,
        
        # Risk analysis
        "risk": risk_analysis,
        
        # ML features (for neural network)
        "ml_features": ml_features,
        
        # Action guidance (soft, not hard rules)
        "guidance": {
            "can_trade": True,  # SIEMPRE puede tradear
            "recommended_size": risk_analysis["recommended_size"],
            "risk_multiplier": risk_analysis["combined_risk_multiplier"],
            "cautions": []
        }
    }
    
    # Add cautions (advisory, not blocking)
    if liquidity_features["liquidity"] < 0.3:
        result["guidance"]["cautions"].append("⚠️ Baja liquidez: considera reducir tamaño")
    if h4_proximity["is_near_h4_close"]:
        result["guidance"]["cautions"].append("🕐 Cerca de cierre H4: posible barrido")
    if weekday == 0 and ny_hour < 12:
        result["guidance"]["cautions"].append("📅 Lunes AM: alta probabilidad de falsos breakouts")
    if news_upcoming:
        result["guidance"]["cautions"].append("📰 Noticia próxima: aumentar stop o reducir tamaño")
    
    logger.info(f"[FEAT-T] NY={ny_time.strftime('%H:%M')}, Phase={liquidity_features['phase']}, RiskMult={risk_analysis['combined_risk_multiplier']}")
    
    return result


# =============================================================================
# FUNCIONES LEGACY (retrocompatibilidad)
# =============================================================================

def analyze_tiempo(server_time_gmt: str = None, h4_candle: str = "NEUTRAL", news_in_minutes: int = 999, proposed_direction: str = None) -> Dict[str, Any]:
    """Wrapper de retrocompatibilidad."""
    result = generate_chrono_features(server_time_gmt, news_in_minutes <= 30)
    
    # Formato legacy
    return {
        "module": "FEAT_Tiempo",
        "status": "OPEN",  # Siempre abierto (probabilístico)
        "session": result["session"]["phase"],
        "instruction": "PROCEED_TO_MODULE_F",  # Siempre procede
        "analysis": result["session"],
        "risk_multiplier": result["risk"]["combined_risk_multiplier"],
        "advanced": result
    }


def analyze_tiempo_advanced(server_time_utc: str = None, news_event_upcoming: bool = False, h4_direction: str = "NEUTRAL", h1_direction: str = "NEUTRAL") -> Dict[str, Any]:
    """Wrapper avanzado."""
    return generate_chrono_features(server_time_utc, news_event_upcoming)


# =============================================================================
# ASYNC MCP WRAPPERS
# =============================================================================

async def feat_check_tiempo(server_time_gmt: str = None, h4_candle: str = "NEUTRAL", news_in_minutes: int = 999) -> Dict[str, Any]:
    """MCP Tool: Tiempo legacy."""
    return analyze_tiempo(server_time_gmt, h4_candle, news_in_minutes)


async def feat_check_tiempo_advanced(server_time_utc: str = None, news_event_upcoming: bool = False, h4_direction: str = "NEUTRAL", h1_direction: str = "NEUTRAL") -> Dict[str, Any]:
    """MCP Tool: Tiempo advanced."""
    return analyze_tiempo_advanced(server_time_utc, news_event_upcoming, h4_direction, h1_direction)


async def feat_generate_chrono_features(server_time_utc: str = None, news_upcoming: bool = False, current_spread_pips: float = None) -> Dict[str, Any]:
    """MCP Tool: Generate ML-ready chrono features."""
    return generate_chrono_features(server_time_utc, news_upcoming, current_spread_pips)
