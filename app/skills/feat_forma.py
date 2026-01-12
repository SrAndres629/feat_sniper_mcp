"""
FEAT Module F: FORMA (Arquitectura Estructural) - CHRONO-AWARE
===============================================================
Define el sesgo direccional (Bias) mediante anlisis fractal CON CONTEXTO TEMPORAL.

Conceptos Clave SMC:
- Tendencia Alcista: HH (Higher High) + HL (Higher Low)
- Tendencia Bajista: LH (Lower High) + LL (Lower Low)
- BOS (Break of Structure): Ruptura con CUERPO de vela (no mecha)
- CHoCH (Change of Character): Primera seal de reversin

INTEGRACIN TEMPORAL:
- Valida BOS/CHoCH contra ciclo semanal (Induction vs Direction)
- Detecta trampas de Lunes (falsos breakouts)
- Ajusta confianza segn alineacin con Kill Zones
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger("FEAT.FormaChronoAware")


# =============================================================================
# ENUMS
# =============================================================================

class Trend(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    RANGING = "RANGING"


class StructuralEvent(Enum):
    BOS_BULLISH = "BOS_ALCISTA"
    BOS_BEARISH = "BOS_BAJISTA"
    CHOCH_BULLISH = "CHOCH_ALCISTA"
    CHOCH_BEARISH = "CHOCH_BAJISTA"
    SWEEP_BULLISH = "SWEEP_ALCISTA"   # Mecha sin cierre (liquidacin)
    SWEEP_BEARISH = "SWEEP_BAJISTA"
    NONE = "SIN_EVENTO"


class WyckoffPhase(Enum):
    ACCUMULATION_A = "ACUMULACION_FASE_A"
    ACCUMULATION_B = "ACUMULACION_FASE_B"
    SPRING = "SPRING"
    MARKUP = "MARKUP"
    DISTRIBUTION_A = "DISTRIBUCION_FASE_A"
    DISTRIBUTION_B = "DISTRIBUCION_FASE_B"
    UPTHRUST = "UPTHRUST"
    MARKDOWN = "MARKDOWN"
    UNKNOWN = "DESCONOCIDO"


# =============================================================================
# SWING POINT DETECTION
# =============================================================================

def find_swing_points(candles: pd.DataFrame, lookback: int = 2) -> Dict[str, List]:
    """
    Identifica Swing Highs y Swing Lows usando mtodo N-velas.
    """
    swing_highs = []
    swing_lows = []
    
    if len(candles) < lookback * 2 + 1:
        return {"swing_highs": [], "swing_lows": []}
    
    for i in range(lookback, len(candles) - lookback):
        current_high = candles.iloc[i]["high"]
        current_low = candles.iloc[i]["low"]
        
        # Check Swing High
        is_swing_high = True
        for j in range(1, lookback + 1):
            if candles.iloc[i - j]["high"] >= current_high or \
               candles.iloc[i + j]["high"] >= current_high:
                is_swing_high = False
                break
        
        if is_swing_high:
            swing_highs.append({
                "index": i,
                "price": current_high,
                "time": candles.iloc[i].get("time", str(i))
            })
        
        # Check Swing Low
        is_swing_low = True
        for j in range(1, lookback + 1):
            if candles.iloc[i - j]["low"] <= current_low or \
               candles.iloc[i + j]["low"] <= current_low:
                is_swing_low = False
                break
        
        if is_swing_low:
            swing_lows.append({
                "index": i,
                "price": current_low,
                "time": candles.iloc[i].get("time", str(i))
            })
    
    return {"swing_highs": swing_highs, "swing_lows": swing_lows}


def detect_structure(swing_highs: List, swing_lows: List) -> Tuple[Trend, str, float]:
    """
    Analiza secuencia de swing points para determinar tendencia.
    
    Returns:
        Tuple[Trend, reason, confidence_score]
    """
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return (Trend.NEUTRAL, "Insuficientes puntos estructurales", 0.3)
    
    last_high = swing_highs[-1]["price"]
    prev_high = swing_highs[-2]["price"]
    last_low = swing_lows[-1]["price"]
    prev_low = swing_lows[-2]["price"]
    
    # Tendencia Alcista: HH + HL
    if last_high > prev_high and last_low > prev_low:
        confidence = 0.9 if (last_high - prev_high) > (prev_high - swing_highs[-3]["price"] if len(swing_highs) > 2 else 0) else 0.75
        return (Trend.BULLISH, "HH + HL confirmado", confidence)
    
    # Tendencia Bajista: LH + LL
    if last_high < prev_high and last_low < prev_low:
        confidence = 0.9 if (prev_low - last_low) > (swing_lows[-3]["price"] - prev_low if len(swing_lows) > 2 else 0) else 0.75
        return (Trend.BEARISH, "LH + LL confirmado", confidence)
    
    # Rango/Consolidacin
    if last_high < prev_high and last_low > prev_low:
        return (Trend.RANGING, "Compresin de rango detectada", 0.6)
    
    # Expansin (breakout potencial)
    if last_high > prev_high and last_low < prev_low:
        return (Trend.NEUTRAL, "Expansin de volatilidad", 0.5)
    
    return (Trend.NEUTRAL, "Estructura mixta", 0.4)


# =============================================================================
# BOS/CHoCH DETECTION WITH CHRONO CONTEXT
# =============================================================================

def detect_bos_choch(
    candles: pd.DataFrame,
    swing_highs: List,
    swing_lows: List,
    current_trend: Trend,
    weekly_phase: str = None
) -> Dict[str, Any]:
    """
    Detecta BOS/CHoCH con validacin temporal.
    
    REGLA CRTICA: 
    - Solo cuenta como BOS si el CUERPO cierra ms all del nivel
    - Una MECHA es un SWEEP (toma de liquidez), no un BOS
    - En INDUCTION (Lunes), los BOS tienen menor confianza
    """
    if len(candles) < 3 or not swing_highs or not swing_lows:
        return {"event": StructuralEvent.NONE.value, "confidence": 0, "details": None}
    
    last_candle = candles.iloc[-1]
    last_close = last_candle["close"]
    last_open = last_candle["open"]
    last_high = last_candle["high"]
    last_low = last_candle["low"]
    
    last_swing_high = swing_highs[-1]["price"]
    last_swing_low = swing_lows[-1]["price"]
    
    # Confidence modifier based on weekly phase
    chrono_modifier = 1.0
    if weekly_phase == "INDUCTION":
        chrono_modifier = 0.5  # Lunes: alto riesgo de trampas
    elif weekly_phase == "DIRECTION":
        chrono_modifier = 1.2  # Martes: mxima confianza
    elif weekly_phase == "EXPANSION":
        chrono_modifier = 1.1  # Jueves: buena confianza
    
    result = {"event": StructuralEvent.NONE.value, "confidence": 0, "details": None, "chrono_modifier": chrono_modifier}
    
    # ===== BOS DETECTION =====
    
    # BOS Alcista: CUERPO cierra encima del swing high
    if last_close > last_swing_high:
        if last_open < last_swing_high:  # Rompimiento limpio
            base_confidence = 0.85
        else:
            base_confidence = 0.70  # Ya abri arriba
        
        result = {
            "event": StructuralEvent.BOS_BULLISH.value,
            "level_broken": last_swing_high,
            "close_price": last_close,
            "confidence": min(1.0, base_confidence * chrono_modifier),
            "details": "Cuerpo cerr encima del mximo estructural",
            "chrono_warning": " BOS en INDUCTION - posible trampa" if weekly_phase == "INDUCTION" else None
        }
        return result
    
    # BOS Bajista: CUERPO cierra debajo del swing low
    if last_close < last_swing_low:
        if last_open > last_swing_low:
            base_confidence = 0.85
        else:
            base_confidence = 0.70
        
        result = {
            "event": StructuralEvent.BOS_BEARISH.value,
            "level_broken": last_swing_low,
            "close_price": last_close,
            "confidence": min(1.0, base_confidence * chrono_modifier),
            "details": "Cuerpo cerr debajo del mnimo estructural",
            "chrono_warning": " BOS en INDUCTION - posible trampa" if weekly_phase == "INDUCTION" else None
        }
        return result
    
    # ===== SWEEP DETECTION (Mecha sin cierre) =====
    
    # Sweep Alcista: Mecha toc pero cuerpo cerr abajo
    if last_high > last_swing_high and last_close < last_swing_high:
        result = {
            "event": StructuralEvent.SWEEP_BULLISH.value,
            "level_swept": last_swing_high,
            "wick_high": last_high,
            "close_price": last_close,
            "confidence": 0.60 * chrono_modifier,
            "details": "Liquidacin de stops - mecha sin cierre estructural"
        }
        return result
    
    # Sweep Bajista
    if last_low < last_swing_low and last_close > last_swing_low:
        result = {
            "event": StructuralEvent.SWEEP_BEARISH.value,
            "level_swept": last_swing_low,
            "wick_low": last_low,
            "close_price": last_close,
            "confidence": 0.60 * chrono_modifier,
            "details": "Liquidacin de stops - mecha sin cierre estructural"
        }
        return result
    
    # ===== CHoCH DETECTION (Cambio de carcter) =====
    
    if current_trend == Trend.BULLISH and len(swing_lows) >= 2:
        second_last_low = swing_lows[-2]["price"]
        if last_close < second_last_low:
            result = {
                "event": StructuralEvent.CHOCH_BEARISH.value,
                "level_broken": second_last_low,
                "close_price": last_close,
                "confidence": 0.80 * chrono_modifier,
                "details": "Tendencia alcista ROTA - posible reversin bajista"
            }
            return result
    
    if current_trend == Trend.BEARISH and len(swing_highs) >= 2:
        second_last_high = swing_highs[-2]["price"]
        if last_close > second_last_high:
            result = {
                "event": StructuralEvent.CHOCH_BULLISH.value,
                "level_broken": second_last_high,
                "close_price": last_close,
                "confidence": 0.80 * chrono_modifier,
                "details": "Tendencia bajista ROTA - posible reversin alcista"
            }
            return result
    
    return result


# =============================================================================
# WYCKOFF PHASE DETECTION
# =============================================================================

def detect_wyckoff_phase(
    trend: Trend,
    swing_highs: List,
    swing_lows: List,
    candles: pd.DataFrame
) -> Tuple[WyckoffPhase, float]:
    """
    Identifica fase de Wyckoff con score de confianza.
    """
    if len(candles) < 20:
        return (WyckoffPhase.UNKNOWN, 0.3)
    
    recent = candles.tail(20)
    range_high = recent["high"].max()
    range_low = recent["low"].min()
    range_size = range_high - range_low
    
    # Calculate body/range ratio
    avg_body = abs(recent["close"] - recent["open"]).mean()
    body_ratio = avg_body / range_size if range_size > 0 else 0
    
    # Check for ranging
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        high_range = abs(swing_highs[-1]["price"] - swing_highs[-2]["price"])
        low_range = abs(swing_lows[-1]["price"] - swing_lows[-2]["price"])
        
        # Tight range = Accumulation/Distribution
        if high_range < range_size * 0.3 and low_range < range_size * 0.3:
            if trend in [Trend.BULLISH, Trend.NEUTRAL]:
                return (WyckoffPhase.ACCUMULATION_B, 0.70)
            else:
                return (WyckoffPhase.DISTRIBUTION_B, 0.70)
        
        # Spring detection (false break below)
        last_candle = candles.iloc[-1]
        if last_candle["low"] < swing_lows[-1]["price"] and \
           last_candle["close"] > swing_lows[-1]["price"]:
            return (WyckoffPhase.SPRING, 0.85)
        
        # Upthrust detection (false break above)
        if last_candle["high"] > swing_highs[-1]["price"] and \
           last_candle["close"] < swing_highs[-1]["price"]:
            return (WyckoffPhase.UPTHRUST, 0.85)
    
    # Trending phases
    if trend == Trend.BULLISH:
        return (WyckoffPhase.MARKUP, 0.80)
    elif trend == Trend.BEARISH:
        return (WyckoffPhase.MARKDOWN, 0.80)
    
    return (WyckoffPhase.UNKNOWN, 0.40)


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_forma(
    h4_candles: List[Dict] = None,
    h1_candles: List[Dict] = None,
    m15_candles: List[Dict] = None,
    current_price: float = None,
    chrono_features: Dict = None
) -> Dict[str, Any]:
    """
     FEAT MODULE F: Anlisis de Estructura de Mercado (Chrono-Aware).
    
    Analiza en 3 temporalidades (fractalidad):
    - Macro (H4): Tendencia dominante
    - Intermedio (H1): Estructura operativa
    - Micro (M15): Confirmaciones tempranas
    
    INTEGRA contexto temporal de Module T para validar seales.
    """
    result = {
        "module": "FEAT_Forma_ChronoAware",
        "status": "ANALYZED",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Extract chrono context
    weekly_phase = None
    chrono_risk_mult = 1.0
    if chrono_features:
        weekly_phase = chrono_features.get("weekly", {}).get("phase")
        chrono_risk_mult = chrono_features.get("risk", {}).get("combined_risk_multiplier", 1.0)
    
    # Analyze each timeframe
    tf_analysis = {}
    
    for tf_name, candles_list, lookback in [("H4", h4_candles, 3), ("H1", h1_candles, 2), ("M15", m15_candles, 2)]:
        if not candles_list:
            tf_analysis[tf_name] = {"trend": Trend.NEUTRAL.value, "available": False, "confidence": 0}
            continue
        
        df = pd.DataFrame(candles_list)
        
        # Find swing points
        swings = find_swing_points(df, lookback=lookback)
        
        # Detect structure with confidence
        trend, reason, confidence = detect_structure(swings["swing_highs"], swings["swing_lows"])
        
        # Detect BOS/CHoCH with chrono context
        event = detect_bos_choch(df, swings["swing_highs"], swings["swing_lows"], trend, weekly_phase)
        
        # Wyckoff phase
        wyckoff, wyckoff_confidence = detect_wyckoff_phase(trend, swings["swing_highs"], swings["swing_lows"], df)
        
        tf_analysis[tf_name] = {
            "trend": trend.value,
            "trend_reason": reason,
            "trend_confidence": round(confidence, 2),
            "last_event": event["event"],
            "event_confidence": round(event.get("confidence", 0), 2),
            "event_details": event.get("details"),
            "chrono_warning": event.get("chrono_warning"),
            "wyckoff_phase": wyckoff.value,
            "wyckoff_confidence": round(wyckoff_confidence, 2),
            "structural_points": {
                "last_valid_high": swings["swing_highs"][-1]["price"] if swings["swing_highs"] else None,
                "last_valid_low": swings["swing_lows"][-1]["price"] if swings["swing_lows"] else None
            },
            "swing_count": {"highs": len(swings["swing_highs"]), "lows": len(swings["swing_lows"])},
            "available": True
        }
    
    result["analysis"] = tf_analysis
    
    # Determine overall bias with confidence
    h4_trend = tf_analysis.get("H4", {}).get("trend", "NEUTRAL")
    h1_trend = tf_analysis.get("H1", {}).get("trend", "NEUTRAL")
    h4_conf = tf_analysis.get("H4", {}).get("trend_confidence", 0.5)
    h1_conf = tf_analysis.get("H1", {}).get("trend_confidence", 0.5)
    
    # Calculate alignment score
    if h4_trend == h1_trend and h4_trend != "NEUTRAL":
        alignment_score = (h4_conf * 0.6 + h1_conf * 0.4) * chrono_risk_mult
        if h4_trend == "BULLISH":
            bias = "COMPRAS_FUERTES"
        else:
            bias = "VENTAS_FUERTES"
        instruction = "PROCEED_TO_MODULE_E"
    elif h4_trend != "NEUTRAL" and h1_trend == "NEUTRAL":
        alignment_score = h4_conf * 0.5 * chrono_risk_mult
        bias = "ESPERANDO_CONFIRMACION_H1"
        instruction = "WAIT_FOR_H1_STRUCTURE"
    elif h4_trend != h1_trend and h4_trend != "NEUTRAL" and h1_trend != "NEUTRAL":
        alignment_score = min(h4_conf, h1_conf) * 0.4 * chrono_risk_mult
        bias = "RETROCESO_COMPLEJO"
        instruction = "WAIT_FOR_H1_ALIGNMENT"
    else:
        alignment_score = 0.3 * chrono_risk_mult
        bias = "NEUTRAL"
        instruction = "WAIT_FOR_CLARITY"
    
    result["bias_conclusion"] = bias
    result["instruction"] = instruction
    result["alignment_score"] = round(min(1.0, alignment_score), 2)
    
    # ML Features (numeric, ready for neural network)
    result["ml_features"] = {
        # Trend encoding
        "h4_trend_bullish": 1 if h4_trend == "BULLISH" else 0,
        "h4_trend_bearish": 1 if h4_trend == "BEARISH" else 0,
        "h1_trend_bullish": 1 if h1_trend == "BULLISH" else 0,
        "h1_trend_bearish": 1 if h1_trend == "BEARISH" else 0,
        
        # Alignment
        "htf_aligned": 1 if h4_trend == h1_trend and h4_trend != "NEUTRAL" else 0,
        "alignment_score": result["alignment_score"],
        
        # Confidence scores
        "h4_confidence": h4_conf,
        "h1_confidence": h1_conf,
        
        # Event flags
        "has_bos": 1 if "BOS" in tf_analysis.get("H1", {}).get("last_event", "") else 0,
        "has_choch": 1 if "CHOCH" in tf_analysis.get("H1", {}).get("last_event", "") else 0,
        "has_sweep": 1 if "SWEEP" in tf_analysis.get("H1", {}).get("last_event", "") else 0,
        
        # Chrono context
        "chrono_risk_mult": chrono_risk_mult,
        "is_induction_trap_risk": 1 if weekly_phase == "INDUCTION" else 0
    }
    
    # Guidance (soft, not blocking)
    result["guidance"] = {
        "can_proceed": alignment_score > 0.5,
        "confidence": "HIGH" if alignment_score > 0.7 else ("MEDIUM" if alignment_score > 0.5 else "LOW"),
        "cautions": []
    }
    
    if weekly_phase == "INDUCTION":
        result["guidance"]["cautions"].append(" LUNES: Alto riesgo de falsos breakouts")
    if h4_trend != h1_trend:
        result["guidance"]["cautions"].append(" H4/H1 no alineados - esperar confirmacin")
    if any("SWEEP" in tf_analysis.get(tf, {}).get("last_event", "") for tf in ["H4", "H1"]):
        result["guidance"]["cautions"].append(" Sweep detectado - posible reversin")
    
    logger.info(f"[FEAT-F] H4={h4_trend}({h4_conf:.2f}), H1={h1_trend}({h1_conf:.2f}), Align={alignment_score:.2f}")
    
    return result


def generate_structure_features(
    h4_candles: List[Dict] = None,
    h1_candles: List[Dict] = None,
    m15_candles: List[Dict] = None,
    chrono_features: Dict = None
) -> Dict[str, Any]:
    """
    Genera vector de features para ML desde anlisis estructural.
    """
    analysis = analyze_forma(h4_candles, h1_candles, m15_candles, None, chrono_features)
    return {
        "module": "FEAT_Forma_ML",
        "features": analysis.get("ml_features", {}),
        "bias": analysis.get("bias_conclusion"),
        "alignment_score": analysis.get("alignment_score", 0)
    }


# =============================================================================
# ASYNC MCP WRAPPERS
# =============================================================================

async def feat_analyze_forma(
    h4_candles: List[Dict] = None,
    h1_candles: List[Dict] = None,
    m15_candles: List[Dict] = None,
    current_price: float = None
) -> Dict[str, Any]:
    """MCP Tool: FEAT Module F - Forma (legacy)."""
    return analyze_forma(h4_candles, h1_candles, m15_candles, current_price)


async def feat_analyze_forma_advanced(
    h4_candles: List[Dict] = None,
    h1_candles: List[Dict] = None,
    m15_candles: List[Dict] = None,
    chrono_features: Dict = None
) -> Dict[str, Any]:
    """MCP Tool: FEAT Module F - Forma (chrono-aware)."""
    return analyze_forma(h4_candles, h1_candles, m15_candles, None, chrono_features)


async def feat_generate_structure_features(
    h4_candles: List[Dict] = None,
    h1_candles: List[Dict] = None,
    m15_candles: List[Dict] = None,
    chrono_features: Dict = None
) -> Dict[str, Any]:
    """MCP Tool: Generate ML-ready structure features."""
    return generate_structure_features(h4_candles, h1_candles, m15_candles, chrono_features)
