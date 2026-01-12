"""
FEAT Module F: FORMA (Arquitectura Estructural)
================================================
Define el sesgo direccional (Bias) mediante el análisis de la microestructura fractal.

Conceptos Clave:
- Tendencia Alcista: HH (Higher High) + HL (Higher Low)
- Tendencia Bajista: LH (Lower High) + LL (Lower Low)
- BOS (Break of Structure): Ruptura con CUERPO de vela (no mecha)
- CHoCH (Change of Character): Primera señal de reversión
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger("FEAT.Forma")


class Trend(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    RANGING = "RANGING"


class StructuralEvent(Enum):
    BOS_BULLISH = "BOS_ALCISTA"      # Break of Structure alcista
    BOS_BEARISH = "BOS_BAJISTA"      # Break of Structure bajista
    CHOCH_BULLISH = "CHOCH_ALCISTA"  # Change of Character (giro a alcista)
    CHOCH_BEARISH = "CHOCH_BAJISTA"  # Change of Character (giro a bajista)
    NONE = "SIN_EVENTO"


class WyckoffPhase(Enum):
    ACCUMULATION_A = "ACUMULACION_FASE_A"
    ACCUMULATION_B = "ACUMULACION_FASE_B"
    MARKUP = "MARKUP"
    DISTRIBUTION_A = "DISTRIBUCION_FASE_A"
    DISTRIBUTION_B = "DISTRIBUCION_FASE_B"
    MARKDOWN = "MARKDOWN"
    UNKNOWN = "DESCONOCIDO"


def find_swing_points(candles: pd.DataFrame, lookback: int = 2) -> Dict[str, List]:
    """
    Identifica Swing Highs y Swing Lows usando el método de N velas.
    
    Un Swing High tiene un máximo mayor que las N velas a izquierda y derecha.
    Un Swing Low tiene un mínimo menor que las N velas a izquierda y derecha.
    
    Args:
        candles: DataFrame con columnas 'high', 'low', 'close', 'time'
        lookback: Número de velas a cada lado para confirmar
        
    Returns:
        Dict con listas de swing_highs y swing_lows
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


def detect_structure(swing_highs: List, swing_lows: List) -> Tuple[Trend, str]:
    """
    Analiza la secuencia de swing points para determinar la tendencia.
    
    Returns:
        Tuple[Trend, reason]
    """
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return (Trend.NEUTRAL, "Insuficientes puntos estructurales")
    
    # Get last 2 highs and lows
    last_high = swing_highs[-1]["price"]
    prev_high = swing_highs[-2]["price"]
    last_low = swing_lows[-1]["price"]
    prev_low = swing_lows[-2]["price"]
    
    # Tendencia Alcista: HH + HL
    if last_high > prev_high and last_low > prev_low:
        return (Trend.BULLISH, "HH + HL confirmado")
    
    # Tendencia Bajista: LH + LL
    if last_high < prev_high and last_low < prev_low:
        return (Trend.BEARISH, "LH + LL confirmado")
    
    # Rango/Consolidación
    if last_high < prev_high and last_low > prev_low:
        return (Trend.RANGING, "Compresión de rango detectada")
    
    return (Trend.NEUTRAL, "Estructura mixta")


def detect_bos_choch(
    candles: pd.DataFrame,
    swing_highs: List,
    swing_lows: List,
    current_trend: Trend
) -> Dict[str, Any]:
    """
    Detecta eventos BOS (Break of Structure) y CHoCH (Change of Character).
    
    REGLA CRÍTICA: Solo cuenta como BOS si el CUERPO de la vela cierra 
    más allá del nivel. Una mecha NO es un BOS, es una toma de liquidez.
    """
    if len(candles) < 3 or not swing_highs or not swing_lows:
        return {"event": StructuralEvent.NONE.value, "details": None}
    
    last_candle = candles.iloc[-1]
    last_close = last_candle["close"]
    last_open = last_candle["open"]
    
    # Get last structural levels
    last_swing_high = swing_highs[-1]["price"]
    last_swing_low = swing_lows[-1]["price"]
    
    # Detect BOS (continuación)
    if current_trend == Trend.BULLISH:
        # BOS alcista: Cierre POR ENCIMA del último swing high
        if last_close > last_swing_high and last_open < last_swing_high:
            return {
                "event": StructuralEvent.BOS_BULLISH.value,
                "level_broken": last_swing_high,
                "close_price": last_close,
                "details": "Cuerpo cerró por encima del máximo estructural"
            }
    
    elif current_trend == Trend.BEARISH:
        # BOS bajista: Cierre POR DEBAJO del último swing low
        if last_close < last_swing_low and last_open > last_swing_low:
            return {
                "event": StructuralEvent.BOS_BEARISH.value,
                "level_broken": last_swing_low,
                "close_price": last_close,
                "details": "Cuerpo cerró por debajo del mínimo estructural"
            }
    
    # Detect CHoCH (reversión)
    if current_trend == Trend.BULLISH:
        # CHoCH bajista: En tendencia alcista, se rompe el último HL
        second_last_low = swing_lows[-2]["price"] if len(swing_lows) >= 2 else swing_lows[-1]["price"]
        if last_close < second_last_low:
            return {
                "event": StructuralEvent.CHOCH_BEARISH.value,
                "level_broken": second_last_low,
                "close_price": last_close,
                "details": "Tendencia alcista rota - posible reversión bajista"
            }
    
    elif current_trend == Trend.BEARISH:
        # CHoCH alcista: En tendencia bajista, se rompe el último LH
        second_last_high = swing_highs[-2]["price"] if len(swing_highs) >= 2 else swing_highs[-1]["price"]
        if last_close > second_last_high:
            return {
                "event": StructuralEvent.CHOCH_BULLISH.value,
                "level_broken": second_last_high,
                "close_price": last_close,
                "details": "Tendencia bajista rota - posible reversión alcista"
            }
    
    return {"event": StructuralEvent.NONE.value, "details": None}


def detect_wyckoff_phase(
    trend: Trend,
    swing_highs: List,
    swing_lows: List,
    candles: pd.DataFrame
) -> WyckoffPhase:
    """
    Identifica la fase de Wyckoff actual.
    """
    if len(candles) < 20:
        return WyckoffPhase.UNKNOWN
    
    # Calculate range
    recent = candles.tail(20)
    range_high = recent["high"].max()
    range_low = recent["low"].min()
    range_size = range_high - range_low
    
    # Calculate average candle size
    avg_body = abs(recent["close"] - recent["open"]).mean()
    
    # Check for ranging (small bodies, contained range)
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        high_range = abs(swing_highs[-1]["price"] - swing_highs[-2]["price"])
        low_range = abs(swing_lows[-1]["price"] - swing_lows[-2]["price"])
        
        if high_range < range_size * 0.3 and low_range < range_size * 0.3:
            if trend == Trend.BULLISH or trend == Trend.NEUTRAL:
                return WyckoffPhase.ACCUMULATION_B
            else:
                return WyckoffPhase.DISTRIBUTION_B
    
    # Trending phases
    if trend == Trend.BULLISH:
        return WyckoffPhase.MARKUP
    elif trend == Trend.BEARISH:
        return WyckoffPhase.MARKDOWN
    
    return WyckoffPhase.UNKNOWN


def analyze_forma(
    h4_candles: List[Dict] = None,
    h1_candles: List[Dict] = None,
    m15_candles: List[Dict] = None,
    current_price: float = None
) -> Dict[str, Any]:
    """
    🏗️ FEAT MODULE F: Análisis de Estructura de Mercado.
    
    Analiza la estructura en 3 temporalidades (fractalidad):
    - Macro (H4): Tendencia dominante
    - Intermedio (H1): Estructura operativa
    - Micro (M15): Confirmaciones tempranas
    
    Returns:
        Dict con análisis completo de estructura
    """
    result = {
        "module": "FEAT_Forma",
        "status": "ANALYZED",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Convert to DataFrames
    tf_analysis = {}
    
    for tf_name, candles_list in [("H4", h4_candles), ("H1", h1_candles), ("M15", m15_candles)]:
        if not candles_list:
            tf_analysis[tf_name] = {"trend": Trend.NEUTRAL.value, "available": False}
            continue
        
        df = pd.DataFrame(candles_list)
        
        # Find swing points
        swings = find_swing_points(df)
        
        # Detect structure
        trend, reason = detect_structure(swings["swing_highs"], swings["swing_lows"])
        
        # Detect BOS/CHoCH
        event = detect_bos_choch(df, swings["swing_highs"], swings["swing_lows"], trend)
        
        # Wyckoff phase
        wyckoff = detect_wyckoff_phase(trend, swings["swing_highs"], swings["swing_lows"], df)
        
        tf_analysis[tf_name] = {
            "trend": trend.value,
            "trend_reason": reason,
            "last_event": event["event"],
            "event_details": event.get("details"),
            "wyckoff_phase": wyckoff.value,
            "structural_points": {
                "last_valid_high": swings["swing_highs"][-1]["price"] if swings["swing_highs"] else None,
                "last_valid_low": swings["swing_lows"][-1]["price"] if swings["swing_lows"] else None
            },
            "available": True
        }
    
    result["analysis"] = tf_analysis
    
    # Determine overall bias
    h4_trend = tf_analysis.get("H4", {}).get("trend", "NEUTRAL")
    h1_trend = tf_analysis.get("H1", {}).get("trend", "NEUTRAL")
    
    if h4_trend == "BULLISH" and h1_trend == "BULLISH":
        bias = "COMPRAS_FUERTES"
        instruction = "PROCEED_TO_MODULE_E"
    elif h4_trend == "BEARISH" and h1_trend == "BEARISH":
        bias = "VENTAS_FUERTES"
        instruction = "PROCEED_TO_MODULE_E"
    elif h4_trend == "BULLISH" and h1_trend == "BEARISH":
        bias = "RETROCESO_COMPLEJO"
        instruction = "WAIT_FOR_H1_ALIGNMENT"
    elif h4_trend == "BEARISH" and h1_trend == "BULLISH":
        bias = "RETROCESO_COMPLEJO"
        instruction = "WAIT_FOR_H1_ALIGNMENT"
    else:
        bias = "NEUTRAL"
        instruction = "WAIT_FOR_CLARITY"
    
    result["bias_conclusion"] = bias
    result["instruction"] = instruction
    
    logger.info(f"[FEAT-F] H4={h4_trend}, H1={h1_trend}, Bias={bias}")
    
    return result


# =============================================================================
# Async wrapper for MCP
# =============================================================================

async def feat_analyze_forma(
    h4_candles: List[Dict] = None,
    h1_candles: List[Dict] = None,
    m15_candles: List[Dict] = None,
    current_price: float = None
) -> Dict[str, Any]:
    """
    MCP Tool: FEAT Module F - Forma analysis.
    """
    return analyze_forma(h4_candles, h1_candles, m15_candles, current_price)
