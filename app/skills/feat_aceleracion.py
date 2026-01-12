"""
FEAT Module A: ACELERACIÓN (Vector de Intención)
=================================================
Mide la velocidad y el volumen para descartar manipulaciones (Fakeouts).

Conceptos Clave:
- Regla de Volumen/Cuerpo: Vela de intención = cuerpo grande, mechas pequeñas
- Fakeout Detection: Mecha gigante que cierra dentro del rango = trampa
- ATR Filter: Si volatilidad < 50% del promedio, mercado dormido

Este módulo es el GATILLO FINAL antes de ejecutar.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger("FEAT.Aceleracion")


class MomentumType(Enum):
    IMPULSIVE = "IMPULSIVO"           # Cuerpo grande, alta probabilidad
    ABSORBING = "ABSORBIENDO"         # Alto volumen, cuerpo pequeño = frenado
    WEAK = "DEBIL"                    # Cuerpo pequeño, bajo volumen
    FAKEOUT = "FAKEOUT"               # Trampa detectada


class IntentionCandle(Enum):
    STRONG_BULLISH = "VELA_ALCISTA_FUERTE"
    STRONG_BEARISH = "VELA_BAJISTA_FUERTE"
    ENGULFING_BULLISH = "ENVOLVENTE_ALCISTA"
    ENGULFING_BEARISH = "ENVOLVENTE_BAJISTA"
    PINBAR_BULLISH = "PINBAR_ALCISTA"
    PINBAR_BEARISH = "PINBAR_BAJISTA"
    DOJI = "DOJI"
    NEUTRAL = "NEUTRAL"


def calculate_body_wick_ratio(candle: Dict) -> Dict[str, float]:
    """
    Calcula el ratio Cuerpo/Mecha de una vela.
    
    Velas de intención tienen > 70% cuerpo.
    Velas de absorción tienen < 30% cuerpo (mucha mecha).
    """
    high = candle["high"]
    low = candle["low"]
    open_ = candle["open"]
    close = candle["close"]
    
    total_range = high - low
    if total_range == 0:
        return {"body_ratio": 0, "upper_wick": 0, "lower_wick": 0}
    
    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    
    return {
        "body_ratio": (body / total_range) * 100,
        "upper_wick_ratio": (upper_wick / total_range) * 100,
        "lower_wick_ratio": (lower_wick / total_range) * 100,
        "body_size": body,
        "direction": "BULLISH" if close > open_ else "BEARISH" if close < open_ else "NEUTRAL"
    }


def classify_candle(candle: Dict, prev_candle: Dict = None) -> IntentionCandle:
    """
    Clasifica una vela según su estructura de intención.
    """
    ratios = calculate_body_wick_ratio(candle)
    body_ratio = ratios["body_ratio"]
    direction = ratios["direction"]
    
    # Strong candle (>70% body)
    if body_ratio > 70:
        if direction == "BULLISH":
            return IntentionCandle.STRONG_BULLISH
        elif direction == "BEARISH":
            return IntentionCandle.STRONG_BEARISH
    
    # Pinbar (long wick, small body)
    if body_ratio < 30:
        if ratios["lower_wick_ratio"] > 60:
            return IntentionCandle.PINBAR_BULLISH  # Hammer
        elif ratios["upper_wick_ratio"] > 60:
            return IntentionCandle.PINBAR_BEARISH  # Shooting star
        else:
            return IntentionCandle.DOJI
    
    # Engulfing pattern
    if prev_candle:
        prev_body = abs(prev_candle["close"] - prev_candle["open"])
        curr_body = ratios["body_size"]
        
        if curr_body > prev_body * 1.5:
            prev_dir = "BULLISH" if prev_candle["close"] > prev_candle["open"] else "BEARISH"
            
            if direction == "BULLISH" and prev_dir == "BEARISH":
                return IntentionCandle.ENGULFING_BULLISH
            elif direction == "BEARISH" and prev_dir == "BULLISH":
                return IntentionCandle.ENGULFING_BEARISH
    
    return IntentionCandle.NEUTRAL


def calculate_relative_volume(current_vol: float, avg_vol: float) -> Dict[str, Any]:
    """
    Calcula el volumen relativo vs el promedio.
    
    >2x = Alta actividad institucional
    <0.5x = Mercado dormido
    """
    if avg_vol == 0:
        return {"relative": 1.0, "classification": "NORMAL"}
    
    rel = current_vol / avg_vol
    
    if rel > 3:
        classification = "EXTREME"
    elif rel > 2:
        classification = "HIGH"
    elif rel > 1.5:
        classification = "ELEVATED"
    elif rel > 0.5:
        classification = "NORMAL"
    else:
        classification = "LOW"
    
    return {"relative": round(rel, 2), "classification": classification}


def detect_fakeout(candles: pd.DataFrame, level: float, direction: str) -> Dict[str, Any]:
    """
    Detecta si hubo un Fakeout (trampa) en un nivel.
    
    Fakeout = precio rompe nivel pero cierra con mecha gigante volviendo al rango.
    """
    if len(candles) < 2:
        return {"is_fakeout": False}
    
    last = candles.iloc[-1]
    
    if direction == "BUY":
        # Buscamos si rompió abajo y volvió
        if last["low"] < level and last["close"] > level:
            wick_size = level - last["low"]
            body_size = abs(last["close"] - last["open"])
            
            if wick_size > body_size * 2:
                return {
                    "is_fakeout": True,
                    "type": "STOP_HUNT_UP",
                    "swept_level": last["low"],
                    "recovery_close": last["close"]
                }
    
    elif direction == "SELL":
        # Buscamos si rompió arriba y volvió
        if last["high"] > level and last["close"] < level:
            wick_size = last["high"] - level
            body_size = abs(last["close"] - last["open"])
            
            if wick_size > body_size * 2:
                return {
                    "is_fakeout": True,
                    "type": "STOP_HUNT_DOWN",
                    "swept_level": last["high"],
                    "recovery_close": last["close"]
                }
    
    return {"is_fakeout": False}


def calculate_atr(candles: pd.DataFrame, period: int = 14) -> float:
    """
    Calcula el Average True Range.
    """
    if len(candles) < period:
        return 0.0
    
    high = candles["high"]
    low = candles["low"]
    close = candles["close"].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    
    return atr if not pd.isna(atr) else 0.0


def analyze_aceleracion(
    recent_candles: List[Dict],
    poi_status: str = "NEUTRAL",
    proposed_direction: str = None,
    atr_period: int = 14
) -> Dict[str, Any]:
    """
    ⚡ FEAT MODULE A: Validación de Momentum e Intención.
    
    Confirma si hay aceleración real o es una trampa.
    Este es el último filtro antes de ejecutar.
    
    Returns:
        Dict con análisis de momentum y decisión final
    """
    result = {
        "module": "FEAT_Aceleracion",
        "status": "ANALYZED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "poi_status": poi_status
    }
    
    if not recent_candles or len(recent_candles) < 3:
        result["status"] = "INSUFFICIENT_DATA"
        result["instruction"] = "NEED_MORE_CANDLES"
        return result
    
    df = pd.DataFrame(recent_candles)
    
    # Calculate ATR
    atr = calculate_atr(df, atr_period)
    avg_body = abs(df["close"] - df["open"]).tail(20).mean()
    
    # Current candle analysis
    last_candle = df.iloc[-1].to_dict()
    prev_candle = df.iloc[-2].to_dict()
    
    body_analysis = calculate_body_wick_ratio(last_candle)
    candle_type = classify_candle(last_candle, prev_candle)
    
    # Volume analysis (if available)
    if "volume" in df.columns:
        avg_vol = df["volume"].tail(20).mean()
        curr_vol = last_candle.get("volume", avg_vol)
        vol_analysis = calculate_relative_volume(curr_vol, avg_vol)
    else:
        vol_analysis = {"relative": 1.0, "classification": "UNKNOWN"}
    
    # Momentum classification
    momentum_score = 0
    momentum_factors = []
    
    # Factor 1: Body ratio (max 30 points)
    if body_analysis["body_ratio"] > 70:
        momentum_score += 30
        momentum_factors.append("BODY_STRONG")
    elif body_analysis["body_ratio"] > 50:
        momentum_score += 15
        momentum_factors.append("BODY_MODERATE")
    
    # Factor 2: Volume (max 30 points)
    if vol_analysis["classification"] in ["HIGH", "EXTREME"]:
        momentum_score += 30
        momentum_factors.append("VOLUME_HIGH")
    elif vol_analysis["classification"] == "ELEVATED":
        momentum_score += 15
        momentum_factors.append("VOLUME_ELEVATED")
    
    # Factor 3: Candle type (max 25 points)
    if candle_type in [IntentionCandle.STRONG_BULLISH, IntentionCandle.STRONG_BEARISH]:
        momentum_score += 25
        momentum_factors.append("INTENTION_CANDLE")
    elif candle_type in [IntentionCandle.ENGULFING_BULLISH, IntentionCandle.ENGULFING_BEARISH]:
        momentum_score += 20
        momentum_factors.append("ENGULFING_PATTERN")
    elif candle_type in [IntentionCandle.PINBAR_BULLISH, IntentionCandle.PINBAR_BEARISH]:
        momentum_score += 15
        momentum_factors.append("PINBAR_REJECTION")
    
    # Factor 4: ATR filter (max 15 points)
    current_range = last_candle["high"] - last_candle["low"]
    if atr > 0 and current_range > atr:
        momentum_score += 15
        momentum_factors.append("ABOVE_ATR")
    elif atr > 0 and current_range > atr * 0.5:
        momentum_score += 7
        momentum_factors.append("NORMAL_ATR")
    else:
        momentum_factors.append("LOW_VOLATILITY")
    
    # Fakeout check
    fakeout_check = {"is_fakeout": False}
    if proposed_direction:
        current_price = last_candle["close"]
        key_level = prev_candle["low"] if proposed_direction == "BUY" else prev_candle["high"]
        fakeout_check = detect_fakeout(df, key_level, proposed_direction)
    
    # Final momentum classification
    if fakeout_check.get("is_fakeout"):
        momentum_type = MomentumType.FAKEOUT
    elif momentum_score >= 70:
        momentum_type = MomentumType.IMPULSIVE
    elif momentum_score >= 40:
        momentum_type = MomentumType.ABSORBING
    else:
        momentum_type = MomentumType.WEAK
    
    result["analysis"] = {
        "momentum_score": momentum_score,
        "momentum_type": momentum_type.value,
        "momentum_factors": momentum_factors,
        "candle_analysis": {
            "body_ratio": round(body_analysis["body_ratio"], 1),
            "direction": body_analysis["direction"],
            "candle_type": candle_type.value
        },
        "volume_analysis": vol_analysis,
        "atr": round(atr, 5) if atr else 0,
        "fakeout_check": fakeout_check
    }
    
    # Determine instruction
    if fakeout_check.get("is_fakeout"):
        result["instruction"] = "STOP_CHAIN_FAKEOUT_DETECTED"
    elif momentum_type == MomentumType.IMPULSIVE:
        result["instruction"] = "EXECUTE_TRADE"
        result["confidence"] = "HIGH"
    elif momentum_type == MomentumType.ABSORBING:
        result["instruction"] = "PREPARE_ENTRY_ON_RETRACEMENT"
        result["confidence"] = "MEDIUM"
    else:
        result["instruction"] = "STOP_CHAIN_WEAK_MOMENTUM"
        result["confidence"] = "LOW"
    
    logger.info(f"[FEAT-A] Score={momentum_score}, Type={momentum_type.value}, Factors={momentum_factors}")
    
    return result


# =============================================================================
# Async wrapper for MCP
# =============================================================================

async def feat_validate_aceleracion(
    recent_candles: List[Dict],
    poi_status: str = "NEUTRAL",
    proposed_direction: str = None,
    atr_period: int = 14
) -> Dict[str, Any]:
    """
    MCP Tool: FEAT Module A - Aceleración validation.
    """
    return analyze_aceleracion(recent_candles, poi_status, proposed_direction, atr_period)
