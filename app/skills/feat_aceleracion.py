"""
FEAT Module A: ACELERACIN (Vector de Intencin) - CHRONO-AWARE
================================================================
Mide velocidad y volumen para validar entrada y descartar Fakeouts.

Conceptos Clave:
- Body/Wick Ratio: Vela de intencin = cuerpo > 70%, mecha < 30%
- Volume Confirmation: Volumen > 1.5x promedio
- Fakeout Detection: Mecha que barre nivel pero cierra dentro

INTEGRACIN TEMPORAL:
- Momentum en Kill Zone tiene mayor peso
- Fakeouts de Lunes (INDUCTION) son ms comunes
- ATR normalizado por sesin
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger("FEAT.AceleracionChronoAware")


# =============================================================================
# ENUMS
# =============================================================================

class MomentumType(Enum):
    IMPULSIVE = "IMPULSIVO"
    ABSORBING = "ABSORBIENDO"
    WEAK = "DEBIL"
    FAKEOUT = "FAKEOUT"
    EXHAUSTION = "AGOTAMIENTO"


class CandlePattern(Enum):
    STRONG_BULLISH = "VELA_ALCISTA_FUERTE"
    STRONG_BEARISH = "VELA_BAJISTA_FUERTE"
    ENGULFING_BULLISH = "ENVOLVENTE_ALCISTA"
    ENGULFING_BEARISH = "ENVOLVENTE_BAJISTA"
    PINBAR_BULLISH = "PINBAR_ALCISTA"
    PINBAR_BEARISH = "PINBAR_BAJISTA"
    DOJI = "DOJI"
    EXHAUSTION_CANDLE = "VELA_AGOTAMIENTO"
    NEUTRAL = "NEUTRAL"


# =============================================================================
# CANDLE ANALYSIS
# =============================================================================

def analyze_candle_structure(candle: Dict) -> Dict[str, Any]:
    """
    Analiza estructura de una vela individual.
    """
    high = candle["high"]
    low = candle["low"]
    open_ = candle["open"]
    close = candle["close"]
    
    total_range = high - low
    if total_range == 0:
        return {"body_ratio": 0, "direction": "NEUTRAL", "is_intention": False}
    
    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    
    body_ratio = body / total_range
    upper_wick_ratio = upper_wick / total_range
    lower_wick_ratio = lower_wick / total_range
    
    direction = "BULLISH" if close > open_ else "BEARISH" if close < open_ else "NEUTRAL"
    
    # Intention candle: body > 70%, both wicks < 15%
    is_intention = body_ratio > 0.70 and upper_wick_ratio < 0.15 and lower_wick_ratio < 0.15
    
    # Exhaustion candle: long wick opposite to direction
    is_exhaustion = False
    if direction == "BULLISH" and upper_wick_ratio > 0.50:
        is_exhaustion = True
    elif direction == "BEARISH" and lower_wick_ratio > 0.50:
        is_exhaustion = True
    
    return {
        "body_ratio": round(body_ratio * 100, 1),
        "upper_wick_ratio": round(upper_wick_ratio * 100, 1),
        "lower_wick_ratio": round(lower_wick_ratio * 100, 1),
        "body_size": body,
        "total_range": total_range,
        "direction": direction,
        "is_intention": is_intention,
        "is_exhaustion": is_exhaustion,
        "intention_score": min(1.0, body_ratio * 1.2) if is_intention else body_ratio * 0.5
    }


def classify_candle_pattern(candle: Dict, prev_candle: Dict = None) -> Tuple[CandlePattern, float]:
    """
    Clasifica patrn de vela con score de confianza.
    """
    structure = analyze_candle_structure(candle)
    body_ratio = structure["body_ratio"]
    direction = structure["direction"]
    
    # Strong candles
    if body_ratio > 70:
        if direction == "BULLISH":
            return (CandlePattern.STRONG_BULLISH, 0.90)
        elif direction == "BEARISH":
            return (CandlePattern.STRONG_BEARISH, 0.90)
    
    # Exhaustion
    if structure["is_exhaustion"]:
        return (CandlePattern.EXHAUSTION_CANDLE, 0.75)
    
    # Pinbar
    if body_ratio < 30:
        if structure["lower_wick_ratio"] > 60:
            return (CandlePattern.PINBAR_BULLISH, 0.80)
        elif structure["upper_wick_ratio"] > 60:
            return (CandlePattern.PINBAR_BEARISH, 0.80)
        else:
            return (CandlePattern.DOJI, 0.50)
    
    # Engulfing
    if prev_candle:
        prev_body = abs(prev_candle["close"] - prev_candle["open"])
        curr_body = structure["body_size"]
        prev_dir = "BULLISH" if prev_candle["close"] > prev_candle["open"] else "BEARISH"
        
        if curr_body > prev_body * 1.5:
            if direction == "BULLISH" and prev_dir == "BEARISH":
                return (CandlePattern.ENGULFING_BULLISH, 0.85)
            elif direction == "BEARISH" and prev_dir == "BULLISH":
                return (CandlePattern.ENGULFING_BEARISH, 0.85)
    
    return (CandlePattern.NEUTRAL, 0.40)


from typing import Tuple


# =============================================================================
# VOLUME ANALYSIS
# =============================================================================

def analyze_volume(current_vol: float, avg_vol: float, prev_vol: float = None) -> Dict[str, Any]:
    """
    Analiza volumen relativo.
    """
    if avg_vol == 0:
        return {"relative": 1.0, "classification": "UNKNOWN", "volume_score": 0.5}
    
    relative = current_vol / avg_vol
    
    if relative > 3.0:
        classification = "EXTREME"
        score = 1.0
    elif relative > 2.0:
        classification = "HIGH"
        score = 0.90
    elif relative > 1.5:
        classification = "ELEVATED"
        score = 0.75
    elif relative > 0.8:
        classification = "NORMAL"
        score = 0.50
    else:
        classification = "LOW"
        score = 0.25
    
    # Volume divergence check
    vol_increasing = prev_vol and current_vol > prev_vol * 1.2 if prev_vol else False
    
    return {
        "relative": round(relative, 2),
        "classification": classification,
        "volume_score": score,
        "vol_increasing": vol_increasing
    }


# =============================================================================
# FAKEOUT DETECTION
# =============================================================================

def detect_fakeout(candles: pd.DataFrame, level: float, direction: str) -> Dict[str, Any]:
    """
    Detecta si hubo un Fakeout en un nivel.
    Fakeout = precio rompe nivel pero cierra con mecha volviendo al rango.
    """
    if len(candles) < 2:
        return {"is_fakeout": False, "fakeout_score": 0}
    
    last = candles.iloc[-1]
    prev = candles.iloc[-2]
    
    fakeout_result = {"is_fakeout": False, "fakeout_score": 0}
    
    if direction == "BUY":
        # Fakeout bajista: mecha rompi bajo pero cerr arriba
        if last["low"] < level and last["close"] > level:
            wick_size = level - last["low"]
            body_size = abs(last["close"] - last["open"])
            
            if wick_size > body_size * 1.5:
                fakeout_result = {
                    "is_fakeout": True,
                    "type": "STOP_HUNT_DOWN",
                    "swept_level": last["low"],
                    "close_above": last["close"],
                    "fakeout_score": min(1.0, wick_size / (body_size + 0.001) * 0.3)
                }
    
    elif direction == "SELL":
        # Fakeout alcista: mecha rompi arriba pero cerr abajo
        if last["high"] > level and last["close"] < level:
            wick_size = last["high"] - level
            body_size = abs(last["close"] - last["open"])
            
            if wick_size > body_size * 1.5:
                fakeout_result = {
                    "is_fakeout": True,
                    "type": "STOP_HUNT_UP",
                    "swept_level": last["high"],
                    "close_below": last["close"],
                    "fakeout_score": min(1.0, wick_size / (body_size + 0.001) * 0.3)
                }
    
    return fakeout_result


# =============================================================================
# ATR CALCULATION
# =============================================================================

def calculate_atr(candles: pd.DataFrame, period: int = 14) -> float:
    """
    Calcula Average True Range.
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
    
    return float(atr) if not pd.isna(atr) else 0.0


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_aceleracion(
    recent_candles: List[Dict],
    poi_status: str = "NEUTRAL",
    proposed_direction: str = None,
    atr_period: int = 14,
    chrono_features: Dict = None
) -> Dict[str, Any]:
    """
     FEAT MODULE A: Validacin de Momentum (Chrono-Aware).
    
    Confirma si hay aceleracin real o es una trampa.
    Este es el ltimo filtro antes de ejecutar.
    """
    result = {
        "module": "FEAT_Aceleracion_ChronoAware",
        "status": "ANALYZED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "poi_status": poi_status
    }
    
    if not recent_candles or len(recent_candles) < 3:
        result["status"] = "INSUFFICIENT_DATA"
        return result
    
    # Extract chrono context
    weekly_phase = None
    chrono_risk_mult = 1.0
    is_killzone = False
    
    if chrono_features:
        weekly_phase = chrono_features.get("weekly", {}).get("phase")
        chrono_risk_mult = chrono_features.get("risk", {}).get("combined_risk_multiplier", 1.0)
        ml_features = chrono_features.get("ml_features", {})
        is_killzone = ml_features.get("is_any_killzone", 0) == 1
    
    # Chrono modifiers
    momentum_chrono_mult = 1.0
    if weekly_phase == "INDUCTION":
        momentum_chrono_mult = 0.7  # Lunes: momentum menos confiable
    elif is_killzone:
        momentum_chrono_mult = 1.15  # Kill Zone: momentum ms confiable
    elif weekly_phase == "EXPANSION":
        momentum_chrono_mult = 1.1  # Jueves: buena expansin
    
    df = pd.DataFrame(recent_candles)
    
    # ATR
    atr = calculate_atr(df, atr_period)
    
    # Candle analysis
    last_candle = df.iloc[-1].to_dict()
    prev_candle = df.iloc[-2].to_dict()
    
    candle_structure = analyze_candle_structure(last_candle)
    candle_pattern, pattern_confidence = classify_candle_pattern(last_candle, prev_candle)
    
    # Volume analysis
    if "volume" in df.columns:
        avg_vol = df["volume"].tail(20).mean()
        prev_vol = df.iloc[-2]["volume"] if len(df) > 1 else None
        vol_analysis = analyze_volume(last_candle.get("volume", avg_vol), avg_vol, prev_vol)
    else:
        vol_analysis = {"relative": 1.0, "classification": "UNKNOWN", "volume_score": 0.5}
    
    # Fakeout check
    fakeout_check = {"is_fakeout": False, "fakeout_score": 0}
    if proposed_direction:
        key_level = prev_candle["low"] if proposed_direction == "BUY" else prev_candle["high"]
        fakeout_check = detect_fakeout(df, key_level, proposed_direction)
    
    # Calculate momentum score
    momentum_score = 0
    momentum_factors = []
    
    # Factor 1: Body ratio (max 30)
    if candle_structure["body_ratio"] > 70:
        momentum_score += 30
        momentum_factors.append("STRONG_BODY")
    elif candle_structure["body_ratio"] > 50:
        momentum_score += 15
        momentum_factors.append("MODERATE_BODY")
    
    # Factor 2: Volume (max 25)
    if vol_analysis["classification"] in ["HIGH", "EXTREME"]:
        momentum_score += 25
        momentum_factors.append("HIGH_VOLUME")
    elif vol_analysis["classification"] == "ELEVATED":
        momentum_score += 12
        momentum_factors.append("ELEVATED_VOLUME")
    
    # Factor 3: Pattern (max 25)
    if candle_pattern in [CandlePattern.STRONG_BULLISH, CandlePattern.STRONG_BEARISH]:
        momentum_score += 25
        momentum_factors.append("INTENTION_CANDLE")
    elif candle_pattern in [CandlePattern.ENGULFING_BULLISH, CandlePattern.ENGULFING_BEARISH]:
        momentum_score += 20
        momentum_factors.append("ENGULFING")
    elif candle_pattern in [CandlePattern.PINBAR_BULLISH, CandlePattern.PINBAR_BEARISH]:
        momentum_score += 15
        momentum_factors.append("PINBAR_REJECTION")
    
    # Factor 4: ATR (max 20)
    current_range = last_candle["high"] - last_candle["low"]
    if atr > 0:
        atr_ratio = current_range / atr
        if atr_ratio > 1.5:
            momentum_score += 20
            momentum_factors.append("STRONG_ATR_BREAK")
        elif atr_ratio > 1.0:
            momentum_score += 10
            momentum_factors.append("ABOVE_ATR")
        elif atr_ratio < 0.5:
            momentum_factors.append("LOW_VOLATILITY")
    
    # Apply chrono modifier
    adjusted_momentum = momentum_score * momentum_chrono_mult
    
    # Classify momentum
    if fakeout_check.get("is_fakeout"):
        momentum_type = MomentumType.FAKEOUT
    elif candle_structure["is_exhaustion"]:
        momentum_type = MomentumType.EXHAUSTION
    elif adjusted_momentum >= 65:
        momentum_type = MomentumType.IMPULSIVE
    elif adjusted_momentum >= 40:
        momentum_type = MomentumType.ABSORBING
    else:
        momentum_type = MomentumType.WEAK
    
    # Final execution probability
    base_probability = adjusted_momentum / 100
    
    if fakeout_check.get("is_fakeout"):
        execution_probability = 0.1  # Almost no execution on fakeout
    elif momentum_type == MomentumType.EXHAUSTION:
        execution_probability = 0.3
    else:
        execution_probability = min(1.0, base_probability * chrono_risk_mult)
    
    result["analysis"] = {
        "momentum_score": round(adjusted_momentum, 1),
        "momentum_type": momentum_type.value,
        "momentum_factors": momentum_factors,
        "candle_analysis": candle_structure,
        "candle_pattern": candle_pattern.value,
        "pattern_confidence": pattern_confidence,
        "volume_analysis": vol_analysis,
        "atr": round(atr, 5) if atr else 0,
        "fakeout_check": fakeout_check,
        "chrono_modifier": momentum_chrono_mult
    }
    
    result["execution_probability"] = round(execution_probability, 2)
    
    # ML Features
    result["ml_features"] = {
        # Candle structure
        "body_ratio": candle_structure["body_ratio"] / 100,
        "upper_wick_ratio": candle_structure["upper_wick_ratio"] / 100,
        "lower_wick_ratio": candle_structure["lower_wick_ratio"] / 100,
        "is_intention": 1 if candle_structure["is_intention"] else 0,
        "is_exhaustion": 1 if candle_structure["is_exhaustion"] else 0,
        
        # Volume
        "volume_relative": vol_analysis["relative"],
        "volume_score": vol_analysis["volume_score"],
        
        # Momentum
        "momentum_score_normalized": adjusted_momentum / 100,
        "is_impulsive": 1 if momentum_type == MomentumType.IMPULSIVE else 0,
        "is_fakeout": 1 if fakeout_check.get("is_fakeout") else 0,
        
        # Chrono
        "chrono_momentum_mult": momentum_chrono_mult,
        "chrono_risk_mult": chrono_risk_mult,
        
        # Final
        "execution_probability": execution_probability
    }
    
    # Guidance
    result["guidance"] = {
        "can_execute": execution_probability > 0.5,
        "recommended_action": "EXECUTE" if execution_probability > 0.7 else ("PREPARE" if execution_probability > 0.5 else "WAIT"),
        "size_multiplier": round(min(1.0, execution_probability * 1.2), 2),
        "cautions": []
    }
    
    if fakeout_check.get("is_fakeout"):
        result["guidance"]["cautions"].append(" FAKEOUT detectado - NO ejecutar")
    if momentum_type == MomentumType.EXHAUSTION:
        result["guidance"]["cautions"].append(" Vela de agotamiento - posible reversin")
    if weekly_phase == "INDUCTION":
        result["guidance"]["cautions"].append(" LUNES: Momentum puede ser falso")
    if vol_analysis["classification"] == "LOW":
        result["guidance"]["cautions"].append(" Volumen bajo - confirmar antes de entry")
    
    logger.info(f"[FEAT-A] Score={adjusted_momentum:.1f}, Type={momentum_type.value}, ExecProb={execution_probability:.2f}")
    
    return result


def generate_momentum_features(
    recent_candles: List[Dict],
    proposed_direction: str = None,
    chrono_features: Dict = None
) -> Dict[str, Any]:
    """
    Genera vector de features de momentum para ML.
    """
    analysis = analyze_aceleracion(recent_candles, "NEUTRAL", proposed_direction, 14, chrono_features)
    return {
        "module": "FEAT_Aceleracion_ML",
        "features": analysis.get("ml_features", {}),
        "execution_probability": analysis.get("execution_probability", 0)
    }


# =============================================================================
# ASYNC MCP WRAPPERS
# =============================================================================

async def feat_validate_aceleracion(
    recent_candles: List[Dict],
    poi_status: str = "NEUTRAL",
    proposed_direction: str = None
) -> Dict[str, Any]:
    """MCP Tool: FEAT Module A - Aceleracin (legacy)."""
    return analyze_aceleracion(recent_candles, poi_status, proposed_direction)


async def feat_validate_aceleracion_advanced(
    recent_candles: List[Dict],
    proposed_direction: str = None,
    chrono_features: Dict = None
) -> Dict[str, Any]:
    """MCP Tool: FEAT Module A - Aceleracin (chrono-aware)."""
    return analyze_aceleracion(recent_candles, "NEUTRAL", proposed_direction, 14, chrono_features)


async def feat_generate_momentum_features(
    recent_candles: List[Dict],
    proposed_direction: str = None,
    chrono_features: Dict = None
) -> Dict[str, Any]:
    """MCP Tool: Generate ML-ready momentum features."""
    return generate_momentum_features(recent_candles, proposed_direction, chrono_features)
