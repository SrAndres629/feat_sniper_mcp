"""
FEAT Module E: ESPACIO (Geometría de Liquidez) - CHRONO-AWARE
==============================================================
Determina DÓNDE el Smart Money ha dejado huellas de liquidez.

Zonas Institucionales:
1. FVG (Fair Value Gap): Desequilibrios de precio
2. OB (Order Block): Última vela contraria antes de impulso
3. Breaker Block: OB que falló y ahora es soporte/resistencia

INTEGRACIÓN TEMPORAL:
- Zonas creadas en Kill Zone tienen mayor probabilidad
- Zonas de Lunes (INDUCTION) son potenciales trampas
- Premium/Discount ajustado por ciclo semanal
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger("FEAT.EspacioChronoAware")


# =============================================================================
# ENUMS
# =============================================================================

class ZoneType(Enum):
    FVG_BULLISH = "FVG_ALCISTA"
    FVG_BEARISH = "FVG_BAJISTA"
    OB_BULLISH = "OB_ALCISTA"
    OB_BEARISH = "OB_BAJISTA"
    BREAKER_BULLISH = "BREAKER_ALCISTA"
    BREAKER_BEARISH = "BREAKER_BAJISTA"


class ZoneStatus(Enum):
    FRESH = "FRESH"
    TESTED = "TESTED"
    MITIGATED = "MITIGATED"


class ZoneQuality(Enum):
    INSTITUTIONAL = "INSTITUTIONAL"  # Creado en Kill Zone
    STANDARD = "STANDARD"            # Creado fuera de KZ
    SUSPICIOUS = "SUSPICIOUS"        # Creado en INDUCTION (Lunes)


# =============================================================================
# FVG DETECTION
# =============================================================================

def detect_fvg(candles: pd.DataFrame, lookback: int = 30) -> List[Dict]:
    """
    Detecta Fair Value Gaps con quality scoring.
    """
    fvgs = []
    
    if len(candles) < 3:
        return fvgs
    
    recent = candles.tail(lookback)
    current_price = candles.iloc[-1]["close"]
    
    for i in range(len(recent) - 2):
        c1 = recent.iloc[i]
        c2 = recent.iloc[i + 1]  # Impulse candle
        c3 = recent.iloc[i + 2]
        
        impulse_size = abs(c2["close"] - c2["open"])
        
        # FVG Alcista
        if c1["high"] < c3["low"]:
            gap_size = c3["low"] - c1["high"]
            zone = {
                "type": ZoneType.FVG_BULLISH.value,
                "top": c3["low"],
                "bottom": c1["high"],
                "midpoint": (c3["low"] + c1["high"]) / 2,
                "gap_size": gap_size,
                "impulse_size": impulse_size,
                "candle_index": i,
                "quality_score": min(1.0, impulse_size / (gap_size + 0.001) * 0.5)
            }
            
            # Check mitigation
            future = candles.iloc[i + 2:]
            touches = sum(1 for _, row in future.iterrows() if row["low"] <= zone["top"])
            
            if touches == 0:
                zone["status"] = ZoneStatus.FRESH.value
                zone["status_score"] = 1.0
            elif touches == 1:
                zone["status"] = ZoneStatus.TESTED.value
                zone["status_score"] = 0.7
            else:
                zone["status"] = ZoneStatus.MITIGATED.value
                zone["status_score"] = 0.2
            
            zone["distance_to_price"] = abs(current_price - zone["midpoint"])
            zone["distance_pct"] = zone["distance_to_price"] / current_price * 100
            
            fvgs.append(zone)
        
        # FVG Bajista
        if c1["low"] > c3["high"]:
            gap_size = c1["low"] - c3["high"]
            zone = {
                "type": ZoneType.FVG_BEARISH.value,
                "top": c1["low"],
                "bottom": c3["high"],
                "midpoint": (c1["low"] + c3["high"]) / 2,
                "gap_size": gap_size,
                "impulse_size": impulse_size,
                "candle_index": i,
                "quality_score": min(1.0, impulse_size / (gap_size + 0.001) * 0.5)
            }
            
            future = candles.iloc[i + 2:]
            touches = sum(1 for _, row in future.iterrows() if row["high"] >= zone["bottom"])
            
            if touches == 0:
                zone["status"] = ZoneStatus.FRESH.value
                zone["status_score"] = 1.0
            elif touches == 1:
                zone["status"] = ZoneStatus.TESTED.value
                zone["status_score"] = 0.7
            else:
                zone["status"] = ZoneStatus.MITIGATED.value
                zone["status_score"] = 0.2
            
            zone["distance_to_price"] = abs(current_price - zone["midpoint"])
            zone["distance_pct"] = zone["distance_to_price"] / current_price * 100
            
            fvgs.append(zone)
    
    return fvgs


# =============================================================================
# ORDER BLOCK DETECTION
# =============================================================================

def detect_order_blocks(candles: pd.DataFrame, lookback: int = 30) -> List[Dict]:
    """
    Detecta Order Blocks con validación de FVG.
    """
    obs = []
    
    if len(candles) < 5:
        return obs
    
    recent = candles.tail(lookback)
    current_price = candles.iloc[-1]["close"]
    
    for i in range(len(recent) - 3):
        c_ob = recent.iloc[i]
        c_next = recent.iloc[i + 1]
        c_after = recent.iloc[i + 2]
        
        ob_body = c_ob["close"] - c_ob["open"]
        next_body = c_next["close"] - c_next["open"]
        
        # OB Alcista: vela bajista seguida de alcista fuerte
        if ob_body < 0 and next_body > 0 and abs(next_body) > abs(ob_body) * 1.5:
            has_fvg = c_ob["high"] < c_after["low"]
            
            zone = {
                "type": ZoneType.OB_BULLISH.value,
                "top": c_ob["high"],
                "bottom": c_ob["low"],
                "midpoint": (c_ob["high"] + c_ob["low"]) / 2,
                "body_ratio": abs(ob_body) / (c_ob["high"] - c_ob["low"]) if (c_ob["high"] - c_ob["low"]) > 0 else 0,
                "has_fvg": has_fvg,
                "quality_score": 0.8 if has_fvg else 0.5,
                "candle_index": i
            }
            
            # Check mitigation
            future = candles.iloc[i + 1:]
            touches = sum(1 for _, row in future.iterrows() if row["low"] <= zone["midpoint"])
            
            if touches == 0:
                zone["status"] = ZoneStatus.FRESH.value
                zone["status_score"] = 1.0
            elif touches <= 2:
                zone["status"] = ZoneStatus.TESTED.value
                zone["status_score"] = 0.6
            else:
                zone["status"] = ZoneStatus.MITIGATED.value
                zone["status_score"] = 0.2
            
            zone["distance_to_price"] = abs(current_price - zone["midpoint"])
            obs.append(zone)
        
        # OB Bajista
        if ob_body > 0 and next_body < 0 and abs(next_body) > abs(ob_body) * 1.5:
            has_fvg = c_ob["low"] > c_after["high"]
            
            zone = {
                "type": ZoneType.OB_BEARISH.value,
                "top": c_ob["high"],
                "bottom": c_ob["low"],
                "midpoint": (c_ob["high"] + c_ob["low"]) / 2,
                "body_ratio": abs(ob_body) / (c_ob["high"] - c_ob["low"]) if (c_ob["high"] - c_ob["low"]) > 0 else 0,
                "has_fvg": has_fvg,
                "quality_score": 0.8 if has_fvg else 0.5,
                "candle_index": i
            }
            
            future = candles.iloc[i + 1:]
            touches = sum(1 for _, row in future.iterrows() if row["high"] >= zone["midpoint"])
            
            if touches == 0:
                zone["status"] = ZoneStatus.FRESH.value
                zone["status_score"] = 1.0
            elif touches <= 2:
                zone["status"] = ZoneStatus.TESTED.value
                zone["status_score"] = 0.6
            else:
                zone["status"] = ZoneStatus.MITIGATED.value
                zone["status_score"] = 0.2
            
            zone["distance_to_price"] = abs(current_price - zone["midpoint"])
            obs.append(zone)
    
    return obs


# =============================================================================
# PREMIUM/DISCOUNT CALCULATION
# =============================================================================

def calculate_premium_discount(
    current_price: float,
    range_high: float,
    range_low: float
) -> Dict[str, Any]:
    """
    Calcula si el precio está en Premium (vender) o Discount (comprar).
    Equilibrium = 50%
    """
    range_size = range_high - range_low
    if range_size == 0:
        return {"zone": "EQUILIBRIUM", "percentage": 50, "position_score": 0.5}
    
    position = ((current_price - range_low) / range_size) * 100
    
    if position >= 79:
        zone = "EXTREME_PREMIUM"
        action_bias = "SELL"
        position_score = 0.95
    elif position >= 61.8:
        zone = "PREMIUM_OTE"
        action_bias = "SELL"
        position_score = 0.80
    elif position >= 50:
        zone = "PREMIUM"
        action_bias = "SELL_BIAS"
        position_score = 0.65
    elif position >= 38.2:
        zone = "DISCOUNT"
        action_bias = "BUY_BIAS"
        position_score = 0.65
    elif position >= 21:
        zone = "DISCOUNT_OTE"
        action_bias = "BUY"
        position_score = 0.80
    else:
        zone = "EXTREME_DISCOUNT"
        action_bias = "BUY"
        position_score = 0.95
    
    return {
        "zone": zone,
        "percentage": round(position, 1),
        "action_bias": action_bias,
        "position_score": round(position_score, 2),
        "is_ote": 21 <= position <= 38.2 or 61.8 <= position <= 79,
        "fib_proximity": {
            "to_618": abs(position - 61.8),
            "to_382": abs(position - 38.2),
            "to_50": abs(position - 50)
        }
    }


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_espacio(
    candles: List[Dict],
    current_price: float,
    market_structure: str = "NEUTRAL",
    timeframe: str = "H1",
    chrono_features: Dict = None
) -> Dict[str, Any]:
    """
    📍 FEAT MODULE E: Mapeo de Zonas de Liquidez (Chrono-Aware).
    
    Busca FVGs, Order Blocks y calcula Premium/Discount.
    Integra contexto temporal para validar calidad de zonas.
    """
    result = {
        "module": "FEAT_Espacio_ChronoAware",
        "status": "ZONES_MAPPED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timeframe": timeframe,
        "current_price": current_price,
        "market_structure": market_structure
    }
    
    if not candles:
        result["status"] = "NO_DATA"
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
    
    # Zone quality modifier
    zone_quality_mult = 1.0
    if weekly_phase == "INDUCTION":
        zone_quality_mult = 0.6  # Lunes: zonas sospechosas
    elif is_killzone:
        zone_quality_mult = 1.2  # Kill Zone: zonas de alta calidad
    
    df = pd.DataFrame(candles)
    
    # Detect zones
    fvgs = detect_fvg(df)
    obs = detect_order_blocks(df)
    
    # Apply chrono quality modifier
    for zone in fvgs + obs:
        zone["chrono_quality_mult"] = zone_quality_mult
        zone["final_score"] = round(zone["quality_score"] * zone["status_score"] * zone_quality_mult, 2)
    
    # Filter by score and status
    high_quality_zones = [z for z in fvgs + obs if z["final_score"] >= 0.5 and z["status"] != ZoneStatus.MITIGATED.value]
    
    # Separate demand and supply
    demand_zones = [z for z in high_quality_zones if "ALCISTA" in z["type"]]
    supply_zones = [z for z in high_quality_zones if "BAJISTA" in z["type"]]
    
    # Sort by distance
    demand_zones.sort(key=lambda z: z.get("distance_to_price", float('inf')))
    supply_zones.sort(key=lambda z: z.get("distance_to_price", float('inf')))
    
    # Premium/Discount
    if len(df) >= 20:
        range_high = df.tail(50)["high"].max()
        range_low = df.tail(50)["low"].min()
        pd_zone = calculate_premium_discount(current_price, range_high, range_low)
    else:
        pd_zone = {"zone": "UNKNOWN", "percentage": 50, "position_score": 0.5}
    
    # Check if price is near a zone
    nearest_demand = demand_zones[0] if demand_zones else None
    nearest_supply = supply_zones[0] if supply_zones else None
    
    price_in_zone = False
    active_zone = None
    
    if nearest_demand and nearest_demand["distance_pct"] < 0.3:
        price_in_zone = True
        active_zone = nearest_demand
    elif nearest_supply and nearest_supply["distance_pct"] < 0.3:
        price_in_zone = True
        active_zone = nearest_supply
    
    result["analysis"] = {
        "premium_discount": pd_zone,
        "zones_detected": {
            "total_fvg": len(fvgs),
            "total_ob": len(obs),
            "high_quality": len(high_quality_zones),
            "demand_zones": len(demand_zones),
            "supply_zones": len(supply_zones)
        },
        "chrono_modifier": zone_quality_mult
    }
    
    result["nearest_pois"] = {
        "demand_zones": demand_zones[:3],
        "supply_zones": supply_zones[:3]
    }
    
    result["price_in_zone"] = price_in_zone
    result["active_zone"] = active_zone
    
    # ML Features
    result["ml_features"] = {
        # Zone counts
        "total_fresh_zones": len([z for z in fvgs + obs if z["status"] == ZoneStatus.FRESH.value]),
        "total_tested_zones": len([z for z in fvgs + obs if z["status"] == ZoneStatus.TESTED.value]),
        "demand_zone_count": len(demand_zones),
        "supply_zone_count": len(supply_zones),
        
        # Premium/Discount
        "pd_percentage": pd_zone["percentage"] / 100,
        "is_premium": 1 if pd_zone["percentage"] > 50 else 0,
        "is_discount": 1 if pd_zone["percentage"] < 50 else 0,
        "is_ote": 1 if pd_zone.get("is_ote") else 0,
        
        # Proximity
        "price_in_zone": 1 if price_in_zone else 0,
        "nearest_zone_score": active_zone["final_score"] if active_zone else 0,
        
        # Chrono
        "zone_quality_mult": zone_quality_mult,
        "chrono_risk_mult": chrono_risk_mult
    }
    
    # Guidance
    zone_probability = active_zone["final_score"] if active_zone else 0.3
    combined_probability = zone_probability * chrono_risk_mult
    
    result["guidance"] = {
        "zone_probability": round(zone_probability, 2),
        "combined_probability": round(min(1.0, combined_probability), 2),
        "can_proceed": combined_probability > 0.4,
        "cautions": []
    }
    
    if weekly_phase == "INDUCTION":
        result["guidance"]["cautions"].append("📅 LUNES: Zonas pueden ser trampas")
    if not price_in_zone and high_quality_zones:
        result["guidance"]["cautions"].append(f"⏳ Precio lejos de POI. Esperar retroceso a zona")
    if pd_zone["percentage"] > 70 and market_structure == "BULLISH":
        result["guidance"]["cautions"].append("⚠️ Comprando en Premium - riesgo elevado")
    if pd_zone["percentage"] < 30 and market_structure == "BEARISH":
        result["guidance"]["cautions"].append("⚠️ Vendiendo en Discount - riesgo elevado")
    
    logger.info(f"[FEAT-E] HQ Zones: {len(high_quality_zones)}, P/D: {pd_zone['zone']}, InZone: {price_in_zone}")
    
    return result


def generate_liquidity_features(
    candles: List[Dict],
    current_price: float,
    chrono_features: Dict = None
) -> Dict[str, Any]:
    """
    Genera vector de features de liquidez para ML.
    """
    analysis = analyze_espacio(candles, current_price, "NEUTRAL", "H1", chrono_features)
    return {
        "module": "FEAT_Espacio_ML",
        "features": analysis.get("ml_features", {}),
        "zone_probability": analysis.get("guidance", {}).get("zone_probability", 0)
    }


# =============================================================================
# ASYNC MCP WRAPPERS
# =============================================================================

async def feat_map_espacio(
    candles: List[Dict],
    current_price: float,
    market_structure: str = "NEUTRAL",
    timeframe: str = "H1"
) -> Dict[str, Any]:
    """MCP Tool: FEAT Module E - Espacio (legacy)."""
    return analyze_espacio(candles, current_price, market_structure, timeframe)


async def feat_map_espacio_advanced(
    candles: List[Dict],
    current_price: float,
    market_structure: str = "NEUTRAL",
    chrono_features: Dict = None
) -> Dict[str, Any]:
    """MCP Tool: FEAT Module E - Espacio (chrono-aware)."""
    return analyze_espacio(candles, current_price, market_structure, "H1", chrono_features)


async def feat_generate_liquidity_features(
    candles: List[Dict],
    current_price: float,
    chrono_features: Dict = None
) -> Dict[str, Any]:
    """MCP Tool: Generate ML-ready liquidity features."""
    return generate_liquidity_features(candles, current_price, chrono_features)
