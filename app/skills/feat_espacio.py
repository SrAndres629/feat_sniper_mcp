"""
FEAT Module E: ESPACIO (Geometría y Localización)
==================================================
Determina DÓNDE el Smart Money ha dejado huellas de liquidez (POIs).

Zonas Institucionales:
1. FVG (Fair Value Gap): Desequilibrios de precio
2. OB (Order Block): Última vela contraria antes de impulso
3. ZS (Zona de Sombra): Mechas de rechazo
4. PC (Punto Crítico): Zonas de indecisión con volumen

Estados:
- FRESH: Zona virgen, alta probabilidad
- MITIGATED: Zona ya tocada, baja probabilidad
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger("FEAT.Espacio")


class ZoneType(Enum):
    FVG_BULLISH = "FVG_ALCISTA"
    FVG_BEARISH = "FVG_BAJISTA"
    OB_BULLISH = "OB_ALCISTA"      # Order Block de demanda
    OB_BEARISH = "OB_BAJISTA"      # Order Block de oferta
    ZS_BULLISH = "ZS_ALCISTA"      # Zona de Sombra (rechazo)
    ZS_BEARISH = "ZS_BAJISTA"


class ZoneStatus(Enum):
    FRESH = "FRESH"
    MITIGATED = "MITIGATED"


def detect_fvg(candles: pd.DataFrame, lookback: int = 30) -> List[Dict]:
    """
    Detecta Fair Value Gaps (FVG) - zonas de ineficiencia de precio.
    
    FVG Alcista: Candle1.high < Candle3.low (gap hacia arriba)
    FVG Bajista: Candle1.low > Candle3.high (gap hacia abajo)
    
    Returns:
        Lista de FVGs detectados con sus coordenadas
    """
    fvgs = []
    
    if len(candles) < 3:
        return fvgs
    
    recent = candles.tail(lookback)
    current_price = candles.iloc[-1]["close"]
    
    for i in range(len(recent) - 2):
        c1 = recent.iloc[i]      # Candle 1
        c2 = recent.iloc[i + 1]  # Candle 2 (la de impulso)
        c3 = recent.iloc[i + 2]  # Candle 3
        
        # FVG Alcista (Gap hacia arriba)
        if c1["high"] < c3["low"]:
            zone = {
                "type": ZoneType.FVG_BULLISH.value,
                "top": c3["low"],
                "bottom": c1["high"],
                "midpoint": (c3["low"] + c1["high"]) / 2,
                "candle_index": i,
                "impulse_size": c2["close"] - c2["open"],
                "time": c2.get("time", "")
            }
            
            # Check if mitigated
            future = candles.iloc[i + 2:]
            if any(future["low"] <= zone["top"]):
                zone["status"] = ZoneStatus.MITIGATED.value
            else:
                zone["status"] = ZoneStatus.FRESH.value
            
            # Zone location (Premium/Discount)
            zone["location"] = "DEMAND" if zone["midpoint"] < current_price else "SUPPLY"
            
            fvgs.append(zone)
        
        # FVG Bajista (Gap hacia abajo)
        if c1["low"] > c3["high"]:
            zone = {
                "type": ZoneType.FVG_BEARISH.value,
                "top": c1["low"],
                "bottom": c3["high"],
                "midpoint": (c1["low"] + c3["high"]) / 2,
                "candle_index": i,
                "impulse_size": c2["open"] - c2["close"],
                "time": c2.get("time", "")
            }
            
            # Check if mitigated
            future = candles.iloc[i + 2:]
            if any(future["high"] >= zone["bottom"]):
                zone["status"] = ZoneStatus.MITIGATED.value
            else:
                zone["status"] = ZoneStatus.FRESH.value
            
            zone["location"] = "SUPPLY" if zone["midpoint"] > current_price else "DEMAND"
            
            fvgs.append(zone)
    
    return fvgs


def detect_order_blocks(candles: pd.DataFrame, lookback: int = 30) -> List[Dict]:
    """
    Detecta Order Blocks (OB) - última vela contraria antes de expansión.
    
    OB Alcista: Última vela bajista antes de fuerte movimiento alcista
    OB Bajista: Última vela alcista antes de fuerte movimiento bajista
    
    Validación: Debe haber generado un FVG después.
    """
    obs = []
    
    if len(candles) < 5:
        return obs
    
    recent = candles.tail(lookback)
    current_price = candles.iloc[-1]["close"]
    
    for i in range(len(recent) - 3):
        c_ob = recent.iloc[i]      # Potential Order Block
        c_next = recent.iloc[i + 1]  # Impulse candle
        c_after = recent.iloc[i + 2]  # FVG check
        
        ob_body = c_ob["close"] - c_ob["open"]
        next_body = c_next["close"] - c_next["open"]
        
        # OB Alcista: Vela bajista seguida de vela alcista fuerte
        if ob_body < 0 and next_body > 0 and abs(next_body) > abs(ob_body) * 2:
            # Check for FVG creation
            has_fvg = c_ob["high"] < c_after["low"]
            
            zone = {
                "type": ZoneType.OB_BULLISH.value,
                "top": c_ob["high"],
                "bottom": c_ob["low"],
                "midpoint": (c_ob["high"] + c_ob["low"]) / 2,
                "has_fvg": has_fvg,
                "candle_index": i,
                "time": c_ob.get("time", "")
            }
            
            # Mitigation check
            future = candles.iloc[i + 1:]
            if any(future["low"] <= zone["midpoint"]):
                zone["status"] = ZoneStatus.MITIGATED.value
            else:
                zone["status"] = ZoneStatus.FRESH.value
            
            zone["location"] = "DEMAND"
            obs.append(zone)
        
        # OB Bajista: Vela alcista seguida de vela bajista fuerte
        if ob_body > 0 and next_body < 0 and abs(next_body) > abs(ob_body) * 2:
            has_fvg = c_ob["low"] > c_after["high"]
            
            zone = {
                "type": ZoneType.OB_BEARISH.value,
                "top": c_ob["high"],
                "bottom": c_ob["low"],
                "midpoint": (c_ob["high"] + c_ob["low"]) / 2,
                "has_fvg": has_fvg,
                "candle_index": i,
                "time": c_ob.get("time", "")
            }
            
            future = candles.iloc[i + 1:]
            if any(future["high"] >= zone["midpoint"]):
                zone["status"] = ZoneStatus.MITIGATED.value
            else:
                zone["status"] = ZoneStatus.FRESH.value
            
            zone["location"] = "SUPPLY"
            obs.append(zone)
    
    return obs


def calculate_premium_discount(
    current_price: float,
    range_high: float,
    range_low: float
) -> Dict[str, Any]:
    """
    Calcula si el precio está en zona Premium (vender) o Discount (comprar).
    
    Premium: Por encima del 50% del rango
    Discount: Por debajo del 50% del rango
    OTE (Optimal Trade Entry): 61.8% - 79% del retroceso
    """
    range_size = range_high - range_low
    if range_size == 0:
        return {"zone": "NEUTRAL", "percentage": 50}
    
    # Calculate position in range (0% = low, 100% = high)
    position = ((current_price - range_low) / range_size) * 100
    
    # Determine zone
    if position > 79:
        zone = "EXTREME_PREMIUM"
    elif position > 61.8:
        zone = "PREMIUM_OTE"
    elif position > 50:
        zone = "PREMIUM"
    elif position > 38.2:
        zone = "DISCOUNT"
    elif position > 21:
        zone = "DISCOUNT_OTE"
    else:
        zone = "EXTREME_DISCOUNT"
    
    return {
        "zone": zone,
        "percentage": round(position, 1),
        "is_ote": 21 <= position <= 38.2 or 61.8 <= position <= 79
    }


def analyze_espacio(
    candles: List[Dict],
    current_price: float,
    market_structure: str = "NEUTRAL",
    timeframe: str = "H1"
) -> Dict[str, Any]:
    """
    📍 FEAT MODULE E: Mapeo de Zonas de Liquidez (POIs).
    
    Busca FVGs, Order Blocks y calcula Premium/Discount zones.
    Solo retorna zonas FRESH (no mitigadas).
    
    Returns:
        Dict con POIs mapeados y decisión
    """
    result = {
        "module": "FEAT_Espacio",
        "status": "ZONES_MAPPED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timeframe": timeframe,
        "current_price": current_price,
        "market_structure": market_structure
    }
    
    if not candles:
        result["status"] = "NO_DATA"
        result["instruction"] = "COLLECT_DATA_FIRST"
        return result
    
    df = pd.DataFrame(candles)
    
    # Detect all zones
    fvgs = detect_fvg(df)
    obs = detect_order_blocks(df)
    
    # Filter FRESH zones only (high probability)
    fresh_fvgs = [z for z in fvgs if z["status"] == ZoneStatus.FRESH.value]
    fresh_obs = [z for z in obs if z["status"] == ZoneStatus.FRESH.value]
    
    # Separate demand and supply zones
    demand_zones = []
    supply_zones = []
    
    for zone in fresh_fvgs + fresh_obs:
        if zone["location"] == "DEMAND":
            demand_zones.append(zone)
        else:
            supply_zones.append(zone)
    
    # Sort by distance to current price
    demand_zones.sort(key=lambda z: abs(current_price - z["midpoint"]))
    supply_zones.sort(key=lambda z: abs(current_price - z["midpoint"]))
    
    # Premium/Discount calculation
    if len(df) >= 20:
        range_high = df.tail(50)["high"].max()
        range_low = df.tail(50)["low"].min()
        pd_zone = calculate_premium_discount(current_price, range_high, range_low)
    else:
        pd_zone = {"zone": "UNKNOWN", "percentage": 50}
    
    result["analysis"] = {
        "premium_discount": pd_zone,
        "total_fresh_zones": len(demand_zones) + len(supply_zones),
        "total_mitigated": len(fvgs) + len(obs) - len(fresh_fvgs) - len(fresh_obs)
    }
    
    result["nearest_pois"] = {
        "demand_zones": demand_zones[:3],  # Top 3 closest
        "supply_zones": supply_zones[:3]
    }
    
    # Determine instruction
    nearest_demand = demand_zones[0] if demand_zones else None
    nearest_supply = supply_zones[0] if supply_zones else None
    
    if market_structure in ["BULLISH", "COMPRAS_FUERTES"]:
        if nearest_demand and abs(current_price - nearest_demand["midpoint"]) < current_price * 0.002:
            result["instruction"] = "PROCEED_TO_MODULE_A"
            result["target_zone"] = nearest_demand
        elif nearest_demand:
            result["instruction"] = f"WAIT_FOR_PRICE_TO_REACH_{nearest_demand['midpoint']:.2f}"
            result["target_zone"] = nearest_demand
        else:
            result["instruction"] = "NO_FRESH_DEMAND_ZONES"
    
    elif market_structure in ["BEARISH", "VENTAS_FUERTES"]:
        if nearest_supply and abs(current_price - nearest_supply["midpoint"]) < current_price * 0.002:
            result["instruction"] = "PROCEED_TO_MODULE_A"
            result["target_zone"] = nearest_supply
        elif nearest_supply:
            result["instruction"] = f"WAIT_FOR_PRICE_TO_REACH_{nearest_supply['midpoint']:.2f}"
            result["target_zone"] = nearest_supply
        else:
            result["instruction"] = "NO_FRESH_SUPPLY_ZONES"
    else:
        result["instruction"] = "WAIT_FOR_STRUCTURE_CLARITY"
    
    logger.info(f"[FEAT-E] Fresh Zones: {len(demand_zones)} demand, {len(supply_zones)} supply")
    
    return result


# =============================================================================
# Async wrapper for MCP
# =============================================================================

async def feat_map_espacio(
    candles: List[Dict],
    current_price: float,
    market_structure: str = "NEUTRAL",
    timeframe: str = "H1"
) -> Dict[str, Any]:
    """
    MCP Tool: FEAT Module E - Espacio analysis.
    """
    return analyze_espacio(candles, current_price, market_structure, timeframe)
