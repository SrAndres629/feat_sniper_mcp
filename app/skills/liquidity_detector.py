"""
Liquidity Intelligence Module - FEAT-DEEP Protocol
===================================================
Deteccin de piscinas de liquidez institucional:
- Swing Highs/Lows no mitigados
- Asian Session Sweeps
- Fair Value Gaps (FVG)

Kill Zone Detection:
- NY Session: 07:00-11:00 UTC-4 (11:00-15:00 UTC)
- London: 03:00-05:00 UTC-4 (07:00-09:00 UTC)
- Asia: 20:00-00:00 UTC-4 (00:00-04:00 UTC)
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from app.core.config import settings

logger = logging.getLogger("FEAT.LiquidityIntelligence")

# [LEVEL 57] Doctoral KILL_ZONES (Linked to Settings)
KILL_ZONES = {
    "NY": {"start": settings.KILLZONE_NY_START, "end": settings.KILLZONE_NY_END},
    "LONDON": {"start": settings.KILLZONE_LONDON_START, "end": settings.KILLZONE_LONDON_END},
    "ASIA": {"start": settings.KILLZONE_ASIA_START, "end": settings.KILLZONE_ASIA_END},
}


def get_current_kill_zone(utc_offset: int = -4) -> Optional[str]:
    """
    Determina la Kill Zone activa basndose en la hora actual.
    
    Returns:
        "NY", "LONDON", "ASIA" o None si no hay KZ activa.
    """
    now = datetime.now(timezone.utc) + timedelta(hours=utc_offset)
    hour = now.hour
    
    for zone, times in KILL_ZONES.items():
        start, end = times["start"], times["end"]
        
        # Handle midnight wrap (Asia)
        if start > end:
            if hour >= start or hour < end:
                return zone
        else:
            if start <= hour < end:
                return zone
    
    return None


def is_in_kill_zone(zone: str = "NY", utc_offset: int = None) -> bool:
    """
    Verifica si estamos dentro de una Kill Zone especfica.
    """
    if utc_offset is None:
        utc_offset = settings.UTC_OFFSET
    current = get_current_kill_zone(utc_offset)
    return current == zone


def detect_liquidity_pools(
    candles: pd.DataFrame,
    lookback: int = 50,
    min_touches: int = 2
) -> Dict[str, Any]:
    """
    Detecta piscinas de liquidez basadas en Swing Highs/Lows no mitigados.
    
    Args:
        candles: DataFrame con columnas ['high', 'low', 'close', 'time']
        lookback: Nmero de velas a analizar
        min_touches: Mnimo de toques para considerar zona vlida
        
    Returns:
        Dict con liquidity_above, liquidity_below, y pools list
    """
    if len(candles) < lookback:
        return {"liquidity_above": 0, "liquidity_below": 0, "pools": []}
    
    recent = candles.tail(lookback).copy()
    current_price = candles.iloc[-1]["close"]
    
    pools = []
    
    # Detect Swing Highs (potential sell-side liquidity)
    for i in range(2, len(recent) - 2):
        if (recent.iloc[i]["high"] > recent.iloc[i-1]["high"] and 
            recent.iloc[i]["high"] > recent.iloc[i-2]["high"] and
            recent.iloc[i]["high"] > recent.iloc[i+1]["high"] and 
            recent.iloc[i]["high"] > recent.iloc[i+2]["high"]):
            
            swing_high = recent.iloc[i]["high"]
            # Check if NOT mitigated (price hasn't gone above)
            future_candles = recent.iloc[i+1:]
            if not any(future_candles["high"] > swing_high):
                pools.append({
                    "type": "SELL_SIDE",
                    "price": swing_high,
                    "time": recent.iloc[i].get("time", ""),
                    "distance_pips": abs(swing_high - current_price)
                })
    
    # Detect Swing Lows (potential buy-side liquidity)
    for i in range(2, len(recent) - 2):
        if (recent.iloc[i]["low"] < recent.iloc[i-1]["low"] and 
            recent.iloc[i]["low"] < recent.iloc[i-2]["low"] and
            recent.iloc[i]["low"] < recent.iloc[i+1]["low"] and 
            recent.iloc[i]["low"] < recent.iloc[i+2]["low"]):
            
            swing_low = recent.iloc[i]["low"]
            # Check if NOT mitigated
            future_candles = recent.iloc[i+1:]
            if not any(future_candles["low"] < swing_low):
                pools.append({
                    "type": "BUY_SIDE",
                    "price": swing_low,
                    "time": recent.iloc[i].get("time", ""),
                    "distance_pips": abs(current_price - swing_low)
                })
    
    # Closest pools
    above = [p for p in pools if p["price"] > current_price]
    below = [p for p in pools if p["price"] < current_price]
    
    nearest_above = min(above, key=lambda x: x["distance_pips"]) if above else None
    nearest_below = min(below, key=lambda x: x["distance_pips"]) if below else None
    
    return {
        "liquidity_above": nearest_above["price"] if nearest_above else 0,
        "liquidity_below": nearest_below["price"] if nearest_below else 0,
        "pools": pools,
        "total_pools": len(pools)
    }


def detect_asian_sweep(
    candles: pd.DataFrame,
    asian_start_hour: int = 20,
    asian_end_hour: int = 4,
    utc_offset: int = -4
) -> Dict[str, Any]:
    """
    Detecta si el precio ha barrido (sweep) el rango de la sesión asiática.
    
    Un sweep ocurre cuando el precio rompe el High/Low de Asia y 
    luego vuelve a entrar en el rango con fuerza.
    """
    if len(candles) < 50:
        return {"asian_sweep": False, "sweep_type": None}
    
    # Ensure 'time' column is datetime
    df = candles.copy()
    if 'time' in df.columns:
        if df['time'].dtype == 'object' or df['time'].dtype == 'O':
            df['time'] = pd.to_datetime(df['time'])
        elif not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'], unit='s')
    else:
        return {"asian_sweep": False, "sweep_type": None, "asian_high": 0, "asian_low": 0}
    
    # Extract hour from time
    df['hour'] = df['time'].dt.hour
    
    # Filter Asian session candles (handles midnight wrap: 20:00 - 04:00)
    if asian_start_hour > asian_end_hour:
        asian_candles = df[(df['hour'] >= asian_start_hour) | (df['hour'] < asian_end_hour)]
    else:
        asian_candles = df[(df['hour'] >= asian_start_hour) & (df['hour'] < asian_end_hour)]
    
    if len(asian_candles) < 3:
        return {"asian_sweep": False, "sweep_type": None, "asian_high": 0, "asian_low": 0}
    
    # Calculate Asian range
    asian_high = float(asian_candles['high'].max())
    asian_low = float(asian_candles['low'].min())
    
    # Get post-Asian candles (typically NY session)
    last_asian_time = asian_candles['time'].max()
    post_asian = df[df['time'] > last_asian_time].tail(20)
    
    if len(post_asian) < 3:
        return {"asian_sweep": False, "sweep_type": None, "asian_high": asian_high, "asian_low": asian_low}
    
    # Check for sweeps
    current_price = float(post_asian['close'].iloc[-1])
    swept_high = float(post_asian['high'].max()) > asian_high
    swept_low = float(post_asian['low'].min()) < asian_low
    
    # Price returned to range after sweep?
    in_range = asian_low <= current_price <= asian_high
    
    sweep_type = None
    asian_sweep = False
    
    if swept_high and in_range:
        sweep_type = "BEARISH_SWEEP"  # Price took highs and rejected
        asian_sweep = True
    elif swept_low and in_range:
        sweep_type = "BULLISH_SWEEP"  # Price took lows and rejected
        asian_sweep = True
    
    return {
        "asian_sweep": asian_sweep,
        "sweep_type": sweep_type,
        "asian_high": asian_high,
        "asian_low": asian_low,
        "current_in_range": in_range
    }


def detect_fvg(
    candles: pd.DataFrame,
    lookback: int = 20
) -> List[Dict[str, Any]]:
    """
    Detecta Fair Value Gaps (FVG) - zonas de ineficiencia de precio.
    
    Bullish FVG: Candle1.high < Candle3.low (gap alcista)
    Bearish FVG: Candle1.low > Candle3.high (gap bajista)
    """
    if len(candles) < 3:
        return []
    
    fvgs = []
    recent = candles.tail(lookback)
    
    for i in range(len(recent) - 2):
        c1 = recent.iloc[i]
        c3 = recent.iloc[i + 2]
        
        # Bullish FVG
        if c1["high"] < c3["low"]:
            fvgs.append({
                "type": "BULLISH",
                "top": c3["low"],
                "bottom": c1["high"],
                "midpoint": (c3["low"] + c1["high"]) / 2,
                "index": i
            })
        
        # Bearish FVG  
        if c1["low"] > c3["high"]:
            fvgs.append({
                "type": "BEARISH",
                "top": c1["low"],
                "bottom": c3["high"],
                "midpoint": (c1["low"] + c3["high"]) / 2,
                "index": i
            })
    
    return fvgs


def calculate_body_wick_ratio(candle: Dict[str, float]) -> float:
    """
    Calcula el ratio Cuerpo/Mecha de una vela.
    Velas de intencin tienen > 70% cuerpo.
    """
    high, low = candle["high"], candle["low"]
    open_, close = candle["open"], candle["close"]
    
    total_range = high - low
    if total_range == 0:
        return 0.0
    
    body = abs(close - open_)
    return (body / total_range) * 100


def is_intention_candle(candle: Dict[str, float], threshold: float = 70.0) -> bool:
    """
    Determina si una vela es de "intencin" (cuerpo > 70% del rango).
    """
    return calculate_body_wick_ratio(candle) >= threshold


# =============================================================================
# ORDER BLOCK DETECTION (from CLiquidity.mqh)
# =============================================================================

@dataclass
class OrderBlock:
    """Institutional Order Block zone."""
    zone_type: str              # "BULLISH_OB" or "BEARISH_OB"
    top: float                  # Zone top price
    bottom: float               # Zone bottom price
    midpoint: float             # Zone midpoint
    time_index: int             # Bar index where OB formed
    strength: float             # 0.0-1.0 strength score
    mitigated: bool = False     # Has price returned to zone?
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.zone_type,
            "top": self.top,
            "bottom": self.bottom,
            "midpoint": self.midpoint,
            "strength": round(self.strength, 3),
            "mitigated": self.mitigated
        }


@dataclass
class SpaceConfidence:
    """Probabilistic result for Space/Liquidity analysis."""
    fvg_confidence: float = 0.0        # 0.0-1.0: Valid FVG present
    ob_confidence: float = 0.0         # 0.0-1.0: Price at OrderBlock
    liquidity_confidence: float = 0.0  # 0.0-1.0: Near liquidity pool
    confluence_confidence: float = 0.0 # 0.0-1.0: Multiple zones overlap
    
    overall_space_score: float = 0.0   # Weighted combination
    reasoning: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fvg_confidence": round(self.fvg_confidence, 3),
            "ob_confidence": round(self.ob_confidence, 3),
            "liquidity_confidence": round(self.liquidity_confidence, 3),
            "confluence_confidence": round(self.confluence_confidence, 3),
            "overall_space_score": round(self.overall_space_score, 3),
            "reasoning": self.reasoning
        }


def detect_order_blocks(
    candles: pd.DataFrame,
    lookback: int = 50,
    impulse_multiplier: float = 2.5
) -> List[OrderBlock]:
    """
    Detect Order Blocks - Last opposite candle before institutional move.
    
    From CLiquidity.mqh logic:
    - Bullish OB: Bearish candle → Strong bullish impulse (>2.5x range)
    - Bearish OB: Bullish candle → Strong bearish impulse (>2.5x range)
    
    Args:
        candles: DataFrame with OHLC columns
        lookback: Number of bars to scan
        impulse_multiplier: Minimum impulse size as multiple of prior range
        
    Returns:
        List of OrderBlock objects
    """
    if len(candles) < lookback:
        return []
    
    recent = candles.tail(lookback).reset_index(drop=True)
    order_blocks = []
    
    # Calculate ATR for normalization
    atr = (recent["high"] - recent["low"]).rolling(14).mean()
    
    for i in range(1, len(recent) - 1):
        curr = recent.iloc[i]
        next_candle = recent.iloc[i + 1]
        
        curr_range = curr["high"] - curr["low"]
        next_range = next_candle["high"] - next_candle["low"]
        
        # Skip if current candle range is too small
        if curr_range < (atr.iloc[i] * 0.2 if not pd.isna(atr.iloc[i]) else 0.001):
            continue
        
        # Current candle direction
        curr_bullish = curr["close"] > curr["open"]
        curr_bearish = curr["close"] < curr["open"]
        
        # Next candle direction and magnitude
        next_bullish = next_candle["close"] > next_candle["open"]
        next_bearish = next_candle["close"] < next_candle["open"]
        
        # Check for impulse (next candle much larger than current)
        is_impulse = next_range > (curr_range * impulse_multiplier)
        
        # Bullish OB: Bearish candle followed by strong bullish impulse
        if curr_bearish and next_bullish and is_impulse:
            # Check if NOT mitigated (price hasn't returned below)
            future = recent.iloc[i + 2:] if i + 2 < len(recent) else pd.DataFrame()
            mitigated = len(future) > 0 and future["low"].min() < curr["low"]
            
            strength = min(1.0, (next_range / curr_range) / 5.0)  # Normalize strength
            
            order_blocks.append(OrderBlock(
                zone_type="BULLISH_OB",
                top=curr["high"],
                bottom=curr["low"],
                midpoint=(curr["high"] + curr["low"]) / 2,
                time_index=i,
                strength=strength,
                mitigated=mitigated
            ))
        
        # Bearish OB: Bullish candle followed by strong bearish impulse
        if curr_bullish and next_bearish and is_impulse:
            future = recent.iloc[i + 2:] if i + 2 < len(recent) else pd.DataFrame()
            mitigated = len(future) > 0 and future["high"].max() > curr["high"]
            
            strength = min(1.0, (next_range / curr_range) / 5.0)
            
            order_blocks.append(OrderBlock(
                zone_type="BEARISH_OB",
                top=curr["high"],
                bottom=curr["low"],
                midpoint=(curr["high"] + curr["low"]) / 2,
                time_index=i,
                strength=strength,
                mitigated=mitigated
            ))
    
    # Return only unmitigated order blocks, sorted by recency
    return [ob for ob in order_blocks if not ob.mitigated][-5:]  # Last 5 valid OBs


# =============================================================================
# CONFLUENCE ZONE DETECTION (from CLiquidity.mqh)
# =============================================================================

@dataclass
class ConfluenceZone:
    """Zone where multiple signals overlap."""
    top: float
    bottom: float
    zone_types: List[str]       # What overlaps: ["FVG", "OB", "LIQUIDITY"]
    direction: str              # "BULLISH" or "BEARISH"
    strength: float             # 0.0-2.0 (confluence adds strength)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "top": self.top,
            "bottom": self.bottom,
            "overlapping": self.zone_types,
            "direction": self.direction,
            "strength": round(self.strength, 3)
        }


def detect_confluence_zones(
    candles: pd.DataFrame,
    atr_tolerance: float = 0.5
) -> List[ConfluenceZone]:
    """
    Detect zones where multiple signals overlap (FVG + OB, OB + Liquidity, etc.).
    
    Confluence zones have 2x strength multiplier.
    
    Args:
        candles: DataFrame with OHLC
        atr_tolerance: ATR multiplier for considering zones "overlapping"
        
    Returns:
        List of ConfluenceZone objects
    """
    if len(candles) < 20:
        return []
    
    # Collect all zones
    fvgs = detect_fvg(candles, lookback=30)
    order_blocks = detect_order_blocks(candles, lookback=50)
    liquidity = detect_liquidity_pools(candles, lookback=50)
    
    # Calculate ATR for tolerance
    atr = (candles["high"] - candles["low"]).rolling(14).mean().iloc[-1]
    tolerance = atr * atr_tolerance
    
    all_zones = []
    
    # Convert FVGs to standard zone format
    for fvg in fvgs:
        all_zones.append({
            "type": "FVG",
            "top": fvg["top"],
            "bottom": fvg["bottom"],
            "direction": fvg["type"]
        })
    
    # Convert OBs to standard zone format
    for ob in order_blocks:
        all_zones.append({
            "type": "OB",
            "top": ob.top,
            "bottom": ob.bottom,
            "direction": "BULLISH" if "BULLISH" in ob.zone_type else "BEARISH"
        })
    
    # Convert liquidity pools to zones
    for pool in liquidity.get("pools", []):
        pool_price = pool["price"]
        all_zones.append({
            "type": "LIQUIDITY",
            "top": pool_price + tolerance * 0.5,
            "bottom": pool_price - tolerance * 0.5,
            "direction": "BULLISH" if pool["type"] == "BUY_SIDE" else "BEARISH"
        })
    
    # Find overlapping zones
    confluences = []
    processed = set()
    
    for i, zone1 in enumerate(all_zones):
        if i in processed:
            continue
            
        overlapping_types = [zone1["type"]]
        merged_top = zone1["top"]
        merged_bottom = zone1["bottom"]
        direction = zone1["direction"]
        
        for j, zone2 in enumerate(all_zones):
            if i == j or j in processed:
                continue
            
            # Check if zones overlap (with tolerance)
            overlap_top = min(zone1["top"], zone2["top"])
            overlap_bottom = max(zone1["bottom"], zone2["bottom"])
            
            if overlap_top >= overlap_bottom - tolerance:
                # Zones overlap!
                if zone1["direction"] == zone2["direction"]:
                    overlapping_types.append(zone2["type"])
                    merged_top = max(merged_top, zone2["top"])
                    merged_bottom = min(merged_bottom, zone2["bottom"])
                    processed.add(j)
        
        # Only create confluence if 2+ zones overlap
        if len(overlapping_types) >= 2:
            processed.add(i)
            base_strength = len(overlapping_types) / 3.0  # More overlaps = stronger
            confluences.append(ConfluenceZone(
                top=merged_top,
                bottom=merged_bottom,
                zone_types=overlapping_types,
                direction=direction,
                strength=min(2.0, base_strength + 0.5)  # Confluence bonus
            ))
    
    return confluences


def compute_space_confidence(
    candles: pd.DataFrame,
    current_price: float
) -> SpaceConfidence:
    """
    Compute probabilistic confidence for Space/Liquidity component.
    
    Returns SpaceConfidence with individual and overall scores.
    """
    result = SpaceConfidence()
    
    if len(candles) < 20:
        result.reasoning.append("Insufficient data for analysis")
        return result
    
    atr = (candles["high"] - candles["low"]).rolling(14).mean().iloc[-1]
    
    # 1. FVG Confidence
    fvgs = detect_fvg(candles, lookback=20)
    if fvgs:
        # Check if price is near any FVG
        for fvg in fvgs:
            if fvg["bottom"] <= current_price <= fvg["top"]:
                result.fvg_confidence = 0.9
                result.reasoning.append(f"Price INSIDE {fvg['type']} FVG")
                break
            
            dist = min(abs(current_price - fvg["top"]), abs(current_price - fvg["bottom"]))
            if dist < atr * 1.5:
                result.fvg_confidence = max(result.fvg_confidence, 0.6)
                result.reasoning.append(f"Price NEAR {fvg['type']} FVG ({dist:.4f})")
    
    # 2. OrderBlock Confidence
    obs = detect_order_blocks(candles, lookback=50)
    if obs:
        for ob in obs:
            if ob.bottom <= current_price <= ob.top:
                result.ob_confidence = 0.85 * ob.strength
                result.reasoning.append(f"Price INSIDE {ob.zone_type}")
                break
            
            dist = min(abs(current_price - ob.top), abs(current_price - ob.bottom))
            if dist < atr * 2.0:
                result.ob_confidence = max(result.ob_confidence, 0.5 * ob.strength)
                result.reasoning.append(f"Price NEAR {ob.zone_type} ({dist:.4f})")
    
    # 3. Liquidity Confidence
    liq = detect_liquidity_pools(candles, lookback=50)
    liq_above = liq.get("liquidity_above", 0)
    liq_below = liq.get("liquidity_below", 0)
    
    if liq_above > 0:
        dist = liq_above - current_price
        if dist < atr * 3.0:
            result.liquidity_confidence = min(0.8, (atr * 3.0 - dist) / (atr * 3.0))
            result.reasoning.append(f"Liquidity above at {liq_above:.5f}")
    
    if liq_below > 0:
        dist = current_price - liq_below
        if dist < atr * 3.0:
            conf = min(0.8, (atr * 3.0 - dist) / (atr * 3.0))
            if conf > result.liquidity_confidence:
                result.liquidity_confidence = conf
                result.reasoning.append(f"Liquidity below at {liq_below:.5f}")
    
    # 4. Confluence Confidence
    confluences = detect_confluence_zones(candles)
    if confluences:
        for conf_zone in confluences:
            if conf_zone.bottom <= current_price <= conf_zone.top:
                result.confluence_confidence = min(1.0, conf_zone.strength / 2.0)
                result.reasoning.append(f"CONFLUENCE: {'+'.join(conf_zone.zone_types)}")
                break
    
    # Calculate overall score (weighted average)
    weights = {"fvg": 0.25, "ob": 0.30, "liquidity": 0.20, "confluence": 0.25}
    result.overall_space_score = (
        weights["fvg"] * result.fvg_confidence +
        weights["ob"] * result.ob_confidence +
        weights["liquidity"] * result.liquidity_confidence +
        weights["confluence"] * result.confluence_confidence
    )
    
    return result


# =============================================================================
# MARKET STATE TENSOR BUILDER
# =============================================================================

class MarketStateTensor:
    """
    Construye un tensor de estado de mercado multidimensional.
    Capas: Macro (H4/D1), Structural (H1/M15), Execution (M5/M1)
    """
    
    def __init__(self):
        self.macro_data = {}
        self.structural_data = {}
        self.execution_data = {}
    
    def build_tensor(
        self,
        h4_candles: pd.DataFrame,
        h1_candles: pd.DataFrame,
        m15_candles: pd.DataFrame,
        m5_candles: pd.DataFrame,
        m1_candles: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Construye el tensor completo de estado de mercado.
        """
        # Macro Layer (H4/D1) - The Judge
        macro = self._build_macro_layer(h4_candles)
        
        # Structural Layer (H1/M15) - The Filter
        structural = self._build_structural_layer(h1_candles, m15_candles)
        
        # Execution Layer (M5/M1) - The Trigger
        execution = self._build_execution_layer(m5_candles, m1_candles)
        
        return {
            "macro": macro,
            "structural": structural,
            "execution": execution,
            "alignment_score": self._calculate_alignment(macro, structural, execution),
            "kill_zone": get_current_kill_zone(),
            "in_ny_kz": is_in_kill_zone("NY")
        }
    
    def _build_macro_layer(self, h4_candles: pd.DataFrame) -> Dict[str, Any]:
        """H4/D1: Bias y Espacio (POI)"""
        if len(h4_candles) < 10:
            return {"H4_Trend": "NEUTRAL", "H4_Break_Valid": False}
        
        last = h4_candles.iloc[-1]
        prev = h4_candles.iloc[-2]
        
        # EMA-based trend
        close = last["close"]
        ema20 = h4_candles["close"].tail(20).mean()
        ema50 = h4_candles["close"].tail(50).mean() if len(h4_candles) >= 50 else ema20
        
        if close > ema20 > ema50:
            trend = "BULLISH"
        elif close < ema20 < ema50:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"
        
        # H4 Body Close Break
        body_close = abs(last["close"] - last["open"]) / (last["high"] - last["low"]) > 0.5
        break_high = last["close"] > prev["high"]
        break_low = last["close"] < prev["low"]
        h4_break_valid = body_close and (break_high or break_low)
        
        # Liquidity
        liq = detect_liquidity_pools(h4_candles)
        
        return {
            "H4_Trend": trend,
            "H4_Break_Valid": h4_break_valid,
            "H4_RSI": self._simple_rsi(h4_candles["close"].tail(14)),
            "H4_Liq_Above": liq["liquidity_above"],
            "H4_Liq_Below": liq["liquidity_below"]
        }
    
    def _build_structural_layer(
        self, 
        h1_candles: pd.DataFrame, 
        m15_candles: pd.DataFrame
    ) -> Dict[str, Any]:
        """H1/M15: Forma y Sincrona (CHoCH, W/M patterns)"""
        if len(m15_candles) < 10:
            return {"M15_Structure": "UNKNOWN", "M15_CHoCH": False}
        
        # Detect Change of Character (CHoCH)
        last_3 = m15_candles.tail(3)
        prev_trend = "UP" if last_3.iloc[0]["close"] < last_3.iloc[1]["close"] else "DOWN"
        curr_move = "UP" if last_3.iloc[-1]["close"] > last_3.iloc[-2]["close"] else "DOWN"
        choch = prev_trend != curr_move
        
        # RSI alignment with H4
        m15_rsi = self._simple_rsi(m15_candles["close"].tail(14))
        
        return {
            "M15_Structure": curr_move,
            "M15_CHoCH": choch,
            "M15_RSI": m15_rsi,
            "H1_FVG_Count": len(detect_fvg(h1_candles))
        }
    
    def _build_execution_layer(
        self, 
        m5_candles: pd.DataFrame, 
        m1_candles: pd.DataFrame
    ) -> Dict[str, Any]:
        """M5/M1: Aceleracin y Tiempo (Volume, Intention)"""
        if len(m1_candles) < 5:
            return {"M1_Acceleration": 0, "M1_Intention": False}
        
        last_m1 = m1_candles.iloc[-1].to_dict()
        
        # Body/Wick Ratio
        acceleration = calculate_body_wick_ratio(last_m1)
        intention = is_intention_candle(last_m1)
        
        # Relative Volume
        avg_vol = m1_candles["volume"].tail(20).mean()
        curr_vol = last_m1.get("volume", 0)
        rel_volume = curr_vol / avg_vol if avg_vol > 0 else 1.0
        
        return {
            "M1_Acceleration": acceleration,
            "M1_Intention": intention,
            "M1_Rel_Volume": rel_volume,
            "M5_Trend": "UP" if m5_candles.iloc[-1]["close"] > m5_candles.iloc[-2]["close"] else "DOWN"
        }
    
    def _calculate_alignment(
        self, 
        macro: Dict, 
        structural: Dict, 
        execution: Dict
    ) -> float:
        """
        Calcula el score de alineacin entre capas (0-100).
        100 = Todas las capas alineadas en la misma direccin.
        """
        score = 0
        
        # Macro-Structural alignment
        h4_trend = macro.get("H4_Trend", "NEUTRAL")
        m15_struct = structural.get("M15_Structure", "UNKNOWN")
        
        if h4_trend == "BULLISH" and m15_struct == "UP":
            score += 40
        elif h4_trend == "BEARISH" and m15_struct == "DOWN":
            score += 40
        
        # Execution alignment
        m5_trend = execution.get("M5_Trend", "")
        if (h4_trend == "BULLISH" and m5_trend == "UP") or \
           (h4_trend == "BEARISH" and m5_trend == "DOWN"):
            score += 30
        
        # Kill Zone bonus
        if execution.get("M1_Intention") and is_in_kill_zone("NY"):
            score += 20
        
        # Volume bonus
        if execution.get("M1_Rel_Volume", 1) > 2.0:
            score += 10
        
        return min(100, score)
    
    def _simple_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Clculo simple de RSI."""
        if len(prices) < period:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
        return 100 - (100 / (1 + rs))


# Singleton
market_tensor = MarketStateTensor()
