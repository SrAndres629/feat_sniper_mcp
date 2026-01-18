import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from .models import OrderBlock, SpaceConfidence, ConfluenceZone

logger = logging.getLogger("FEAT.Liquidity.Detector")

def detect_liquidity_pools(
    candles: pd.DataFrame,
    lookback: int = 50,
    min_touches: int = 2
) -> Dict[str, Any]:
    """Detecta piscinas de liquidez basadas en Swing Highs/Lows no mitigados."""
    if len(candles) < lookback:
        return {"liquidity_above": 0, "liquidity_below": 0, "pools": []}
    
    recent = candles.tail(lookback).copy()
    current_price = candles.iloc[-1]["close"]
    pools = []
    
    # Swing Highs
    for i in range(2, len(recent) - 2):
        if (recent.iloc[i]["high"] > recent.iloc[i-1]["high"] and 
            recent.iloc[i]["high"] > recent.iloc[i-2]["high"] and
            recent.iloc[i]["high"] > recent.iloc[i+1]["high"] and 
            recent.iloc[i]["high"] > recent.iloc[i+2]["high"]):
            
            swing_high = recent.iloc[i]["high"]
            future_candles = recent.iloc[i+1:]
            if not any(future_candles["high"] > swing_high):
                pools.append({
                    "type": "SELL_SIDE", "price": swing_high,
                    "time": recent.iloc[i].get("time", ""),
                    "distance_pips": abs(swing_high - current_price)
                })
    
    # Swing Lows
    for i in range(2, len(recent) - 2):
        if (recent.iloc[i]["low"] < recent.iloc[i-1]["low"] and 
            recent.iloc[i]["low"] < recent.iloc[i-2]["low"] and
            recent.iloc[i]["low"] < recent.iloc[i+1]["low"] and 
            recent.iloc[i]["low"] < recent.iloc[i+2]["low"]):
            
            swing_low = recent.iloc[i]["low"]
            future_candles = recent.iloc[i+1:]
            if not any(future_candles["low"] < swing_low):
                pools.append({
                    "type": "BUY_SIDE", "price": swing_low,
                    "time": recent.iloc[i].get("time", ""),
                    "distance_pips": abs(current_price - swing_low)
                })
    
    above = [p for p in pools if p["price"] > current_price]
    below = [p for p in pools if p["price"] < current_price]
    nearest_above = min(above, key=lambda x: x["distance_pips"]) if above else None
    nearest_below = min(below, key=lambda x: x["distance_pips"]) if below else None
    
    return {
        "liquidity_above": nearest_above["price"] if nearest_above else 0,
        "liquidity_below": nearest_below["price"] if nearest_below else 0,
        "pools": pools, "total_pools": len(pools)
    }

def detect_fvg(candles: pd.DataFrame, lookback: int = 20) -> List[Dict[str, Any]]:
    """Detecta Fair Value Gaps (FVG)."""
    if len(candles) < 3: return []
    fvgs = []
    recent = candles.tail(lookback)
    for i in range(len(recent) - 2):
        c1, c3 = recent.iloc[i], recent.iloc[i + 2]
        if c1["high"] < c3["low"]:
            fvgs.append({"type": "BULLISH", "top": c3["low"], "bottom": c1["high"], "midpoint": (c3["low"] + c1["high"]) / 2, "index": i})
        if c1["low"] > c3["high"]:
            fvgs.append({"type": "BEARISH", "top": c1["low"], "bottom": c3["high"], "midpoint": (c1["low"] + c3["high"]) / 2, "index": i})
    return fvgs

def detect_order_blocks(candles: pd.DataFrame, lookback: int = 50, impulse_multiplier: float = 2.5) -> List[OrderBlock]:
    """Detect Order Blocks."""
    if len(candles) < lookback: return []
    recent = candles.tail(lookback).reset_index(drop=True)
    order_blocks = []
    atr = (recent["high"] - recent["low"]).rolling(14).mean()
    
    for i in range(1, len(recent) - 1):
        curr, next_c = recent.iloc[i], recent.iloc[i + 1]
        curr_range, next_range = curr["high"] - curr["low"], next_c["high"] - next_c["low"]
        if curr_range < (atr.iloc[i] * 0.2 if not pd.isna(atr.iloc[i]) else 0.001): continue
        
        is_impulse = next_range > (curr_range * impulse_multiplier)
        if curr["close"] < curr["open"] and next_c["close"] > next_c["open"] and is_impulse:
            future = recent.iloc[i + 2:]
            mitigated = len(future) > 0 and future["low"].min() < curr["low"]
            order_blocks.append(OrderBlock("BULLISH_OB", curr["high"], curr["low"], (curr["high"]+curr["low"])/2, i, min(1.0, (next_range/curr_range)/5.0), mitigated))
        if curr["close"] > curr["open"] and next_c["close"] < next_c["open"] and is_impulse:
            future = recent.iloc[i + 2:]
            mitigated = len(future) > 0 and future["high"].max() > curr["high"]
            order_blocks.append(OrderBlock("BEARISH_OB", curr["high"], curr["low"], (curr["high"]+curr["low"])/2, i, min(1.0, (next_range/curr_range)/5.0), mitigated))
            
    return [ob for ob in order_blocks if not ob.mitigated][-5:]

def detect_confluence_zones(candles: pd.DataFrame, atr_tolerance: float = 0.5) -> List[ConfluenceZone]:
    """Detect zones where multiple signals overlap."""
    if len(candles) < 20: return []
    fvgs, obs, liq = detect_fvg(candles, 30), detect_order_blocks(candles, 50), detect_liquidity_pools(candles, 50)
    atr = (candles["high"] - candles["low"]).rolling(14).mean().iloc[-1]
    tolerance = atr * atr_tolerance
    
    all_zones = []
    for f in fvgs: all_zones.append({"type": "FVG", "top": f["top"], "bottom": f["bottom"], "direction": f["type"]})
    for o in obs: all_zones.append({"type": "OB", "top": o.top, "bottom": o.bottom, "direction": "BULLISH" if "BULLISH" in o.zone_type else "BEARISH"})
    for p in liq.get("pools", []):
        all_zones.append({"type": "LIQUIDITY", "top": p["price"] + tolerance*0.5, "bottom": p["price"] - tolerance*0.5, "direction": "BULLISH" if p["type"] == "BUY_SIDE" else "BEARISH"})

    confluences, processed = [], set()
    for i, z1 in enumerate(all_zones):
        if i in processed: continue
        types, mt, mb, d = [z1["type"]], z1["top"], z1["bottom"], z1["direction"]
        for j, z2 in enumerate(all_zones):
            if i == j or j in processed: continue
            if min(mt, z2["top"]) >= max(mb, z2["bottom"]) - tolerance and d == z2["direction"]:
                types.append(z2["type"]); mt = max(mt, z2["top"]); mb = min(mb, z2["bottom"]); processed.add(j)
        if len(types) >= 2:
            processed.add(i); confluences.append(ConfluenceZone(mt, mb, types, d, min(2.0, len(types)/3.0 + 0.5)))
    return confluences

def compute_space_confidence(candles: pd.DataFrame, current_price: float) -> SpaceConfidence:
    """Compute probabilistic confidence for Space/Liquidity component."""
    res = SpaceConfidence()
    if len(candles) < 20: return res
    atr = (candles["high"] - candles["low"]).rolling(14).mean().iloc[-1]
    
    fvgs = detect_fvg(candles, 20)
    for f in fvgs:
        dist = min(abs(current_price - f["top"]), abs(current_price - f["bottom"]))
        if f["bottom"] <= current_price <= f["top"]: res.fvg_confidence = 0.9; break
        if dist < atr * 1.5: res.fvg_confidence = max(res.fvg_confidence, 0.6)
            
    obs = detect_order_blocks(candles, 50)
    for o in obs:
        dist = min(abs(current_price - o.top), abs(current_price - o.bottom))
        if o.bottom <= current_price <= o.top: res.ob_confidence = 0.85 * o.strength; break
        if dist < atr * 2.0: res.ob_confidence = max(res.ob_confidence, 0.5 * o.strength)
            
    liq = detect_liquidity_pools(candles, 50)
    for p in ["liquidity_above", "liquidity_below"]:
        lp = liq.get(p, 0)
        if lp > 0:
            dist = abs(lp - current_price)
            if dist < atr * 3.0: res.liquidity_confidence = max(res.liquidity_confidence, min(0.8, (atr*3 - dist)/(atr*3)))
            
    confs = detect_confluence_zones(candles)
    for c in confs:
        if c.bottom <= current_price <= c.top: res.confluence_confidence = min(1.0, c.strength / 2.0); break
            
    w = {"fvg": 0.25, "ob": 0.30, "liquidity": 0.20, "confluence": 0.25}
    res.overall_space_score = (w["fvg"]*res.fvg_confidence + w["ob"]*res.ob_confidence + w["liquidity"]*res.liquidity_confidence + w["confluence"]*res.confluence_confidence)
    return res

def detect_asian_sweep(candles: pd.DataFrame, asian_start_hour: int = 20, asian_end_hour: int = 4, utc_offset: int = -4) -> Dict[str, Any]:
    """Detecta si el precio ha barrido el rango de la sesión asiática."""
    if len(candles) < 50: return {"asian_sweep": False}
    df = candles.copy()
    if 'time' not in df.columns: return {"asian_sweep": False}
    df['time'] = pd.to_datetime(df['time'], unit='s') if not pd.api.types.is_datetime64_any_dtype(df['time']) else df['time']
    df['hour'] = df['time'].dt.hour
    asian = df[(df['hour'] >= asian_start_hour) | (df['hour'] < asian_end_hour)] if asian_start_hour > asian_end_hour else df[(df['hour'] >= asian_start_hour) & (df['hour'] < asian_end_hour)]
    if len(asian) < 3: return {"asian_sweep": False}
    ah, al = float(asian['high'].max()), float(asian['low'].min())
    post = df[df['time'] > asian['time'].max()].tail(20)
    if len(post) < 3: return {"asian_sweep": False, "asian_high": ah, "asian_low": al}
    cp, sh, sl = float(post['close'].iloc[-1]), float(post['high'].max()) > ah, float(post['low'].min()) < al
    ir = al <= cp <= ah
    st = "BEARISH_SWEEP" if sh and ir else "BULLISH_SWEEP" if sl and ir else None
    return {"asian_sweep": st is not None, "sweep_type": st, "asian_high": ah, "asian_low": al, "current_in_range": ir}

def calculate_body_wick_ratio(candle: Dict[str, float]) -> float:
    tr = candle["high"] - candle["low"]
    return (abs(candle["close"] - candle["open"]) / tr * 100) if tr > 0 else 0.0

def is_intention_candle(candle: Dict[str, float], threshold: float = 70.0) -> bool:
    return calculate_body_wick_ratio(candle) >= threshold
