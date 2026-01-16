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
import pandas as pd
import numpy as np

logger = logging.getLogger("FEAT.LiquidityIntelligence")

# Kill Zone Definitions (UTC-4 / Bolivia-Colombia time)
KILL_ZONES = {
    "NY": {"start": 7, "end": 11},      # New York Open
    "LONDON": {"start": 3, "end": 5},   # London Open
    "ASIA": {"start": 20, "end": 0},    # Asia (wraps midnight)
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


def is_in_kill_zone(zone: str = "NY", utc_offset: int = -4) -> bool:
    """
    Verifica si estamos dentro de una Kill Zone especfica.
    """
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
