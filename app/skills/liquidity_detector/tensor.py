import pandas as pd
from typing import Dict, Any
from .sessions import get_current_kill_zone, is_in_kill_zone
from .detector import detect_liquidity_pools, detect_fvg, calculate_body_wick_ratio, is_intention_candle

class MarketStateTensor:
    """
    Construye un tensor de estado de mercado multidimensional.
    Capas: Macro (H4/D1), Structural (H1/M15), Execution (M5/M1)
    """
    def __init__(self):
        self.macro_data = {}
        self.structural_data = {}
        self.execution_data = {}
    
    def build_tensor(self, h4: pd.DataFrame, h1: pd.DataFrame, m15: pd.DataFrame, m5: pd.DataFrame, m1: pd.DataFrame) -> Dict[str, Any]:
        macro = self._build_macro_layer(h4)
        structural = self._build_structural_layer(h1, m15)
        execution = self._build_execution_layer(m5, m1)
        return {
            "macro": macro, "structural": structural, "execution": execution,
            "alignment_score": self._calculate_alignment(macro, structural, execution),
            "kill_zone": get_current_kill_zone(),
            "in_ny_kz": is_in_kill_zone("NY")
        }
    
    def _build_macro_layer(self, df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < 10: return {"H4_Trend": "NEUTRAL"}
        last, prev = df.iloc[-1], df.iloc[-2]
        c, ema20 = last["close"], df["close"].tail(20).mean()
        ema50 = df["close"].tail(50).mean() if len(df) >= 50 else ema20
        trend = "BULLISH" if c > ema20 > ema50 else "BEARISH" if c < ema20 < ema50 else "NEUTRAL"
        liq = detect_liquidity_pools(df)
        return {"H4_Trend": trend, "H4_Break_Valid": abs(last["close"]-last["open"])/(last["high"]-last["low"]) > 0.5 and (last["close"]>prev["high"] or last["close"]<prev["low"]),
                "H4_RSI": self._simple_rsi(df["close"].tail(14)), "H4_Liq_Above": liq["liquidity_above"], "H4_Liq_Below": liq["liquidity_below"]}
    
    def _build_structural_layer(self, h1: pd.DataFrame, m15: pd.DataFrame) -> Dict[str, Any]:
        if len(m15) < 10: return {"M15_CHoCH": False}
        last_3 = m15.tail(3)
        curr = "UP" if last_3.iloc[-1]["close"] > last_3.iloc[-2]["close"] else "DOWN"
        return {"M15_Structure": curr, "M15_CHoCH": ("UP" if last_3.iloc[0]["close"] < last_3.iloc[1]["close"] else "DOWN") != curr,
                "M15_RSI": self._simple_rsi(m15["close"].tail(14)), "H1_FVG_Count": len(detect_fvg(h1))}
    
    def _build_execution_layer(self, m5: pd.DataFrame, m1: pd.DataFrame) -> Dict[str, Any]:
        if len(m1) < 5: return {"M1_Intention": False}
        last_m1 = m1.iloc[-1].to_dict()
        rel_vol = last_m1.get("volume", 0) / (m1["volume"].tail(20).mean() or 1.0)
        return {"M1_Acceleration": calculate_body_wick_ratio(last_m1), "M1_Intention": is_intention_candle(last_m1),
                "M1_Rel_Volume": rel_vol, "M5_Trend": "UP" if m5.iloc[-1]["close"] > m5.iloc[-2]["close"] else "DOWN"}

    def _calculate_alignment(self, macro: Dict, structural: Dict, execution: Dict) -> float:
        score = 0
        h4, m15, m5 = macro.get("H4_Trend"), structural.get("M15_Structure"), execution.get("M5_Trend")
        if (h4 == "BULLISH" and m15 == "UP") or (h4 == "BEARISH" and m15 == "DOWN"): score += 40
        if (h4 == "BULLISH" and m5 == "UP") or (h4 == "BEARISH" and m5 == "DOWN"): score += 30
        if execution.get("M1_Intention") and is_in_kill_zone("NY"): score += 20
        if execution.get("M1_Rel_Volume", 1) > 2.0: score += 10
        return float(score)

    def _simple_rsi(self, prices: pd.Series, period: int = 14) -> float:
        if len(prices) < period: return 50.0
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        return float((100 - (100 / (1 + rs))).iloc[-1])
