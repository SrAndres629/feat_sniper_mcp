from datetime import datetime, timezone
from typing import Dict, List, Tuple
from .constants import TIMEFRAMES, TIMEFRAME_MAP

class Resampler:
    """Engine for Multi-Timeframe (MTF) Candle Aggregation (Sub-1ms performance)."""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.history: Dict[str, List[Dict]] = {tf: [] for tf in TIMEFRAMES if tf != "M1"}
        self.current_candles: Dict[str, Dict] = {}

    def _parse_time(self, t) -> datetime:
        if isinstance(t, datetime): return t
        if isinstance(t, str):
            try: return datetime.fromisoformat(t.replace('Z', '+00:00'))
            except: return datetime.now(timezone.utc)
        return datetime.now(timezone.utc)
    
    def _floor_to_timeframe(self, dt: datetime, minutes: int) -> datetime:
        tm = dt.hour * 60 + dt.minute
        fm = (tm // minutes) * minutes
        return dt.replace(hour=fm//60, minute=fm%60, second=0, microsecond=0)

    def push_tick(self, m1: Dict) -> List[Tuple[str, Dict]]:
        completed = []
        t = self._parse_time(m1.get("tick_time"))
        for tf in [tf for tf in TIMEFRAMES if tf != "M1"]:
            mins = TIMEFRAME_MAP[tf]
            ws = self._floor_to_timeframe(t, mins)
            if tf not in self.current_candles or self.current_candles[tf]["time"] != ws:
                if tf in self.current_candles: completed.append((tf, self.current_candles[tf]))
                self.current_candles[tf] = {"time": ws, "tick_time": ws.isoformat(), "open": m1["open"], "high": m1["high"], "low": m1["low"], "close": m1["close"], "volume": m1["volume"]}
            else:
                curr = self.current_candles[tf]
                curr["high"] = max(curr["high"], m1["high"])
                curr["low"] = min(curr["low"], m1["low"])
                curr["close"] = m1["close"]
                curr["volume"] += m1["volume"]
        for tf, c in completed:
            self.history[tf].append(c)
            if len(self.history[tf]) > 200: self.history[tf].pop(0)
        return completed
