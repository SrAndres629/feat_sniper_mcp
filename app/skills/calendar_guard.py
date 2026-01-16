"""
Economic Guard - News Filter & Risk Modulator
=============================================
Detects High Impact News events to protect capital.
Uses MT5 Calendar.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional

from app.core.mt5_conn import mt5_conn, mt5

logger = logging.getLogger("feat.calendar_guard")

class CalendarGuard:
    def __init__(self):
        self.last_check = 0
        self.cached_events = []
        self.cache_ttl = 60 # Check every minute
    
    async def check_news_impact(self, symbol: str = "USD", lookahead_minutes: int = 60) -> Dict[str, Any]:
        """
        Checks for High Impact news in the upcoming window.
        Returns:
            {
                "is_news_time": bool, 
                "max_lot_cap": float (if news),
                "events": List[str]
            }
        """
        if not mt5_conn.connected:
             return {"is_news_time": False, "events": []}
             
        try:
            # Simple heuristic mapping
            currency = symbol[:3] # e.g. "EUR" from "EURUSD"
            base = symbol[3:]     # e.g. "USD"
            
            # For simplicity, we check USD and the pair's currencies
            currencies = ["USD", currency, base]
            
            now = datetime.now()
            future = now + timedelta(minutes=lookahead_minutes)
            
            # MT5 Calendar Get
            # Note: mt5.calendar_get takes datetime objects
            events = mt5.calendar_get(now, future)
            
            if events is None:
                return {"is_news_time": False, "events": []}
                
            high_impact_events = []
            
            for e in events:
                # 0=Low, 1=Medium, 2=High. We care about 2 (High)
                if e.importance >= 2:
                     # Check currency relevance
                     if e.currency in currencies:
                         high_impact_events.append(f"{e.currency}: {e.event_name}")
            
            if high_impact_events:
                return {
                    "is_news_time": True,
                    "max_lot_cap": 0.02, # DEFENSIVE MODE
                    "events": high_impact_events,
                    "risk_mode": "SMART_DEFENSE"
                }
            
            return {"is_news_time": False, "events": []}

        except Exception as e:
            logger.error(f"Calendar Check Failure: {e}")
            return {"is_news_time": False, "error": str(e)}

calendar_guard = CalendarGuard()
