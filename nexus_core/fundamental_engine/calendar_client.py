"""
[MACRO SENTINEL - Calendar Client]
===================================
Interface for Economic Calendar Data.
Production: Connect to ForexFactory API or Investing.com scraper.
Development: Uses mock data for testing.
"""

import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class EventImpact(Enum):
    """Economic Event Impact Classification."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3

@dataclass
class EconomicEvent:
    """Representation of a single economic event."""
    timestamp: datetime.datetime
    currency: str
    event_name: str
    impact: EventImpact
    forecast: Optional[float] = None
    previous: Optional[float] = None
    actual: Optional[float] = None

class CalendarClient:
    """
    [MACRO SENTINEL - Data Layer]
    Fetches economic events from calendar providers.
    """
    
    def __init__(self, provider: str = "forexfactory"):
        """
        Args:
            provider: "mock", "forexfactory"
        """
        self.provider = provider
        if provider == "forexfactory":
            from .forexfactory_provider import ForexFactoryProvider
            self.ff_provider = ForexFactoryProvider()
        
    def get_upcoming_events(self, hours_ahead: int = 24, currencies: List[str] = None) -> List[EconomicEvent]:
        """
        Fetches events within the specified time window.
        """
        if self.provider == "mock":
            return self._get_mock_events(hours_ahead, currencies)
        elif self.provider == "forexfactory":
            events = self.ff_provider.fetch_events()
            
            # Apply filters
            now = datetime.datetime.now()
            cutoff = now + datetime.timedelta(hours=hours_ahead)
            
            filtered = [e for e in events if e.timestamp <= cutoff]
            if currencies:
                filtered = [e for e in filtered if e.currency in currencies]
            
            return filtered
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _get_mock_events(self, hours_ahead: int, currencies: List[str] = None) -> List[EconomicEvent]:
        """
        Returns realistic mock events for development/testing.
        These simulate a typical trading week.
        """
        now = datetime.datetime.now()
        
        mock_events = [
            EconomicEvent(
                timestamp=now + datetime.timedelta(minutes=15),
                currency="USD",
                event_name="FOMC Member Speech",
                impact=EventImpact.MEDIUM,
            ),
            EconomicEvent(
                timestamp=now + datetime.timedelta(hours=2),
                currency="USD",
                event_name="Non-Farm Payrolls",
                impact=EventImpact.HIGH,
                forecast=200000,
                previous=175000
            ),
            EconomicEvent(
                timestamp=now + datetime.timedelta(hours=8),
                currency="EUR",
                event_name="ECB Interest Rate Decision",
                impact=EventImpact.HIGH,
            ),
            EconomicEvent(
                timestamp=now + datetime.timedelta(days=1),
                currency="GBP",
                event_name="UK GDP",
                impact=EventImpact.HIGH,
            ),
        ]
        
        if currencies:
            mock_events = [e for e in mock_events if e.currency in currencies]
            
        return sorted(mock_events, key=lambda x: x.timestamp)
    
    def get_next_high_impact(self, currencies: List[str] = None) -> Optional[EconomicEvent]:
        """
        Returns the next HIGH impact event.
        """
        events = self.get_upcoming_events(hours_ahead=48, currencies=currencies)
        for event in events:
            if event.impact == EventImpact.HIGH:
                return event
        return None
