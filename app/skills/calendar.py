import pytz
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
try:
    import MetaTrader5 as mt5
except ImportError:
    from unittest.mock import MagicMock
    mt5 = MagicMock()
import pandas as pd
from app.core.mt5_conn import mt5_conn
from app.models.schemas import CalendarRequest

logger = logging.getLogger("MT5_Bridge.Skills.Calendar")

class ChronosEngine:
    """
    Gate T: Chronos Engine.
    Handles institutional Killzones and session transitions synchronized with America/New_York.
    """
    def __init__(self):
        print("[Chronos] Engine Active (Session detection)")
        self.status = "INITIALIZED"

    NY_TZ = pytz.timezone("America/New_York")
    
    # Institutional Killzones (EST/NY)
    SESSIONS = {
        "LONDON_OPEN": {"start": "02:00", "end": "05:00"},
        "NY_AM": {"start": "08:00", "end": "11:00"},
        "PRE_NY": {"start": "07:00", "end": "07:30"},
        "NY_OPEN": {"start": "07:30", "end": "08:30"}
    }

    @classmethod
    def get_market_time(cls, utc_dt: Optional[datetime] = None) -> datetime:
        """Converts UTC timestamp (or current) to America/New_York."""
        if utc_dt is None:
            utc_dt = datetime.now(pytz.UTC)
        elif utc_dt.tzinfo is None:
            utc_dt = pytz.UTC.localize(utc_dt)
        return utc_dt.astimezone(cls.NY_TZ)

    @classmethod
    def get_session_status(cls, utc_dt: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Determines current institutional session status.
        Returns deterministic flags for Killzones.
        """
        ny_time = cls.get_market_time(utc_dt)
        current_time_str = ny_time.strftime("%H:%M")
        
        status = {
            "session_name": "OFF_HOURS",
            "is_killzone": False,
            "ny_time": current_time_str,
            "is_pre_ny": "07:00" <= current_time_str <= "07:30",
            "is_ny_open": "07:30" <= current_time_str <= "08:30",
            "is_london_open": "02:00" <= current_time_str <= "05:00",
            "is_ny_am": "08:00" <= current_time_str <= "11:00"
        }
        
        # Determine dominant session name and is_killzone
        if status["is_ny_open"]:
            status["session_name"] = "NY_OPEN"
            status["is_killzone"] = True
        elif status["is_ny_am"]:
            status["session_name"] = "NY_AM"
            status["is_killzone"] = True
        elif status["is_london_open"]:
            status["session_name"] = "LONDON_OPEN"
            status["is_killzone"] = True
        elif status["is_pre_ny"]:
            status["session_name"] = "PRE_NY"
            status["is_killzone"] = True
            
        return status

chronos_engine = ChronosEngine()

# Mapeo de importancia MT5
# CALENDAR_IMPORTANCE_NONE (0), LOW (1), MEDIUM (2), HIGH (3)
# Nota: Usamos enteros directos porque algunas versiones de la lib no exponen las constantes
IMPORTANCE_MAP = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3
}

async def get_economic_calendar(req: CalendarRequest) -> Dict[str, Any]:
    """
    Obtiene eventos del calendario econmico de MT5.
    """
    from_date = datetime.now()
    to_date = from_date + timedelta(days=req.days_forward)
    
    # Obtener eventos
    events = await mt5_conn.execute(mt5.calendar_events_get, req.currency, from_date, to_date)
    
    if events is None or len(events) == 0:
        return {
            "status": "success",
            "message": "No se encontraron eventos econmicos en el periodo.",
            "events": []
        }
    
    # Procesar eventos
    df = pd.DataFrame(list(events))
    
    # Filtrar por importancia si se solicita
    if req.importance:
        target_importance = IMPORTANCE_MAP.get(req.importance)
        df = df[df['importance'] == target_importance]
    
    # Limpiar y formatear para el LLM
    results = []
    for row in df.itertuples():
        # Obtener nombres descriptivos (opcionalmente podramos llamar a calendar_countries_get)
        results.append({
            "id": int(row.id),
            "time": datetime.fromtimestamp(row.time).strftime('%Y-%m-%d %H:%M:%S'),
            "currency": row.currency,
            "event_name": row.event_name if hasattr(row, 'event_name') else f"Event ID {row.event_id}",
            "importance": row.importance,
            "actual": row.actual if hasattr(row, 'actual') else None,
            "forecast": row.forecast if hasattr(row, 'forecast') else None,
            "prev": row.prev if hasattr(row, 'prev') else None
        })
        
    return {
        "status": "success",
        "currency_filter": req.currency,
        "importance_filter": req.importance,
        "total_events": len(results),
        "events": results[:30] # Limitamos a los 30 ms relevantes
    }
