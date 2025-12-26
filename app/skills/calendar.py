import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import MetaTrader5 as mt5
import pandas as pd
from app.core.mt5_conn import mt5_conn
from app.models.schemas import CalendarRequest

logger = logging.getLogger("MT5_Bridge.Skills.Calendar")

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
    Obtiene eventos del calendario económico de MT5.
    """
    from_date = datetime.now()
    to_date = from_date + timedelta(days=req.days_forward)
    
    # Obtener eventos
    events = await mt5_conn.execute(mt5.calendar_events_get, req.currency, from_date, to_date)
    
    if events is None or len(events) == 0:
        return {
            "status": "success",
            "message": "No se encontraron eventos económicos en el periodo.",
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
    for _, row in df.iterrows():
        # Obtener nombres descriptivos (opcionalmente podríamos llamar a calendar_countries_get)
        results.append({
            "id": int(row['id']),
            "time": datetime.fromtimestamp(row['time']).strftime('%Y-%m-%d %H:%M:%S'),
            "currency": row['currency'],
            "event_name": row['event_name'] if 'event_name' in row else f"Event ID {row['event_id']}",
            "importance": row['importance'],
            "actual": row['actual'] if 'actual' in row else None,
            "forecast": row['forecast'] if 'forecast' in row else None,
            "prev": row['prev'] if 'prev' in row else None
        })
        
    return {
        "status": "success",
        "currency_filter": req.currency,
        "importance_filter": req.importance,
        "total_events": len(results),
        "events": results[:30] # Limitamos a los 30 más relevantes
    }
