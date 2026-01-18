import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from app.core.config import settings

logger = logging.getLogger("FEAT.Liquidity.Sessions")

# [LEVEL 57] Doctoral KILL_ZONES (Linked to Settings)
KILL_ZONES = {
    "NY": {"start": settings.KILLZONE_NY_START, "end": settings.KILLZONE_NY_END},
    "LONDON": {"start": settings.KILLZONE_LONDON_START, "end": settings.KILLZONE_LONDON_END},
    "ASIA": {"start": settings.KILLZONE_ASIA_START, "end": settings.KILLZONE_ASIA_END},
}

def get_current_kill_zone(utc_offset: int = -4) -> Optional[str]:
    """
    Determina la Kill Zone activa basndose en la hora actual.
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
