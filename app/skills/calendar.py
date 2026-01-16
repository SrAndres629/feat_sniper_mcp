import pytz
import logging
from datetime import datetime, timezone, time as dt_time
from typing import Dict, Any, Optional

logger = logging.getLogger("feat.chronos")

class ChronosEngine:
    """
    [T] COMPONENT - TIME (The Chronos Engine)
    Handles strict institutional Killzones with 'America/New_York' alignment.
    """
    NY_TZ = pytz.timezone("America/New_York")
    
    # FEAT Killzones (NY Time)
    KILLZONES = {
        "LONDON_OPEN": (dt_time(2, 0), dt_time(5, 0)),    # 02:00-05:00 AM NY
        "TRANSFER_ZONE": (dt_time(4, 30), dt_time(5, 0)), # 04:30-05:00 AM NY (Overlap)
        "NY_AM": (dt_time(8, 0), dt_time(11, 0)),        # 08:00-11:00 AM NY
        "SILVER_BULLET_NY": (dt_time(10, 0), dt_time(11, 0)), # 10:00-11:00 AM NY
        "NY_PM": (dt_time(13, 0), dt_time(16, 0))         # 13:00-16:00 PM NY
    }

    def __init__(self):
        logger.info("[Chronos] Engine Active (Strict NY-Alignment)")

    def get_ny_now(self) -> datetime:
        """Returns current time in America/New_York."""
        return datetime.now(timezone.utc).astimezone(self.NY_TZ)

    def is_ny_killzone(self) -> bool:
        """Logic: NY Open + AM session + Silver Bullet."""
        now = self.get_ny_now().time()
        start, end = self.KILLZONES["NY_AM"]
        return start <= now <= end

    def is_london_killzone(self) -> bool:
        """Logic: London Open session."""
        now = self.get_ny_now().time()
        start, end = self.KILLZONES["LONDON_OPEN"]
        return start <= now <= end

    def is_transfer_zone(self) -> bool:
        """Logic: Transfer Zone (Institutional Rebalancing)."""
        now = self.get_ny_now().time()
        start, end = self.KILLZONES["TRANSFER_ZONE"]
        return start <= now <= end

    def is_silver_bullet(self) -> bool:
        """Logic: ICT Silver Bullet Hour."""
        now = self.get_ny_now().time()
        start, end = self.KILLZONES["SILVER_BULLET_NY"]
        return start <= now <= end

    def validate_window(self) -> Dict[str, Any]:
        """
        Master validation gate for the [T] vector.
        """
        ny_now = self.get_ny_now()
        t_now = ny_now.time()
        
        active_session = "OFF_HOURS"
        is_killzone = False
        
        # Check all defined zones
        checks = {
            "is_ny": self.is_ny_killzone(),
            "is_london": self.is_london_killzone(),
            "is_transfer": self.is_transfer_zone(),
            "is_silver_bullet": self.is_silver_bullet()
        }
        
        if checks["is_ny"]: active_session = "NY_AM"; is_killzone = True
        elif checks["is_london"]: active_session = "LONDON"; is_killzone = True
        
        # Priority for specialized sub-windows
        if checks["is_silver_bullet"]: active_session = "SILVER_BULLET"
        if checks["is_transfer"]: active_session = "TRANSFER_ZONE"

        return {
            "is_valid": is_killzone or checks["is_transfer"] or checks["is_silver_bullet"],
            "session": active_session,
            "ny_time": t_now.strftime("%H:%M:%S"),
            "checks": checks
        }

    def get_sentiment_impact(self) -> Dict[str, Any]:
        """
        [NLP] Reads latest sentiment from RAG/News feed.
        Returns: { 'sentiment': 'BULLISH'|'BEARISH'|'UNCERTAIN', 'impact': 'HIGH'|'LOW' }
        """
        try:
             import os
             import json
             # RAG Memory writes to this file
             path = "data/sentiment_analysis.json"
             if os.path.exists(path):
                 with open(path, "r") as f:
                     data = json.load(f)
                     # In a real system, check timestamp freshness here
                     return data
        except Exception:
             pass
        return {"sentiment": "NEUTRAL", "impact": "LOW"}

# Global singleton
chronos_engine = ChronosEngine()
