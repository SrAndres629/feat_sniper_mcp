from datetime import datetime, timezone
from .db import SQLiteWALConnection
from .constants import N_LOOKAHEAD, PROFIT_THRESHOLD

class OracleLabeler:
    """SQL-optimized labeling engine for deep learning datasets."""
    def __init__(self, db: SQLiteWALConnection, lookahead: int = N_LOOKAHEAD, threshold: float = PROFIT_THRESHOLD):
        self.db = db
        self.lookahead = lookahead
        self.threshold = threshold
        
    def process_pending_labels(self, symbol: str, timeframe: str = "M1") -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.db.get_connection() as conn:
            q = """
                UPDATE market_data
                SET label = CASE 
                        WHEN ((SELECT close FROM market_data m2 WHERE m2.symbol = market_data.symbol AND m2.timeframe = market_data.timeframe AND m2.id = market_data.id + :lk) - close) / close > :th 
                        THEN 1 ELSE 0 END,
                    labeled_at = :now
                WHERE symbol = :symbol AND timeframe = :tf AND label IS NULL
                AND EXISTS (SELECT 1 FROM market_data m2 WHERE m2.symbol = market_data.symbol AND m2.timeframe = market_data.timeframe AND m2.id = market_data.id + :lk)
            """
            cursor = conn.execute(q, {"symbol": symbol, "tf": timeframe, "lk": self.lookahead, "th": self.threshold, "now": now})
            conn.commit()
            return cursor.rowcount
