import aiosqlite
import json
import logging
import dataclasses
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger("feat.rag_memory")

class TradeMemory:
    """
    Módulo 7: The Black Box.
    Memoria persistente asíncrona para auditoría y aprendizaje RAG.
    """
    def __init__(self, db_path: str = "trade_history.db"):
        self.db_path = db_path

    async def initialize(self):
        """Crea el esquema de base de datos si no existe."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticket_id INTEGER,
                        symbol TEXT,
                        action TEXT,
                        timestamp DATETIME,
                        price FLOAT,
                        feat_analysis JSON,
                        brain_prediction JSON,
                        physics_snapshot JSON,
                        outcome TEXT
                    )
                """)
                await db.commit()
            logger.info("[OK] Black Box Memory Initialized")
        except Exception as e:
            logger.warning(f"Memory Init Failed (Sqlite?): {e}")

    async def log_trade_context(self, trade_data: Dict, feat_result: Any, brain_result: Dict, physics: Any):
        """
        Guarda el snapshot completo de la decisión.
        Accepts complex objects and converts to JSON.
        """
        try:
            # Serializadores Seguros
            def safe_dict(obj):
                if dataclasses.is_dataclass(obj):
                    return dataclasses.asdict(obj)
                if hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                return str(obj)

            feat_json = json.dumps(safe_dict(feat_result), default=str)
            brain_json = json.dumps(brain_result, default=str)
            physics_json = json.dumps(safe_dict(physics), default=str)
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO trades 
                    (ticket_id, symbol, action, timestamp, price, feat_analysis, brain_prediction, physics_snapshot, outcome)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_data.get('ticket', 0),
                    trade_data.get('symbol', 'UNKNOWN'),
                    trade_data.get('action', 'SIGNAL'),
                    datetime.now().isoformat(),
                    float(trade_data.get('price', 0.0) or 0.0),
                    feat_json,
                    brain_json,
                    physics_json,
                    "PENDING"
                ))
                await db.commit()
                # logger.debug(f"Trade {trade_data.get('ticket')} Logged to Black Box")
        except Exception as e:
            logger.error(f"Memory Write Error: {e}")

    async def get_hourly_performance(self, hour: int) -> Dict[str, float]:
        """
        Consulta el rendimiento histórico para una hora específica.
        Analiza trades pasados en el mismo bloque horario.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Query simple para promediar outcome o profi real si existiera
                # Por ahora usamos 'outcome' (WIN/LOSS) o calculamos de price si outcome es PENDING
                query = """
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins
                    FROM trades 
                    WHERE strftime('%H', timestamp) = ?
                """
                async with db.execute(query, (f"{hour:02d}",)) as cursor:
                    row = await cursor.fetchone()
                    if row and row[0] > 0:
                        win_rate = row[1] / row[0]
                        return {"win_rate": win_rate, "sample_size": row[0]}
            return {"win_rate": 0.5, "sample_size": 0} # Default neutral
        except Exception as e:
            logger.error(f"Memory Read Error (Hourly): {e}")
            return {"win_rate": 0.5, "sample_size": 0}

    async def get_last_n_trades(self, n: int = 10) -> list:
        """
        Recupera los N últimos trades para análisis de racha.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                query = """
                    SELECT outcome, physics_snapshot
                    FROM trades
                    ORDER BY timestamp DESC
                    LIMIT ?
                """
                trades = []
                async with db.execute(query, (n,)) as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        outcome, physics_json = row
                        try:
                            physics = json.loads(physics_json) if physics_json else {}
                            trades.append({"outcome": outcome, "regime": physics.get("regime", "UNKNOWN")})
                        except json.JSONDecodeError:
                            trades.append({"outcome": outcome, "regime": "UNKNOWN"})
                return trades
        except Exception as e:
            logger.error(f"Memory Read Error (Last N): {e}")
            return []

# Instancia Global
rag_memory = TradeMemory()
