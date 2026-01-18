import os
import sqlite3
import logging
import threading
from contextlib import contextmanager

logger = logging.getLogger("DataCollector.DB")

class SQLiteWALConnection:
    """Thread-safe SQLite connection with WAL mode for High-Frequency concurrency."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_db()
        
    @contextmanager
    def get_connection(self):
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA cache_size=20000")
            self._local.conn.execute("PRAGMA temp_store=MEMORY")
            self._local.conn.row_factory = sqlite3.Row
        try:
            yield self._local.conn
        except Exception as e:
            if hasattr(self._local, 'conn') and self._local.conn: self._local.conn.rollback()
            raise e
            
    def _init_db(self):
        schema_path = os.path.join(os.getcwd(), "app", "db", "institutional_schema.sql")
        with self.get_connection() as conn:
            if os.path.exists(schema_path):
                with open(schema_path, "r") as f: conn.executescript(f.read())
                logger.info("Institutional schema applied.")
            
            cursor = conn.execute("PRAGMA table_info(market_data)")
            existing = [row[1] for row in cursor.fetchall()]
            new_cols = [
                "momentum_kinetic_micro", "entropy_coefficient", "cycle_harmonic_phase",
                "institutional_mass_flow", "volatility_regime_norm", "acceptance_ratio",
                "wick_stress", "poc_z_score", "cvd_acceleration",
                "micro_comp", "micro_slope", "oper_slope", "macro_slope", "bias_slope", "fan_bullish"
            ]
            for col in new_cols:
                if col not in existing:
                    try: conn.execute(f"ALTER TABLE market_data ADD COLUMN {col} REAL")
                    except: pass
            conn.commit()
