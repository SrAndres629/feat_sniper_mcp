"""
Data Collector - Quantum Leap Phase 1 (Optimized)
===================================================
Captura de ticks/velas con persistencia en SQLite WAL.

Optimizaciones:
- SQLite WAL mode para concurrencia
- Batch inserts cada N ticks
- Índices para queries rápidas del Oracle
"""

import os
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional
from contextlib import contextmanager
import threading

# Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DATA_DIR = os.getenv("DATA_DIR", "data")
DB_PATH = os.path.join(DATA_DIR, "market_data.db")
N_LOOKAHEAD = int(os.getenv("N_LOOKAHEAD", "10"))
PROFIT_THRESHOLD = float(os.getenv("PROFIT_THRESHOLD", "0.002"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
FLUSH_INTERVAL = float(os.getenv("FLUSH_INTERVAL", "1.0"))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("QuantumLeap.DataCollector")

# Feature names for consistency
FEATURE_NAMES = [
    "close", "open", "high", "low", "volume",
    "rsi", "ema_fast", "ema_slow", "ema_spread",
    "feat_score", "fsm_state", "atr", "compression",
    "liquidity_above", "liquidity_below"
]


class SQLiteWALConnection:
    """
    Thread-safe SQLite connection with WAL mode.
    Permite lecturas y escrituras simultáneas.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._ensure_directory()
        self._init_db()
        
    def _ensure_directory(self):
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        
    @contextmanager
    def get_connection(self):
        """Thread-local connection context manager."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            # WAL mode for concurrent access
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA cache_size=10000")
            self._local.conn.execute("PRAGMA temp_store=MEMORY")
            self._local.conn.row_factory = sqlite3.Row
            
        try:
            yield self._local.conn
        except Exception as e:
            self._local.conn.rollback()
            raise e
            
    def _init_db(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            # Tabla de ticks raw (sin etiquetar)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    close REAL, open REAL, high REAL, low REAL, volume REAL,
                    rsi REAL, ema_fast REAL, ema_slow REAL, ema_spread REAL,
                    feat_score REAL, fsm_state REAL, atr REAL, compression REAL,
                    liquidity_above REAL, liquidity_below REAL,
                    label INTEGER DEFAULT NULL,
                    labeled_at TEXT DEFAULT NULL
                )
            """)
            
            # Índices para queries rápidas
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticks_symbol_ts 
                ON ticks(symbol, timestamp DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticks_unlabeled 
                ON ticks(label) WHERE label IS NULL
            """)
            
            # Tabla de training samples (etiquetados)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tick_id INTEGER REFERENCES ticks(id),
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    close REAL, open REAL, high REAL, low REAL, volume REAL,
                    rsi REAL, ema_fast REAL, ema_slow REAL, ema_spread REAL,
                    feat_score REAL, fsm_state REAL, atr REAL, compression REAL,
                    liquidity_above REAL, liquidity_below REAL,
                    label INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Índice para training
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_ts 
                ON training_samples(timestamp)
            """)
            
            conn.commit()
            logger.info(f"✅ Database initialized: {self.db_path}")


class OracleLabeler:
    """
    Oracle que etiqueta datos mirando N velas hacia el futuro.
    Usa queries SQL eficientes en lugar de cargar todo en memoria.
    """
    
    def __init__(self, db: SQLiteWALConnection, lookahead: int = N_LOOKAHEAD, 
                 threshold: float = PROFIT_THRESHOLD):
        self.db = db
        self.lookahead = lookahead
        self.threshold = threshold
        
    def process_pending_labels(self, symbol: str) -> int:
        """
        Etiqueta ticks pendientes que ya tienen suficiente historia futura.
        Usa SQL para eficiencia - no carga todo en RAM.
        
        Returns:
            Número de muestras etiquetadas
        """
        with self.db.get_connection() as conn:
            # Encontrar ticks sin etiquetar que ya tienen N velas futuras
            query = """
                SELECT t1.id, t1.timestamp, t1.close as entry_price,
                       (SELECT close FROM ticks t2 
                        WHERE t2.symbol = t1.symbol 
                        AND t2.id = t1.id + :lookahead) as future_price
                FROM ticks t1
                WHERE t1.symbol = :symbol
                AND t1.label IS NULL
                AND EXISTS (
                    SELECT 1 FROM ticks t2 
                    WHERE t2.symbol = t1.symbol 
                    AND t2.id = t1.id + :lookahead
                )
                LIMIT 1000
            """
            
            pending = conn.execute(query, {
                "symbol": symbol,
                "lookahead": self.lookahead
            }).fetchall()
            
            labeled_count = 0
            now = datetime.utcnow().isoformat()
            
            for row in pending:
                if row["future_price"] is None:
                    continue
                    
                # Calcular PnL
                pnl = (row["future_price"] - row["entry_price"]) / row["entry_price"]
                label = 1 if pnl > self.threshold else 0
                
                # Actualizar tick con label
                conn.execute("""
                    UPDATE ticks 
                    SET label = :label, labeled_at = :now
                    WHERE id = :id
                """, {"label": label, "now": now, "id": row["id"]})
                
                # Copiar a training_samples para acceso rápido
                conn.execute("""
                    INSERT INTO training_samples 
                    (tick_id, timestamp, symbol, close, open, high, low, volume,
                     rsi, ema_fast, ema_slow, ema_spread, feat_score, fsm_state,
                     atr, compression, liquidity_above, liquidity_below, label)
                    SELECT id, timestamp, symbol, close, open, high, low, volume,
                           rsi, ema_fast, ema_slow, ema_spread, feat_score, fsm_state,
                           atr, compression, liquidity_above, liquidity_below, :label
                    FROM ticks WHERE id = :id
                """, {"label": label, "id": row["id"]})
                
                labeled_count += 1
                
            conn.commit()
            
            if labeled_count > 0:
                logger.info(f"Oracle labeled {labeled_count} samples for {symbol}")
                
            return labeled_count


class DataCollector:
    """
    Colector principal con batch inserts para alta frecuencia.
    """
    
    def __init__(self, db_path: str = DB_PATH):
        self.db = SQLiteWALConnection(db_path)
        self.oracle = OracleLabeler(self.db)
        self.batch: List[Dict] = []
        self.samples_collected = 0
        self._lock = threading.Lock()
        
    def compute_features(self, candle: Dict, indicators: Dict) -> Dict[str, float]:
        """Construye vector de features X."""
        return {
            "close": float(candle.get("close", 0)),
            "open": float(candle.get("open", 0)),
            "high": float(candle.get("high", 0)),
            "low": float(candle.get("low", 0)),
            "volume": float(candle.get("volume", 0)),
            "rsi": float(indicators.get("rsi", 50.0)),
            "ema_fast": float(indicators.get("ema_fast", candle.get("close", 0))),
            "ema_slow": float(indicators.get("ema_slow", candle.get("close", 0))),
            "ema_spread": float(
                indicators.get("ema_fast", 0) - indicators.get("ema_slow", 0)
            ),
            "feat_score": float(indicators.get("feat_score", 0.0)),
            "fsm_state": float(indicators.get("fsm_state", 0)),
            "atr": float(indicators.get("atr", 0.001)),
            "compression": float(indicators.get("compression", 0.5)),
            "liquidity_above": float(indicators.get("liquidity_above", 0)),
            "liquidity_below": float(indicators.get("liquidity_below", 0))
        }
        
    def collect(self, symbol: str, candle: Dict, indicators: Dict):
        """
        Añade tick al batch. Flush automático cuando alcanza BATCH_SIZE.
        """
        features = self.compute_features(candle, indicators)
        timestamp = datetime.utcnow().isoformat()
        
        record = {
            "timestamp": timestamp,
            "symbol": symbol,
            **features
        }
        
        with self._lock:
            self.batch.append(record)
            
            if len(self.batch) >= BATCH_SIZE:
                self._flush_batch()
                
    def _flush_batch(self):
        """Inserta batch en SQLite."""
        if not self.batch:
            return
            
        with self.db.get_connection() as conn:
            conn.executemany("""
                INSERT INTO ticks (
                    timestamp, symbol, close, open, high, low, volume,
                    rsi, ema_fast, ema_slow, ema_spread, feat_score, fsm_state,
                    atr, compression, liquidity_above, liquidity_below
                ) VALUES (
                    :timestamp, :symbol, :close, :open, :high, :low, :volume,
                    :rsi, :ema_fast, :ema_slow, :ema_spread, :feat_score, :fsm_state,
                    :atr, :compression, :liquidity_above, :liquidity_below
                )
            """, self.batch)
            conn.commit()
            
        self.samples_collected += len(self.batch)
        logger.debug(f"Flushed {len(self.batch)} ticks (total: {self.samples_collected})")
        self.batch = []
        
        # Procesar etiquetas pendientes
        if self.samples_collected % 500 == 0:
            # Asumir símbolo del último tick
            if self.batch:
                self.oracle.process_pending_labels(self.batch[-1]["symbol"])
                
    def force_flush(self):
        """Fuerza flush del batch pendiente."""
        with self._lock:
            self._flush_batch()
            
    def get_stats(self) -> Dict:
        """Estadísticas de recolección."""
        with self.db.get_connection() as conn:
            total_ticks = conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
            labeled_ticks = conn.execute(
                "SELECT COUNT(*) FROM ticks WHERE label IS NOT NULL"
            ).fetchone()[0]
            training_samples = conn.execute(
                "SELECT COUNT(*) FROM training_samples"
            ).fetchone()[0]
            
        return {
            "total_ticks": total_ticks,
            "labeled_ticks": labeled_ticks,
            "training_samples": training_samples,
            "pending_batch": len(self.batch),
            "db_path": self.db.db_path,
            "lookahead": self.oracle.lookahead,
            "threshold": self.oracle.threshold
        }
        
    def export_training_csv(self, path: str = None) -> str:
        """Exporta training_samples a CSV para compatibilidad."""
        import csv
        
        path = path or os.path.join(DATA_DIR, "training_dataset.csv")
        
        with self.db.get_connection() as conn:
            rows = conn.execute("""
                SELECT timestamp, symbol, close, open, high, low, volume,
                       rsi, ema_fast, ema_slow, ema_spread, feat_score, fsm_state,
                       atr, compression, liquidity_above, liquidity_below, label
                FROM training_samples
                ORDER BY timestamp
            """).fetchall()
            
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "symbol", "close", "open", "high", "low", "volume",
                "rsi", "ema_fast", "ema_slow", "ema_spread", "feat_score", "fsm_state",
                "atr", "compression", "liquidity_above", "liquidity_below", "label"
            ])
            for row in rows:
                writer.writerow(row)
                
        logger.info(f"Exported {len(rows)} samples to {path}")
        return path


# Singleton instance
data_collector = DataCollector()


# =============================================================================
# MCP-COMPATIBLE ASYNC WRAPPERS
# =============================================================================

async def collect_sample(symbol: str, candle: Dict, indicators: Dict) -> Dict:
    """MCP Tool: Recolecta muestra."""
    data_collector.collect(symbol, candle, indicators)
    return data_collector.get_stats()
    
    
async def get_collection_stats() -> Dict:
    """MCP Tool: Obtiene estadísticas."""
    return data_collector.get_stats()
    
    
async def export_for_training(path: str = None) -> Dict:
    """MCP Tool: Exporta datos a CSV para training."""
    csv_path = data_collector.export_training_csv(path)
    return {"csv_path": csv_path, "status": "exported"}
    
    
async def flush_pending() -> Dict:
    """MCP Tool: Fuerza flush del batch pendiente."""
    data_collector.force_flush()
    return {"status": "flushed", **data_collector.get_stats()}
