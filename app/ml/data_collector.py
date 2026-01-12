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
from contextlib import contextmanager
import threading
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

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

FEATURE_NAMES: List[str] = [
    "close", "open", "high", "low", "volume",
    "rsi", "atr", "ema_fast", "ema_slow",
    "feat_score", "fsm_state", "liquidity_ratio", "volatility_zscore"
]

TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]
TIMEFRAME_MAP = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440, "W1": 10080
}


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
        
    def _ensure_directory(self) -> None:
        """Creates data directory if non-existent."""
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
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA cache_size=20000") # Higher for MTF
            self._local.conn.execute("PRAGMA temp_store=MEMORY")
            self._local.conn.row_factory = sqlite3.Row
            
        try:
            yield self._local.conn
        except Exception as e:
            if hasattr(self._local, 'conn') and self._local.conn:
                self._local.conn.rollback()
            raise e
            
    def _init_db(self):
        """Initialize database schema from institutional_schema.sql."""
        schema_path = os.path.join(os.getcwd(), "app", "db", "institutional_schema.sql")
        
        with self.get_connection() as conn:
            if os.path.exists(schema_path):
                with open(schema_path, "r") as f:
                    conn.executescript(f.read())
                logger.info("✅ Institutional schema applied.")
            else:
                # Failsafe basic schema
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tick_time TIMESTAMP NOT NULL,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        close REAL, open REAL, high REAL, low REAL, volume REAL,
                        rsi REAL, atr REAL, ema_fast REAL, ema_slow REAL,
                        fsm_state INTEGER, feat_score REAL,
                        label INTEGER DEFAULT NULL,
                        labeled_at TIMESTAMP DEFAULT NULL,
                        UNIQUE(tick_time, symbol, timeframe)
                    )
                """)
            conn.commit()


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
        
    def process_pending_labels(self, symbol: str, timeframe: str = "M1") -> int:
        """Labels pending records using bulk SQL operations."""
        now = datetime.now(timezone.utc).isoformat()
        
        with self.db.get_connection() as conn:
            update_query = """
                UPDATE market_data
                SET 
                    label = CASE 
                        WHEN ((SELECT close FROM market_data m2 
                               WHERE m2.symbol = market_data.symbol 
                               AND m2.timeframe = market_data.timeframe
                               AND m2.id = market_data.id + :lookahead) - close) / close > :threshold 
                        THEN 1 ELSE 0 
                    END,
                    labeled_at = :now
                WHERE symbol = :symbol
                AND timeframe = :tf
                AND label IS NULL
                AND EXISTS (
                    SELECT 1 FROM market_data m2 
                    WHERE m2.symbol = market_data.symbol 
                    AND m2.timeframe = market_data.timeframe
                    AND m2.id = market_data.id + :lookahead
                )
            """
            cursor = conn.execute(update_query, {
                "symbol": symbol,
                "tf": timeframe,
                "lookahead": self.lookahead,
                "threshold": self.threshold,
                "now": now
            })
            conn.commit()
            return cursor.rowcount

class Resampler:
    """Engine for Multi-Timeframe (MTF) Candle Aggregation."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.buffers: Dict[str, List[Dict]] = {tf: [] for tf in TIMEFRAMES if tf != "M1"}
        self.current_candles: Dict[str, Dict] = {}

    def push_tick(self, m1_candle: Dict) -> List[Tuple[str, Dict]]:
        """Aggregates M1 candles into larger timeframes.
        
        Returns:
            List[Tuple[str, Dict]]: List of completed candles (timeframe, data).
        """
        completed = []
        m1_time = pd.to_datetime(m1_candle["tick_time"])
        
        for tf in self.buffers.keys():
            minutes = TIMEFRAME_MAP[tf]
            # Calculate the start of the timeframe window
            window_start = m1_time.floor(f"{minutes}T")
            
            if tf not in self.current_candles or self.current_candles[tf]["time"] != window_start:
                # If window changed, the previous one is completed
                if tf in self.current_candles:
                    completed.append((tf, self.current_candles[tf]))
                
                # Start new candle
                self.current_candles[tf] = {
                    "time": window_start,
                    "tick_time": window_start.isoformat(),
                    "open": m1_candle["open"],
                    "high": m1_candle["high"],
                    "low": m1_candle["low"],
                    "close": m1_candle["close"],
                    "volume": m1_candle["volume"]
                }
            else:
                # Update existing candle
                curr = self.current_candles[tf]
                curr["high"] = max(curr["high"], m1_candle["high"])
                curr["low"] = min(curr["low"], m1_candle["low"])
                curr["close"] = m1_candle["close"]
                curr["volume"] += m1_candle["volume"]
                
        return completed


class DataCollector:
    """
    Colector principal con batch inserts para alta frecuencia.
    """
    
    def __init__(self, db_path: str = DB_PATH):
        self.db = SQLiteWALConnection(db_path)
        self.oracle = OracleLabeler(self.db)
        self.resamplers: Dict[str, Resampler] = {}
        self.batch: List[Dict] = []
        self.samples_collected = 0
        self._lock = threading.Lock()
        
    def compute_features(self, candle: Dict[str, Any], indicators: Dict[str, Any]) -> Dict[str, float]:
        """Core feature engineering vectorizer for MIP."""
        return {
            "close": float(candle.get("close", 0)),
            "open": float(candle.get("open", 0)),
            "high": float(candle.get("high", 0)),
            "low": float(candle.get("low", 0)),
            "volume": float(candle.get("volume", 0)),
            "rsi": float(indicators.get("rsi", 50.0)),
            "atr": float(indicators.get("atr", 0.001)),
            "ema_fast": float(indicators.get("ema_fast", candle.get("close", 0))),
            "ema_slow": float(indicators.get("ema_slow", candle.get("close", 0))),
            "fsm_state": int(indicators.get("fsm_state", 0)),
            "feat_score": float(indicators.get("feat_score", 0.0)),
            "liquidity_ratio": float(indicators.get("liquidity_ratio", 1.0)),
            "volatility_zscore": float(indicators.get("volatility_zscore", 0.0))
        }
        
    def collect(self, symbol: str, candle: Dict[str, Any], indicators: Dict[str, Any], timeframe: str = "M1") -> None:
        """Captures a new tick/candle and triggers resampling if M1."""
        features = self.compute_features(candle, indicators)
        tick_time = candle.get("time") or datetime.now(timezone.utc).isoformat()
        
        record = {
            "tick_time": tick_time,
            "symbol": symbol,
            "timeframe": timeframe,
            **features
        }
        
        with self._lock:
            self.batch.append(record)
            
            # Resampling Logic
            if timeframe == "M1":
                if symbol not in self.resamplers:
                    self.resamplers[symbol] = Resampler(symbol)
                
                completed_tf_candles = self.resamplers[symbol].push_tick(record)
                for tf, tf_candle in completed_tf_candles:
                    # For larger timeframes, indicators would ideally be re-calculated here
                    # or passed from a higher-level logic. For now, we store basic OHLC.
                    # In a full MIP, indicators would be calculated on 'tf_candle'.
                    self.batch.append({
                        "symbol": symbol,
                        "timeframe": tf,
                        **tf_candle,
                        # Placeholders for TF indicators
                        "rsi": 50.0, "atr": 0.001, "ema_fast": tf_candle["close"], 
                        "ema_slow": tf_candle["close"], "fsm_state": 0, "feat_score": 0.0,
                        "liquidity_ratio": 1.0, "volatility_zscore": 0.0
                    })

            if len(self.batch) >= BATCH_SIZE:
                self._flush_batch()
                
    def _flush_batch(self):
        """Inserta batch en la nueva tabla market_data."""
        if not self.batch:
            return
            
        with self.db.get_connection() as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO market_data (
                    tick_time, symbol, timeframe, close, open, high, low, volume,
                    rsi, atr, ema_fast, ema_slow, fsm_state, feat_score,
                    liquidity_ratio, volatility_zscore
                ) VALUES (
                    :tick_time, :symbol, :timeframe, :close, :open, :high, :low, :volume,
                    :rsi, :atr, :ema_fast, :ema_slow, :fsm_state, :feat_score,
                    :liquidity_ratio, :volatility_zscore
                )
            """, self.batch)
            conn.commit()
            
        self.samples_collected += len(self.batch)
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
        """Estadísticas de recolección multitemporal."""
        stats = {"total_samples": self.samples_collected}
        with self.db.get_connection() as conn:
            for tf in TIMEFRAMES:
                count = conn.execute(
                    "SELECT COUNT(*) FROM market_data WHERE timeframe = ?", (tf,)
                ).fetchone()[0]
                stats[f"count_{tf}"] = count
            
            stats["labeled"] = conn.execute(
                "SELECT COUNT(*) FROM market_data WHERE label IS NOT NULL"
            ).fetchone()[0]
            
        return stats
        
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


# =============================================================================
# FEAT-DEEP MULTI-TEMPORAL FUNCTIONS
# =============================================================================

async def fetch_multi_tf_data(symbol: str, mt5_conn=None) -> Dict[str, Any]:
    """
    FEAT-DEEP Protocol: Fetches market data across all timeframes simultaneously.
    
    Returns a dict with candles for: H4, H1, M15, M5, M1
    Used to build the Market State Tensor.
    """
    from app.skills.liquidity_detector import MarketStateTensor, market_tensor
    
    result = {
        "symbol": symbol,
        "H4": pd.DataFrame(),
        "H1": pd.DataFrame(),
        "M15": pd.DataFrame(),
        "M5": pd.DataFrame(),
        "M1": pd.DataFrame(),
        "tensor": None,
        "error": None
    }
    
    # Fetch from SQLite cache (real MT5 fetch would require mt5_conn)
    with data_collector.db.get_connection() as conn:
        for tf in ["H4", "H1", "M15", "M5", "M1"]:
            query = """
                SELECT tick_time, open, high, low, close, volume, rsi, atr
                FROM market_data 
                WHERE symbol = ? AND timeframe = ?
                ORDER BY tick_time DESC
                LIMIT 100
            """
            rows = conn.execute(query, (symbol, tf)).fetchall()
            if rows:
                df = pd.DataFrame([dict(row) for row in rows])
                df = df.sort_values("tick_time", ascending=True).reset_index(drop=True)
                result[tf] = df
    
    # Build Market State Tensor if we have data
    if not result["H4"].empty and not result["M1"].empty:
        try:
            tensor = market_tensor.build_tensor(
                h4_candles=result["H4"],
                h1_candles=result["H1"],
                m15_candles=result["M15"],
                m5_candles=result["M5"],
                m1_candles=result["M1"]
            )
            result["tensor"] = tensor
        except Exception as e:
            result["error"] = str(e)
    
    return result


async def get_h4_bias(symbol: str) -> Dict[str, Any]:
    """
    FEAT-DEEP Protocol: Returns the current H4 trend bias for Veto Rule.
    """
    data = await fetch_multi_tf_data(symbol)
    
    if data.get("tensor"):
        h4_trend = data["tensor"]["macro"].get("H4_Trend", "NEUTRAL")
        return {
            "symbol": symbol,
            "H4_Trend": h4_trend,
            "alignment_score": data["tensor"].get("alignment_score", 0),
            "kill_zone": data["tensor"].get("kill_zone"),
            "in_ny_kz": data["tensor"].get("in_ny_kz", False)
        }
    
    return {"symbol": symbol, "H4_Trend": "NEUTRAL", "error": "No H4 data available"}
