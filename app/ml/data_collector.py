"""
Data Collector - Quantum Leap Phase 1 (Optimized)
===================================================
Captura de ticks/velas con persistencia en SQLite WAL.

Optimizaciones:
- SQLite WAL mode para concurrencia
- Batch inserts cada N ticks
- ndices para queries rpidas del Oracle
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
    "feat_score", "fsm_state", "liquidity_ratio", "volatility_zscore",
    "momentum_kinetic_micro", "entropy_coefficient", "cycle_harmonic_phase", 
    "institutional_mass_flow", "volatility_regime_norm", "acceptance_ratio", 
    "wick_stress", "poc_z_score", "cvd_acceleration",
    "micro_comp", "micro_slope", "oper_slope", "macro_slope", "bias_slope", "fan_bullish"
]

TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]
TIMEFRAME_MAP = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440, "W1": 10080
}


class SQLiteWALConnection:
    """
    Thread-safe SQLite connection with WAL mode.
    Permite lecturas y escrituras simultneas.
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
                    logger.info(" Institutional schema applied.")
                
                # Dynamic Migration (Add columns if missing)
                cursor = conn.execute("PRAGMA table_info(market_data)")
                existing_cols = [row[1] for row in cursor.fetchall()]
                new_cols = [
                    "momentum_kinetic_micro", "entropy_coefficient", "cycle_harmonic_phase",
                    "institutional_mass_flow", "volatility_regime_norm", "acceptance_ratio",
                    "wick_stress", "poc_z_score", "cvd_acceleration",
                    "micro_comp", "micro_slope", "oper_slope", "macro_slope", "bias_slope", "fan_bullish"
                ]
                for col in new_cols:
                    if col not in existing_cols:
                        try:
                            conn.execute(f"ALTER TABLE market_data ADD COLUMN {col} REAL")
                            logger.info(f" Migrated: Added column {col} to market_data.")
                        except sqlite3.OperationalError as e:
                            logger.debug(f"Column {col} migration skipped (likely exists): {e}")
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
    """
    Engine for Multi-Timeframe (MTF) Candle Aggregation.
    
    [HOT PATH OPTIMIZED] Uses pure Python datetime instead of pandas.to_datetime
    to achieve sub-1ms tick processing.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.buffers: Dict[str, List[Dict]] = {tf: [] for tf in TIMEFRAMES if tf != "M1"}
        self.current_candles: Dict[str, Dict] = {}

    def _parse_time(self, tick_time) -> datetime:
        """Fast time parsing without Pandas overhead."""
        if isinstance(tick_time, datetime):
            return tick_time
        if isinstance(tick_time, str):
            # ISO format: 2024-01-14T12:30:00+00:00
            try:
                return datetime.fromisoformat(tick_time.replace('Z', '+00:00'))
            except ValueError:
                return datetime.now(timezone.utc)
        return datetime.now(timezone.utc)
    
    def _floor_to_timeframe(self, dt: datetime, minutes: int) -> datetime:
        """Floor datetime to timeframe boundary without Pandas."""
        # Calculate minutes since midnight
        total_minutes = dt.hour * 60 + dt.minute
        floored_minutes = (total_minutes // minutes) * minutes
        new_hour = floored_minutes // 60
        new_minute = floored_minutes % 60
        return dt.replace(hour=new_hour, minute=new_minute, second=0, microsecond=0)

    def push_tick(self, m1_candle: Dict) -> List[Tuple[str, Dict]]:
        """Aggregates M1 candles into larger timeframes.
        
        [HOT PATH OPTIMIZED] No Pandas calls - pure Python datetime operations.
        
        Returns:
            List[Tuple[str, Dict]]: List of completed candles (timeframe, data).
        """
        completed = []
        m1_time = self._parse_time(m1_candle.get("tick_time"))
        
        for tf in self.buffers.keys():
            minutes = TIMEFRAME_MAP[tf]
            # Calculate the start of the timeframe window
            window_start = self._floor_to_timeframe(m1_time, minutes)
            
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
        # Extract Physics from 'physics' sub-object if present
        physics = indicators.get("physics", {})
        
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
            "volatility_zscore": float(indicators.get("volatility_zscore", 0.0)),
            # New Neural Tensors
            "momentum_kinetic_micro": float(indicators.get("momentum_kinetic_micro", 0.0)),
            "entropy_coefficient": float(indicators.get("entropy_coefficient", 0.0)),
            "cycle_harmonic_phase": float(indicators.get("cycle_harmonic_phase", 0.0)),
            "institutional_mass_flow": float(indicators.get("institutional_mass_flow", 0.0)),
            "volatility_regime_norm": float(indicators.get("volatility_regime_norm", 0.0)),
            "acceptance_ratio": float(indicators.get("acceptance_ratio", 0.0)),
            "wick_stress": float(indicators.get("wick_stress", 0.0)),
            "poc_z_score": float(indicators.get("poc_z_score", 0.0)),
            "cvd_acceleration": float(indicators.get("cvd_acceleration", 0.0)),
            
            # Ribbon Physics Mapping (Multifractal Layers)
            "micro_comp": float(physics.get("l1_c", indicators.get("micro_comp", 0.5))),
            "micro_slope": float(physics.get("l1_s", indicators.get("micro_slope", 0.0))),
            "oper_slope": float(physics.get("l2_s", indicators.get("oper_slope", 0.0))),
            "macro_slope": float(physics.get("l3_s", indicators.get("macro_slope", 0.0))),
            "bias_slope": float(physics.get("l4_s", indicators.get("bias_slope", 0.0))),
            "fan_bullish": float(1.0 if (physics.get("l1_m", 0) > physics.get("l2_m", 0) > physics.get("l3_m", 0)) else 0.0)
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
                    liquidity_ratio, volatility_zscore,
                    momentum_kinetic_micro, entropy_coefficient, cycle_harmonic_phase,
                    institutional_mass_flow, volatility_regime_norm, acceptance_ratio,
                    wick_stress, poc_z_score, cvd_acceleration,
                    micro_comp, micro_slope, oper_slope, macro_slope, bias_slope, fan_bullish
                ) VALUES (
                    :tick_time, :symbol, :timeframe, :close, :open, :high, :low, :volume,
                    :rsi, :atr, :ema_fast, :ema_slow, :fsm_state, :feat_score,
                    :liquidity_ratio, :volatility_zscore,
                    :momentum_kinetic_micro, :entropy_coefficient, :cycle_harmonic_phase,
                    :institutional_mass_flow, :volatility_regime_norm, :acceptance_ratio,
                    :wick_stress, :poc_z_score, :cvd_acceleration,
                    :micro_comp, :micro_slope, :oper_slope, :macro_slope, :bias_slope, :fan_bullish
                )
            """, self.batch)
            conn.commit()
            
        self.samples_collected += len(self.batch)
        self.batch = []
        
        # Procesar etiquetas pendientes
        if self.samples_collected % 500 == 0:
            # Asumir smbolo del ltimo tick
            if self.batch:
                self.oracle.process_pending_labels(self.batch[-1]["symbol"])
                
    def force_flush(self):
        """Fuerza flush del batch pendiente."""
        with self._lock:
            self._flush_batch()
            
    def get_stats(self) -> Dict:
        """Estadsticas de recoleccin multitemporal."""
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
            # We select * because the View now defines the exact Training Schema
            rows = conn.execute("SELECT * FROM training_samples ORDER BY timestamp").fetchall()
            
            if not rows:
                logger.warning("No training samples found (maybe no labels yet?)")
                return path

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            # Dynamic header based on View columns
            writer.writerow(rows[0].keys())
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
    """MCP Tool: Obtiene estadsticas."""
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
# MT5 TICK-LEVEL DATA EXTRACTION (Real CVD)
# =============================================================================

async def fetch_historical_ticks(
    symbol: str, 
    date_from: datetime, 
    date_to: datetime,
    mt5_conn=None
) -> pd.DataFrame:
    """
    Extrae ticks históricos con Bid/Ask/Flags desde MT5.
    Los flags permiten identificar Buy/Sell agresivo para CVD real.
    
    Args:
        symbol: Símbolo del instrumento (ej: "XAUUSD")
        date_from: Fecha inicial (datetime con timezone)
        date_to: Fecha final (datetime con timezone)
        mt5_conn: Opcional, conexión MT5 existente
        
    Returns:
        DataFrame con columnas: time, bid, ask, last, volume, flags, is_buy, is_sell
        
    MT5 Tick Flags Reference:
        - TICK_FLAG_BID = 2 (bid price changed)
        - TICK_FLAG_ASK = 4 (ask price changed)  
        - TICK_FLAG_LAST = 8 (last deal price changed)
        - TICK_FLAG_VOLUME = 16 (volume changed)
        - TICK_FLAG_BUY = 32 (last deal was buy)
        - TICK_FLAG_SELL = 64 (last deal was sell)
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        logger.warning("[Docker Mode] MT5 not available for tick extraction")
        return pd.DataFrame()
    
    # Ensure MT5 is initialized
    if not mt5.terminal_info():
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return pd.DataFrame()
    
    # Fetch ticks with ALL data (bid, ask, last, volume, flags)
    ticks = mt5.copy_ticks_range(symbol, date_from, date_to, mt5.COPY_TICKS_ALL)
    
    if ticks is None or len(ticks) == 0:
        logger.warning(f"No ticks received for {symbol} in range {date_from} -> {date_to}")
        return pd.DataFrame()
    
    df = pd.DataFrame(ticks)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Parse flags to identify aggressive buy/sell
    # REAL MT5 FLAGS:
    # TICK_FLAG_BID (2) - Price changed
    # TICK_FLAG_ASK (4) - Price changed
    # TICK_FLAG_LAST (8) - Price changed
    # TICK_FLAG_VOLUME (16) - Volume changed
    # TICK_FLAG_BUY (32) - Buy trade
    # TICK_FLAG_SELL (64) - Sell trade
    # Note: User request mentioned 8/16, but those are LAST/VOLUME. We use 32/64 to ensure trade direction accuracy.
    
    df['is_buy'] = (df['flags'] & 32) > 0
    df['is_sell'] = (df['flags'] & 64) > 0
    
    logger.info(f"Fetched {len(df)} ticks for {symbol}: {df['is_buy'].sum()} buys, {df['is_sell'].sum()} sells")
    
    return df


def compute_real_cvd(tick_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula CVD (Cumulative Volume Delta) real usando flags de MT5.
    
    A diferencia de la aproximación tick-rule, esto usa los flags reales
    de MT5 que indican si cada tick fue una compra o venta agresiva.
    
    Args:
        tick_df: DataFrame de ticks con columnas 'volume', 'is_buy', 'is_sell'
        
    Returns:
        Dict con métricas CVD:
        - cvd: Valor acumulado final
        - cvd_series: Serie temporal del CVD
        - buy_volume: Volumen total de compras
        - sell_volume: Volumen total de ventas
        - imbalance_ratio: Ratio de desbalance (-1 a 1)
        - acceleration: Tasa de cambio del CVD
    """
    if tick_df.empty or 'is_buy' not in tick_df.columns:
        return {
            "cvd": 0.0,
            "cvd_series": [],
            "buy_volume": 0.0,
            "sell_volume": 0.0,
            "imbalance_ratio": 0.0,
            "acceleration": 0.0
        }
    
    # Calculate signed volume: + for buys, - for sells
    tick_df = tick_df.copy()
    tick_df['signed_volume'] = np.where(
        tick_df['is_buy'], 
        tick_df['volume'],
        np.where(tick_df['is_sell'], -tick_df['volume'], 0)
    )
    
    # Cumulative sum = CVD
    tick_df['cvd'] = tick_df['signed_volume'].cumsum()
    
    buy_vol = tick_df.loc[tick_df['is_buy'], 'volume'].sum()
    sell_vol = tick_df.loc[tick_df['is_sell'], 'volume'].sum()
    total_vol = buy_vol + sell_vol + 1e-9
    
    # Imbalance ratio: -1 (all sells) to +1 (all buys)
    imbalance = (buy_vol - sell_vol) / total_vol
    
    # Acceleration: rate of change of CVD (last 10 vs previous 10)
    cvd_series = tick_df['cvd'].values
    if len(cvd_series) >= 20:
        recent = cvd_series[-10:].mean()
        previous = cvd_series[-20:-10].mean()
        acceleration = (recent - previous) / (abs(previous) + 1e-9)
    else:
        acceleration = 0.0
    
    return {
        "cvd": float(cvd_series[-1]) if len(cvd_series) > 0 else 0.0,
        "cvd_series": cvd_series.tolist(),
        "buy_volume": float(buy_vol),
        "sell_volume": float(sell_vol),
        "imbalance_ratio": float(imbalance),
        "acceleration": float(acceleration)
    }


async def fetch_tick_data(symbol: str, minutes_back: int = 5) -> Dict[str, Any]:
    """
    MCP Tool: Extrae ticks y calcula CVD real.
    
    Args:
        symbol: Símbolo del instrumento
        minutes_back: Minutos hacia atrás para extraer
        
    Returns:
        Dict con ticks y métricas CVD
    """
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(minutes=minutes_back)
    
    tick_df = await fetch_historical_ticks(symbol, date_from, date_to)
    
    if tick_df.empty:
        return {
            "symbol": symbol,
            "status": "no_data",
            "ticks_count": 0,
            "cvd_metrics": None
        }
    
    cvd_metrics = compute_real_cvd(tick_df)
    
    return {
        "symbol": symbol,
        "status": "success",
        "ticks_count": len(tick_df),
        "time_range": {
            "from": date_from.isoformat(),
            "to": date_to.isoformat()
        },
        "cvd_metrics": cvd_metrics
    }


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
