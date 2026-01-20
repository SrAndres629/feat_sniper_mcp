import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from .db import SQLiteWALConnection
from .labeler import OracleLabeler
from .resampler import Resampler
from .constants import BATCH_SIZE, DB_PATH, TIMEFRAMES, TIMEFRAME_MAP, SystemState
from app.skills.indicators import calculate_rsi_numpy, calculate_atr_numpy, calculate_feat_layers

logger = logging.getLogger("DataCollector.Engine")

class DataCollector:
    """Orchestrator for multi-timeframe data collection and feature engineering."""
    def __init__(self, db_path: str = DB_PATH):
        self.db = SQLiteWALConnection(db_path)
        self.oracle = OracleLabeler(self.db)
        self.resamplers: Dict[str, Resampler] = {}
        self.batch: List[Dict] = []
        self.tick_batch: List[Dict] = [] # NEW: For raw ticks
        self.samples_collected = 0
        self._lock = threading.Lock()
        self.state = SystemState.BOOTING
        self.hydration_progress = 0.0

    # ... (collect and compute_features stay same)

    def collect_ticks(self, symbol: str, tick_record: Dict):
        """
        Captures high-frequency tick data.
        record format: {tick_time, bid, ask, bid_vol, ask_vol, ofi, entropy, hurst, cvd}
        """
        with self._lock:
            self.tick_batch.append({"symbol": symbol, **tick_record})
            if len(self.tick_batch) >= 100: # Flush ticks more frequently
                self._flush_ticks()

    def _flush_ticks(self):
        if not self.tick_batch: return
        with self.db.get_connection() as conn:
            conn.executemany("""
                INSERT INTO tick_data (
                    tick_time, symbol, bid, ask, bid_vol, ask_vol, ofi, entropy, hurst, cvd
                ) VALUES (
                    :tick_time, :symbol, :bid, :ask, :bid_vol, :ask_vol, :ofi, :entropy, :hurst, :cvd
                )
            """, self.tick_batch)
            conn.commit()
        self.tick_batch = []

    def _flush_batch(self):
        if not self.batch: return
        with self.db.get_connection() as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO market_data (
                    tick_time, symbol, timeframe, close, open, high, low, volume, rsi, atr, ema_fast, ema_slow, fsm_state, feat_score,
                    liquidity_ratio, volatility_zscore, momentum_kinetic_micro, entropy_coefficient, cycle_harmonic_phase, institutional_mass_flow,
                    volatility_regime_norm, acceptance_ratio, wick_stress, poc_z_score, cvd_acceleration, micro_comp, micro_slope, oper_slope, macro_slope, bias_slope, fan_bullish
                ) VALUES (
                    :tick_time, :symbol, :timeframe, :close, :open, :high, :low, :volume, :rsi, :atr, :ema_fast, :ema_slow, :fsm_state, :feat_score,
                    :liquidity_ratio, :volatility_zscore, :momentum_kinetic_micro, :entropy_coefficient, :cycle_harmonic_phase, :institutional_mass_flow,
                    :volatility_regime_norm, :acceptance_ratio, :wick_stress, :poc_z_score, :cvd_acceleration, :micro_comp, :micro_slope, :oper_slope, :macro_slope, :bias_slope, :fan_bullish
                )
            """, self.batch)
            conn.commit()
        self.samples_collected += len(self.batch)
        if self.samples_collected % 500 == 0 and self.batch: self.oracle.process_pending_labels(self.batch[-1]["symbol"])
        self.batch = []

    def force_flush(self):
        with self._lock: 
            self._flush_batch()
            self._flush_ticks()


    def get_stats(self) -> Dict:
        s = {"total_samples": self.samples_collected}
        with self.db.get_connection() as conn:
            for tf in TIMEFRAMES:
                s[f"count_{tf}"] = conn.execute("SELECT COUNT(*) FROM market_data WHERE timeframe = ?", (tf,)).fetchone()[0]
            s["labeled"] = conn.execute("SELECT COUNT(*) FROM market_data WHERE label IS NOT NULL").fetchone()[0]
        return s

    async def hydrate_all_timeframes(self, symbol: str, mt5_fallback=None):
        self.state = SystemState.HYDRATING
        try:
            if symbol not in self.resamplers: self.resamplers[symbol] = Resampler(symbol)
            for i, tf in enumerate(TIMEFRAMES):
                self.hydration_progress = i / len(TIMEFRAMES)
                h = self._get_history_from_db(symbol, tf)
                if len(h) < 200 and mt5_fallback:
                    res = await mt5_fallback(symbol, tf, 500)
                    if res: h = res.get("candles", res) if isinstance(res, dict) else res
                if h:
                    self.resamplers[symbol].history[tf] = h[-200:]
                    from app.ml.ml_engine import ml_engine
                    if tf == "M1":
                        ml_engine.hydrate_hurst(symbol, [float(c['close']) for c in h])
                        ml_engine.hydrate_sequences(symbol, [self.compute_features(c, {"rsi": c.get("rsi", 50)}) for c in h])
            self.state = SystemState.READY
        except Exception as e:
            self.state = SystemState.ERROR
            logger.error(f"Hydration failed: {e}")

    def _get_history_from_db(self, symbol: str, tf: str, limit: int = 500) -> List[Dict]:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_data'")
            if not cursor.fetchone(): return []
            cursor.execute("SELECT * FROM market_data WHERE symbol = ? AND timeframe = ? ORDER BY tick_time DESC LIMIT ?", (symbol, tf, limit))
            return [dict(r) for r in cursor.fetchall()][::-1]
