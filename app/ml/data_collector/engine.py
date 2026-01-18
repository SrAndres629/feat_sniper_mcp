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
        self.samples_collected = 0
        self._lock = threading.Lock()
        self.state = SystemState.BOOTING
        self.hydration_progress = 0.0

    def compute_features(self, candle: Dict, inds: Dict) -> Dict[str, float]:
        phys = inds.get("physics", {})
        return {
            "close": float(candle.get("close", 0)), "open": float(candle.get("open", 0)),
            "high": float(candle.get("high", 0)), "low": float(candle.get("low", 0)),
            "volume": float(candle.get("volume", 0)), "rsi": float(inds.get("rsi", 50.0)),
            "atr": float(inds.get("atr", 0.001)),
            "ema_fast": float(inds.get("ema_fast", candle.get("close", 0))),
            "ema_slow": float(inds.get("ema_slow", candle.get("close", 0))),
            "fsm_state": int(inds.get("fsm_state", 0)), "feat_score": float(inds.get("feat_score", 0.0)),
            "liquidity_ratio": float(inds.get("liquidity_ratio", 1.0)),
            "volatility_zscore": float(inds.get("volatility_zscore", 0.0)),
            "momentum_kinetic_micro": float(inds.get("momentum_kinetic_micro", 0.0)),
            "entropy_coefficient": float(inds.get("entropy_coefficient", 0.0)),
            "cycle_harmonic_phase": float(inds.get("cycle_harmonic_phase", 0.0)),
            "institutional_mass_flow": float(inds.get("institutional_mass_flow", 0.0)),
            "volatility_regime_norm": float(inds.get("volatility_regime_norm", 0.0)),
            "acceptance_ratio": float(inds.get("acceptance_ratio", 0.0)),
            "wick_stress": float(inds.get("wick_stress", 0.0)),
            "poc_z_score": float(inds.get("poc_z_score", 0.0)),
            "cvd_acceleration": float(inds.get("cvd_acceleration", 0.0)),
            "micro_comp": float(phys.get("l1_c", inds.get("micro_comp", 0.5))),
            "micro_slope": float(phys.get("l1_s", inds.get("micro_slope", 0.0))),
            "oper_slope": float(phys.get("l2_s", inds.get("oper_slope", 0.0))),
            "macro_slope": float(phys.get("l3_s", inds.get("macro_slope", 0.0))),
            "bias_slope": float(phys.get("l4_s", inds.get("bias_slope", 0.0))),
            "fan_bullish": float(1.0 if (phys.get("l1_m", 0) > phys.get("l2_m", 0) > phys.get("l3_m", 0)) else 0.0)
        }

    def collect(self, symbol: str, candle: Dict, inds: Dict, tf: str = "M1"):
        feats = self.compute_features(candle, inds)
        rec = {"tick_time": candle.get("time") or datetime.now(timezone.utc).isoformat(), "symbol": symbol, "timeframe": tf, **feats}
        with self._lock:
            self.batch.append(rec)
            if tf == "M1":
                if symbol not in self.resamplers: self.resamplers[symbol] = Resampler(symbol)
                for tf_out, tf_c in self.resamplers[symbol].push_tick(rec):
                    hist = self.resamplers[symbol].history[tf_out]
                    if len(hist) < 14: continue
                    cl, hi, lo = np.array([c["close"] for c in hist]), np.array([c["high"] for c in hist]), np.array([c["low"] for c in hist])
                    rsi, atr = calculate_rsi_numpy(cl, 14), calculate_atr_numpy(hi, lo, cl, 14)
                    fs = 0.0
                    if len(hist) >= 20:
                        fdf = calculate_feat_layers(pd.DataFrame(hist))
                        if not fdf.empty: fs = float(fdf['L1_Mean'].iloc[-1])
                    self.batch.append({"symbol": symbol, "timeframe": tf_out, **tf_c, "tick_time": tf_c["tick_time"], "rsi": rsi, "atr": atr, "ema_fast": tf_c["close"], "ema_slow": tf_c["close"], "fsm_state": 0, "feat_score": fs, "liquidity_ratio": 1.0, "volatility_zscore": 0.0})
            if len(self.batch) >= BATCH_SIZE: self._flush_batch()

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
        with self._lock: self._flush_batch()

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
