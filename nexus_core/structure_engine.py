"""
FEAT NEXUS: STRUCTURE ENGINE (FourJarvis Protocol)
==================================================
Vectorized Price Action & Institutional Structure Quantification.
Author: Antigravity Prime
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from numba import njit

class StructureEngine:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            "va_pct": 0.70,
            "atr_window": 14,
            "weights": {
                "F": 0.30, "E": 0.30, "A": 0.20, "T": 0.20
            }
        }

    def detect_fvg(self, df: pd.DataFrame) -> pd.Series:
        """
        Fair Value Gap Detection (Vectorized).
        Gap between Candle(i-1) High/Low and Candle(i+1) Low/High.
        """
        # Bullish FVG: Low(i+1) > High(i-1)
        bull_fvg = (df['low'].shift(-1) > df['high'].shift(1)).astype(float)
        # Bearish FVG: High(i+1) < Low(i-1)
        bear_fvg = (df['high'].shift(-1) < df['low'].shift(1)).astype(float)
        
        return bull_fvg - bear_fvg # +1 for bull, -1 for bear

    def detect_order_blocks(self, df: pd.DataFrame) -> pd.Series:
        """
        Simplified Vectorized OB Detection.
        Defined as the last opposite candle before a strong structural breakout (BOS).
        """
        # Strong move: Range > 1.5 * ATR
        atr = (df['high'] - df['low']).rolling(14).mean()
        strong_move = (df['close'].diff().abs() > 1.5 * atr).astype(float)
        
        # Last opposite candle
        is_up = (df['close'] > df['open'])
        is_down = (df['close'] < df['open'])
        
        # Bullish OB: Last down candle before strong up move
        bull_ob = (is_down.shift(1) & (df['close'] > df['close'].shift(1)) & (strong_move > 0)).astype(float)
        # Bearish OB: Last up candle before strong down move
        bear_ob = (is_up.shift(1) & (df['close'] < df['close'].shift(1)) & (strong_move > 0)).astype(float)
        
        return bull_ob - bear_ob

    def score_phase_form(self, df: pd.DataFrame) -> pd.Series:
        """
        FASE 1: Forma (F) - Score Structure (0.35 BOS + 0.25 CHOCH + 0.25 Fractal + 0.15 OB)
        """
        # Simplified vector proxies for structural scores
        ob_presence = self.detect_order_blocks(df).abs()
        
        # BOS Proxy: Cierre por fuera de la banda Bollinger superior/inferior o similar
        # Usaremos breakout de máximo/mínimo de 20 periodos
        hh_20 = df['high'].rolling(20).max().shift(1)
        ll_20 = df['low'].rolling(20).min().shift(1)
        bos_score = ((df['close'] > hh_20) | (df['close'] < ll_20)).astype(float)
        
        # CHOCH Proxy: Cambio de signo en la diferencia de EMAs
        ema_f = df['close'].ewm(span=9).mean()
        ema_s = df['close'].ewm(span=21).mean()
        choch_score = (np.sign(ema_f - ema_s) != np.sign(ema_f.shift(1) - ema_s.shift(1))).astype(float)
        
        score_f = (0.35 * bos_score + 0.25 * choch_score + 0.25 * 0.5 + 0.15 * ob_presence)
        return score_f.clip(0, 1)

    def score_phase_space(self, df: pd.DataFrame) -> pd.Series:
        """
        FASE 2: Espacio (E) - Liquidez (0.35 Pools + 0.25 FVG + 0.25 Conflu + 0.15 Prox)
        """
        fvg_score = self.detect_fvg(df).abs()
        
        # Pool Proxy: EQH / EQL (Double tops/bottoms)
        eq_highs = (np.abs(df['high'] - df['high'].shift(1)) < (df['high'] * 0.0001)).astype(float)
        eq_lows = (np.abs(df['low'] - df['low'].shift(1)) < (df['low'] * 0.0001)).astype(float)
        pool_score = (eq_highs | eq_lows).rolling(5).max().fillna(0)
        
        score_e = (0.35 * pool_score + 0.25 * fvg_score + 0.40 * 0.5)
        return score_e.clip(0, 1)

    def score_phase_acceleration(self, df: pd.DataFrame) -> pd.Series:
        """
        FASE 3: Aceleración (A) - Volumen y Dinámica (0.5 Vol + 0.3 Speed + 0.2 Follow)
        """
        vol_mean = df['volume'].rolling(20).mean()
        vol_score = (df['volume'] > 1.5 * vol_mean).astype(float)
        
        # Speed: Range expansion
        atr = (df['high'] - df['low']).rolling(14).mean()
        speed_score = ((df['high'] - df['low']) > 1.2 * atr).astype(float)
        
        score_a = (0.5 * vol_score + 0.3 * speed_score + 0.2 * 0.5)
        return score_a.clip(0, 1)

    def compute_feat_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final FEAT Index calculation (0-100).
        """
        p_f = self.score_phase_form(df)
        p_e = self.score_phase_space(df)
        p_a = self.score_phase_acceleration(df)
        
        # Phase T (Time): Simulated for now (Killzones should be added based on timestamp)
        # 1.0 if between 07:00-11:00 (London) or 13:00-17:00 (NY) UTC
        df_time = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['tick_time'])
        is_london = df_time.hour.map(lambda h: 1.0 if 7 <= h <= 11 else 0.0)
        is_ny = df_time.hour.map(lambda h: 1.0 if 13 <= h <= 17 else 0.0)
        p_t = (is_london | is_ny).astype(float)
        
        w = self.config["weights"]
        p_feat = (w["F"] * p_f + w["E"] * p_e + w["A"] * p_a + w["T"] * p_t)
        
        results = pd.DataFrame(index=df.index)
        results['feat_form'] = p_f
        results['feat_space'] = p_e
        results['feat_acceleration'] = p_a
        results['feat_time'] = p_t
        results['feat_index'] = (p_feat * 100).round(2)
        
        return results

structure_engine = StructureEngine()
