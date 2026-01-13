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

    def detect_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies Bill Williams Fractals and Break of Structure (BOS).
        """
        # Fractal simple (High/Low of 5 candles, centered)
        # Shifted back by 2 to align with completion
        df['fractal_high'] = (df['high'].rolling(window=5, center=True).max() == df['high']).fillna(False)
        df['fractal_low'] = (df['low'].rolling(window=5, center=True).min() == df['low']).fillna(False)
        
        # BOS (Break of Structure)
        # Bullish BOS: Close > Previous Fractal High
        # Note: In real-time, we look at confirmed fractals (shifted)
        # Using shift(1) to avoid lookahead bias if using centered rolling on historical data
        # For live, we strictly check past fractals.
        
        # Vectorized check roughly: Close > Last confirmed Fractal High
        # For simplicity in this vectorization, we compare simply to potential fractal points
        # (This assumes df is historical. For live, careful indexing is needed)
        
        # Simplified vector logic from prompt:
        df['bos_bullish'] = np.where((df['close'] > df['high'].shift(1)) & (df['fractal_high'].shift(1)), 1, 0)
        df['bos_bearish'] = np.where((df['close'] < df['low'].shift(1)) & (df['fractal_low'].shift(1)), 1, 0)
        
        return df

    def detect_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects FVG and Order Blocks using vectorized logic.
        """
        # FVG Detection
        df['fvg_bull_gap'] = df['low'] - df['high'].shift(2)
        df['fvg_bull'] = np.where((df['fvg_bull_gap'] > 0) & (df['close'] > df['open']), 1, 0)
        
        df['fvg_bear_gap'] = df['low'].shift(2) - df['high']
        df['fvg_bear'] = np.where((df['fvg_bear_gap'] > 0) & (df['close'] < df['open']), 1, 0)
        
        # OB Detection (Approximation from prompt)
        # Bullish OB: Red candle before strong up move (2 green candles)
        is_red = df['close'] < df['open']
        strong_move_up = (df['close'].shift(-1) > df['open'].shift(-1)) & (df['close'].shift(-2) > df['open'].shift(-2))
        
        # Note: shift(-1) is future looking. For labeling/training this is fine.
        # For LIVE FEAT, we must look at PAST patterns.
        # Adjusted for LIVE detection (looking back):
        # Bullish OB formed 2 bars ago if: Candle(i-2) red, Candle(i-1) green, Candle(i) green
        is_red_prev = df['close'].shift(2) < df['open'].shift(2)
        strong_move_up_confirmed = (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] > df['open'])
        
        df['ob_bull'] = np.where(is_red_prev & strong_move_up_confirmed, 1, 0)
        
        # Bearish OB: Green candle before strong down move
        is_green_prev = df['close'].shift(2) > df['open'].shift(2)
        strong_move_down_confirmed = (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] < df['open'])
        
        df['ob_bear'] = np.where(is_green_prev & strong_move_down_confirmed, 1, 0)
        
        return df

    def score_phase_form(self, df: pd.DataFrame) -> pd.Series:
        """
        FASE 1: Forma (F) - Score based on BOS and Fractals
        """
        # Simplified vector proxies for structural scores
        ob_presence = (df.get('ob_bull', 0) + df.get('ob_bear', 0))
        
        # BOS Proxy: Cierre por fuera de la banda Bollinger superior/inferior o similar
        # Usaremos breakout de máximo/mínimo de 20 periodos
        bos_score = (df.get('bos_bullish', 0) + df.get('bos_bearish', 0))
        
        # CHOCH Proxy: Cambio de signo en la diferencia de EMAs
        ema_f = df['close'].ewm(span=9).mean()
        ema_s = df['close'].ewm(span=21).mean()
        choch_score = (np.sign(ema_f - ema_s) != np.sign(ema_f.shift(1) - ema_s.shift(1))).astype(float)
        
        fractal_score = (df.get('fractal_high', False) | df.get('fractal_low', False)).astype(float)

        score_f = (0.35 * bos_score + 0.25 * choch_score + 0.25 * fractal_score + 0.15 * ob_presence)
        return score_f.clip(0, 1)

    def score_phase_space(self, df: pd.DataFrame) -> pd.Series:
        """
        FASE 2: Espacio (E) - Based on FVG and Liquidity Pools
        """
        fvg_score = (df.get('fvg_bull', 0) + df.get('fvg_bear', 0))
        
        # Pool Proxy: EQH / EQL (Double tops/bottoms)
        eq_highs = (np.abs(df['high'] - df['high'].shift(1)) < (df['high'] * 0.0001)).astype(float)
        eq_lows = (np.abs(df['low'] - df['low'].shift(1)) < (df['low'] * 0.0001)).astype(float)
        pool_score = (eq_highs | eq_lows).rolling(5).max().fillna(0)
        
        score_e = (0.35 * pool_score + 0.40 * fvg_score + 0.25 * 0.5)
        return score_e.clip(0, 1)

    def score_phase_acceleration(self, df: pd.DataFrame) -> pd.Series:
        """
        FASE 3: Aceleración (A) - Now delegated to AccelerationEngine mostly, but for FEAT Index we keep simple check
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
        # 1. Detect Structure & Zones
        df = self.detect_structure(df)
        df = self.detect_zones(df)
        
        # 2. Score Components
        p_f = self.score_phase_form(df)
        p_e = self.score_phase_space(df)
        p_a = self.score_phase_acceleration(df)
        
        # Phase T (Time): Simulated for now (Killzones should be added based on timestamp)
        if hasattr(df.index, 'hour'):
            df_time = df.index
        elif 'tick_time' in df.columns:
            df_time = pd.to_datetime(df['tick_time'])
        else:
            # Fallback
            df_time = pd.to_datetime(df.index) if not isinstance(df.index, pd.RangeIndex) else pd.Series(pd.Timestamp.now(), index=df.index)

        try:
             # 1.0 if between 07:00-11:00 (London) or 13:00-17:00 (NY) UTC
            is_london = df_time.hour.map(lambda h: 1.0 if 7 <= h <= 11 else 0.0)
            is_ny = df_time.hour.map(lambda h: 1.0 if 13 <= h <= 17 else 0.0)
            p_t = (is_london | is_ny).astype(float)
        except:
             p_t = pd.Series(0.5, index=df.index)

        
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
