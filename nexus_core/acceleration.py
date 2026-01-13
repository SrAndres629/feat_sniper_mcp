"""
FEAT NEXUS: ACCELERATION ENGINE (A)
====================================
Validation of institutional intent through Velocity, Volume, and Displacement.
Author: Antigravity Prime
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

class AccelerationEngine:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            "atr_w": 14,
            "vol_w": 20,
            "score_th": 0.70,
            "weights": {
                "w1": 0.4, # Displacement
                "w2": 0.3, # Volume Z-Score
                "w3": 0.2, # FVG Presence
                "w4": 0.1  # Velocity
            }
        }

    def compute_acceleration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main API for context-aware acceleration analysis.
        Incorporates FEAT acceleration logic: Momentum, RVOL, CVD Proxy.
        """
        res = pd.DataFrame(index=df.index)
        
        # 1. Displacement (Normalized by ATR)
        highs = df['high']
        lows = df['low']
        closes = df['close']
        opens = df['open']
        volumes = df['volume']
        
        # ATR Calculation
        tr = np.maximum(highs - lows, 
                        np.maximum(abs(highs - closes.shift(1)), 
                                   abs(lows - closes.shift(1))))
        atr = tr.rolling(window=self.config['atr_w']).mean()
        
        res['disp_norm'] = (closes - opens).abs() / (atr + 1e-9)
        
        # 2. Volume Z-Score (Effort)
        vol_mu = volumes.rolling(window=self.config['vol_w']).mean()
        vol_std = volumes.rolling(window=self.config['vol_w']).std()
        res['vol_z'] = (volumes - vol_mu) / (vol_std + 1e-9)
        
        # --- NEW FEAT METRICS FROM PROMPT ---
        # Candle Momentum: (Abs Body) / ATR
        res['candle_momentum'] = (abs(closes - opens)) / (atr + 1e-9)
        
        # RVOL (Relative Volume): Volume / Moving Average Volume
        res['rvol'] = volumes / (vol_mu + 1e-9)
        
        # CVD Proxy: if Close >= Open -> +Vol, else -Vol
        delta_proxy = np.where(closes >= opens, volumes, -volumes)
        res['cvd_proxy'] = pd.Series(delta_proxy, index=df.index).cumsum()
        # ------------------------------------
        
        # 3. FVG Recognition (Structural gap)
        # Bullish Gap: Low(i) > High(i-2)
        bull_fvg = (lows > highs.shift(2)).astype(float)
        # Bearish Gap: High(i) < Low(i-2)
        bear_fvg = (highs < lows.shift(2)).astype(float)
        res['fvg_created'] = (bull_fvg + bear_fvg).clip(0, 1)
        
        # 4. Velocity (Delta P / Delta T)
        res['velocity'] = closes.diff(1).abs() / (atr + 1e-9)
        
        # 5. Composite Score (Logit-like)
        w = self.config['weights']
        # Incorporate new metrics optionally or stick to config weights. 
        # For now, map disp_norm ~ candle_momentum roughly.
        linear_comb = (w['w1'] * res['disp_norm'] + 
                       w['w2'] * res['vol_z'] + 
                       w['w3'] * res['fvg_created'] + 
                       w['w4'] * res['velocity'])
        
        res['accel_score'] = 1 / (1 + np.exp(-linear_comb))
        
        # 6. Classification
        res['accel_type'] = self._classify_accel(res)
        
        # 7. Entry Trigger (Flag)
        res['accel_flag'] = ((res['accel_score'] > self.config['score_th']) & 
                             (res['accel_type'].isin(['breakout', 'rejection']))).astype(int)
        
        return res

    def _classify_accel(self, res: pd.DataFrame) -> pd.Series:
        """
        Categorizes acceleration into institutional types.
        """
        conditions = [
            (res['accel_score'] > 0.8) & (res['vol_z'] > 2.0) & (res['disp_norm'] > 1.5), # Breakout
            (res['accel_score'] > 0.6) & (res['vol_z'] > 1.5) & (res['disp_norm'] < 0.5), # Rejection (Abs)
            (res['accel_score'] > 0.85) & (res['vol_z'] > 3.0) & (res['velocity'] > 2.0)  # Climax
        ]
        choices = ['breakout', 'rejection', 'climax']
        
        return pd.Series(np.select(conditions, choices, default='normal'), index=res.index)

acceleration_engine = AccelerationEngine()
