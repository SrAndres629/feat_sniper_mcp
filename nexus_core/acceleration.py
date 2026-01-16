import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

logger = logging.getLogger("feat.acceleration")

class MomentumVector:
    """
    [A] COMPONENT - ACCELERATION (The Physics Engine)
    Calculates Newtonian physics of price: Velocity and Acceleration.
    """
    def __init__(self, threshold: float = 1.5):
        self.threshold = threshold

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Logic: 
        Velocity (V) = Delta Price
        Acceleration (Acc) = Delta Velocity
        """
        if len(df) < 5:
            return {"velocity": 0.0, "acceleration": 0.0, "is_valid": False, "is_trap": False}
            
        prices = df['close'].values
        
        # 1st Derivative: Velocity
        velocity = np.diff(prices)
        # 2nd Derivative: Acceleration
        acceleration = np.diff(velocity)
        
        v_current = velocity[-1]
        a_current = acceleration[-1]
        
        # Rule: Valid breakout requires Acceleration > Threshold
        # We normalize by ATR for asset-agnostic thresholding if possible, 
        # but here we use the specific config threshold.
        is_breakout = abs(a_current) > self.threshold
        
        # DIVERGENCE_TRAP: Price moves (V is high) but Acceleration is low or negative
        # Signifies absorption or lack of institutional follow-through
        is_trap = abs(v_current) > (self.threshold * 0.8) and abs(a_current) < (self.threshold * 0.2)
        
        return {
            "velocity": float(v_current),
            "acceleration": float(a_current),
            "is_valid": is_breakout and not is_trap,
            "is_trap": is_trap,
            "status": "ACCELERATING" if is_breakout else ("DIVERGENCE_TRAP" if is_trap else "INERTIA")
        }

class AccelerationEngine:
    def __init__(self, config: Dict[str, Any] = None):
        print("[Physics] Sigma Monitor ON (Velocity/Acceleration vectors)")
        self.config = config or {
            "atr_w": 14,
            "vol_w": 20,
            "score_th": 0.70,
            "sigma_th": 3.0, # Alert threshold for sigma
            "accel_th": 1.5, # Newtonian threshold
            "weights": {
                "w1": 0.4, # Displacement
                "w2": 0.3, # Volume Z-Score
                "w3": 0.2, # FVG Presence
                "w4": 0.1  # Velocity
            }
        }
        self.momentum_v = MomentumVector(threshold=self.config["accel_th"])

    def calculate_momentum_vector(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Gate A: Velocity/Acceleration Vector.
        Calculates Newtonian physics of price movement.
        """
        return self.momentum_v.analyze(df)

    def compute_acceleration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main API for context-aware acceleration analysis.
        Incorporates FEAT acceleration logic: Momentum, RVOL, CVD Proxy.
        """
        res = pd.DataFrame(index=df.index)
        
        if len(df) < self.config['atr_w'] + 1:
            return res

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
        bull_fvg = (lows > highs.shift(2)).astype(float)
        bear_fvg = (highs < lows.shift(2)).astype(float)
        res['fvg_created'] = (bull_fvg + bear_fvg).clip(0, 1)
        
        # 4. Velocity (Delta P / Delta T)
        res['velocity'] = closes.diff(1).abs() / (atr + 1e-9)
        
        # 5. Composite Score
        w = self.config['weights']
        linear_comb = (w['w1'] * res['disp_norm'] + 
                       w['w2'] * res['vol_z'] + 
                       w['w3'] * res['fvg_created'] + 
                       w['w4'] * res['velocity'])
        
        res['accel_score'] = 1 / (1 + np.exp(-linear_comb))
        
        # 6. Classification
        res['accel_type'] = self._classify_accel(res)
        
        # 7. Initiative Candle (Vela de Iniciativa - FEAT CORE)
        # Rule: Volume > 2.5 * AvgVol AND Body > 1.5 * ATR
        is_initiative = (res['rvol'] > 2.5) & (res['candle_momentum'] > 1.5)
        res['is_initiative'] = is_initiative.astype(float)
        
        # 8. Newtonian Delta (1st & 2nd Derivatives)
        # This aligns the component [A] with strict derivative logic
        velocity = closes.diff(1)
        acceleration = velocity.diff(1)
        res['newton_accel'] = acceleration / (atr + 1e-9) # Normalized acceleration
        
        # 9. Entry Trigger (Flag)
        # Optimized for Operación Navaja: Initiative Candles OR High Newtonian Acceleration
        res['accel_flag'] = (is_initiative | 
                            (abs(res['newton_accel']) > self.config['accel_th']) |
                            ((res['accel_score'] > self.config['score_th']) & 
                             (res['accel_type'].isin(['breakout', 'rejection'])))).astype(int)
        
        # 10. Trap Detection
        # If absolute velocity is high but acceleration is low -> Trap
        res['is_trap'] = ((res['velocity'] > 1.0) & (abs(res['newton_accel']) < 0.2)).astype(int)
        
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
