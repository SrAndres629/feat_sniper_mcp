import pandas as pd
import numpy as np
from typing import Dict, Any

def compute_acceleration_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Main API for context-aware acceleration analysis.
    Incorporates FEAT acceleration logic: Momentum, RVOL, CVD Proxy.
    """
    res = pd.DataFrame(index=df.index)
    
    if len(df) < config['atr_w'] + 1:
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
    atr = tr.rolling(window=config['atr_w']).mean()
    
    res['disp_norm'] = (closes - opens).abs() / (atr + 1e-9)
    
    # 2. Volume Z-Score (Energy)
    vol_mean = volumes.rolling(window=config['vol_w']).mean()
    vol_std = volumes.rolling(window=config['vol_w']).std()
    res['vol_z'] = (volumes - vol_mean) / (vol_std + 1e-9)
    
    # 3. FVG Presence (Space) - Simple gaps detection
    # Gap Up: Low > Prev High
    # Gap Down: High < Prev Low
    res['fvg_bull'] = (lows > highs.shift(1)).astype(int)
    res['fvg_bear'] = (highs < lows.shift(1)).astype(int)
    
    # 4. Composite Acceleration Score (Weighted)
    # Score = w1*Disp + w2*VolZ + w3*FVG
    w = config['weights']
    
    # FVG term (1 if any FVG, else 0)
    fvg_term = (res['fvg_bull'] + res['fvg_bear']).clip(upper=1.0)
    
    res['accel_score'] = (
        (res['disp_norm'].clip(upper=3.0) * w['w1']) +
        (res['vol_z'].clip(upper=3.0) * w['w2']) +
        (fvg_term * w['w3'])
    )
    
    # Normalize score to 0-1 range roughly (soft sigmoid not needed, just clip)
    # A score > 0.7 usually indicates Strong Impulse
    res['is_impulse'] = res['accel_score'] > config['score_th']
    
    return res
