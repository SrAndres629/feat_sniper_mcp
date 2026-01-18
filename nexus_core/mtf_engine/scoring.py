import numpy as np
import pandas as pd
from typing import Optional
from .models import Timeframe, TimeframeScore
from app.skills.market_physics import MarketRegime

def analyze_hydrodynamics(
    tf: Timeframe,
    candles: pd.DataFrame,
    regime: Optional[MarketRegime],
    parent_bias_z: float
) -> TimeframeScore:
    """
    Calculates score based on Fluid Dynamics (OFI, Impact, Liquidity).
    """
    result = TimeframeScore(
        timeframe=tf.value,
        score=0.0,
        direction=0,
        trend="NEUTRAL"
    )
    
    score = 0.0
    reasoning = []
    
    # --- 1. Institutional Logic Calculation (Dataframe Fallback) ---
    closes = candles['close'].values
    volumes = candles['tick_volume'].values
    
    delta_p = np.diff(closes)
    # Direction of trade based on candle color (naive tick rule approximation for candles)
    trade_dir = np.sign(delta_p) 
    ofi_proxy = trade_dir * volumes[1:]
    
    # Z-Score of OFI (Last 20 bars)
    ofi_window = ofi_proxy[-20:]
    ofi_mean = np.mean(ofi_window)
    ofi_std = np.std(ofi_window) + 1e-6
    current_ofi_z = (ofi_proxy[-1] - ofi_mean) / ofi_std
    
    result.ofi_z_score = current_ofi_z
    
    # --- 2. Regime Scoring ---
    
    # A. Flow Alignment (OFI matches Price Move)
    if current_ofi_z > 1.5:
        score += 0.2
        result.direction = 1
        reasoning.append("Strong Buying Pressure (OFI > 1.5σ)")
    elif current_ofi_z < -1.5:
        score += 0.2
        result.direction = -1
        reasoning.append("Strong Selling Pressure (OFI < -1.5σ)")
        
    # B. Liquidity Vacuum (Low Volume, High Move) -> Vacuum Run
    move_z = abs(delta_p[-1]) / (np.std(delta_p[-20:]) + 1e-6)
    vol_z = (volumes[-1] - np.mean(volumes[-20:])) / (np.std(volumes[-20:]) + 1e-6)
    
    if move_z > 2.0 and vol_z < 0:
        result.is_vacuum = True
        result.trend = "VACUUM_RUN"
        score += 0.3 # High potential
        reasoning.append("Liquidity Vacuum Detect (Fast Move/Low Res)")
        
    # C. Absorption (High Volume, Low Move) -> Reversal Warning
    if vol_z > 2.0 and move_z < 0.5:
        result.is_absorption = True
        result.trend = "ABSORPTION"
        score -= 0.1 # Penalty for direction, usually reversal
        reasoning.append("Absorption Detected (Hidden Wall)")
        
    # --- 3. Context Integration ---
    # Add bonus if aligned with Higher TF Flow
    if parent_bias_z != 0 and np.sign(result.ofi_z_score) == np.sign(parent_bias_z):
        score += 0.15
        reasoning.append(f"Aligned with Macro Flow")
        
    result.score = min(1.0, max(0.0, score))
    result.reasoning = reasoning
    
    # Set trend text based on score/dir
    if result.trend == "NEUTRAL":
        if result.score > 0.5:
            result.trend = "BULLISH_FLOW" if result.direction > 0 else "BEARISH_FLOW"
            
    return result
