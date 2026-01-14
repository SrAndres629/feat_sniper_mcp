import pandas as pd
import numpy as np
from app.core.config import settings

def _smma(series: pd.Series, period: int) -> pd.Series:
    """
    Smoothed Moving Average (SMMA).
    Formula: SMMA(i) = (Sum - SMMA(i-1) + Close(i)) / period
    """
    return series.ewm(alpha=1/period, adjust=False).mean()

def calculate_feat_layers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula las capas de la Física de Capas Multifractales (FEAT).
    """
    if df.empty or len(df) < max(settings.LAYER_MACRO_PERIODS):
        return pd.DataFrame()

    results = pd.DataFrame(index=df.index)
    
    # --- LAYER 1: MICRO ---
    l1_smmas = []
    for p in settings.LAYER_MICRO_PERIODS:
        l1_smmas.append(_smma(df['close'], p))
    l1_df = pd.concat(l1_smmas, axis=1)
    results['L1_Mean'] = l1_df.mean(axis=1)
    results['L1_Width'] = l1_df.std(axis=1)
    
    # --- LAYER 2: OPERATIVE ---
    l2_smmas = []
    for p in settings.LAYER_OPERATIVE_PERIODS:
        l2_smmas.append(_smma(df['close'], p))
    l2_df = pd.concat(l2_smmas, axis=1)
    l2_mean = l2_df.mean(axis=1)
    results['Div_L1_L2'] = results['L1_Mean'] / l2_mean
    
    # --- LAYER 4: BIAS ---
    l4_smma = _smma(df['close'], settings.LAYER_BIAS_PERIOD)
    results['L4_Slope'] = l4_smma.pct_change(periods=3) * 100
    
    return results

def detect_regime(physics_df: pd.DataFrame) -> str:
    """
    Modular Engineering: Regime Detection Logic.
    """
    if physics_df.empty:
        return "NO_DATA"
        
    latest = physics_df.iloc[-1]
    
    # 1. Compression Check (Volatility Alert)
    rolling_width = physics_df['L1_Width'].rolling(100).mean().iloc[-1]
    if latest['L1_Width'] < rolling_width * 0.5:
        return "COMPRESSION"
        
    # 2. Expansion Check (Structural Stretch)
    if latest['Div_L1_L2'] > 1.05 or latest['Div_L1_L2'] < 0.95:
        return "EXPANSION"
        
    # 3. Gravity Check (Macro Trend)
    if abs(latest['L4_Slope']) > 0.05:
        return "TREND_GRAVITY"
        
    return "NOISE"

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculates Average True Range (ATR) for volatility measurement.
    """
    if df.empty or len(df) < period:
        return 0.0
        
    high_low = df['high'] - df['low']
    high_cp = np.abs(df['high'] - df['close'].shift())
    low_cp = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if not np.isnan(atr) else 0.0
