import pandas as pd
import numpy as np
from app.core.config import settings

def _smma(series: pd.Series, period: int) -> pd.Series:
    """
    Smoothed Moving Average (SMMA).
    Formula: SMMA(i) = (Sum - SMMA(i-1) + Close(i)) / period
    """
    return series.ewm(alpha=1/period, adjust=False).mean()

def robust_scaler(value: float, median: float, iqr: float) -> float:
    """
    Normalización Robusta: (x - mediana) / IQR.
    Inmune a outliers de volatilidad extrema.
    """
    if iqr == 0: return 0.0
    return (value - median) / iqr

def calculate_feat_layers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula las capas FEAT optimizadas para latencia < 5ms.
    """
    if df.empty or len(df) < settings.LAYER_BIAS_PERIOD:
        return pd.DataFrame()

    # Optimization: Work only on the tail
    window = 50
    tail_df = df.tail(settings.LAYER_BIAS_PERIOD + window)
    
    # Pre-calculate EWMs without concat overhead
    # Layer 1 (Micro)
    l1_data = np.zeros((len(tail_df), len(settings.LAYER_MICRO_PERIODS)))
    for i, p in enumerate(settings.LAYER_MICRO_PERIODS):
        l1_data[:, i] = tail_df['close'].ewm(alpha=1.0/p, adjust=False).mean().values
    
    # Layer 2 (Operative)
    l2_data = np.zeros((len(tail_df), len(settings.LAYER_OPERATIVE_PERIODS)))
    for i, p in enumerate(settings.LAYER_OPERATIVE_PERIODS):
        l2_data[:, i] = tail_df['close'].ewm(alpha=1.0/p, adjust=False).mean().values
    
    # Layer 4 (Bias/Gravity)
    l4_ewm = tail_df['close'].ewm(alpha=1.0/settings.LAYER_BIAS_PERIOD, adjust=False).mean().values
    
    # Construct Output DataFrame efficiently
    res_df = pd.DataFrame(index=tail_df.index)
    res_df['L1_Mean'] = np.mean(l1_data, axis=1)
    res_df['L1_Width'] = np.std(l1_data, axis=1)
    res_df['L4_Slope'] = pd.Series(l4_ewm, index=tail_df.index).pct_change(periods=3) * 100
    res_df['Div_L1_L2'] = res_df['L1_Mean'] / np.mean(l2_data, axis=1).clip(min=1e-6)
    res_df['close'] = tail_df['close'].values # Required for PvP Divergence
    
    return res_df.tail(window).fillna(0.0)

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
def detect_divergence(df: pd.DataFrame, window: int = 10) -> str:
    """
    [PVP ALPHA] Detects Liquidity Traps via divergence between Price and L4_Slope.
    L4_Slope represents 'Physics Gravity'. If Price rises but Gravity falls, it's a Trap.
    """
    if df.empty or len(df) < window + 2:
        return "NEUTRAL"
        
    recent = df.tail(window)
    price_change = recent['close'].iloc[-1] - recent['close'].iloc[0]
    slope_change = recent['L4_Slope'].iloc[-1] - recent['L4_Slope'].iloc[0]
    
    # 1. Bearish Divergence (Retail Trap / Top)
    if price_change > 0 and slope_change < -0.01:
        return "BULL_TRAP"
        
    # 2. Bullish Divergence (Retail Trap / Bottom)
    if price_change < 0 and slope_change > 0.01:
        return "BEAR_TRAP"
        
    return "NEUTRAL"
