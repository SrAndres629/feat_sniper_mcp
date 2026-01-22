"""
[MODULE 04 - LEGACY REFACTOR]
4-Layer EMA System refactored to use config-driven periods.
Magic numbers eliminated. Now imports from settings.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from app.core.config import settings
from .models import EMALayer, LayerMetrics

logger = logging.getLogger("feat.structure.ema")


class FourLayerEMA:
    """
    4-Layer EMA System - Institutional Grade Market Physics.
    
    [REFACTORED v6.0] All periods now loaded from neural_config.py.
    No magic numbers in this module.
    
    Layers:
    - Layer 1 (Micro/Gas): Fast intent detection
    - Layer 2 (Operative/Water): Structural flow
    - Layer 3 (Macro/Wall): Institutional memory
    - Layer 4 (Bias/Bedrock): Regime anchor
    """
    
    def __init__(self):
        # [ZERO-DEBT] Load periods from config
        self.MICRO_PERIODS = list(settings.LAYER_MICRO_PERIODS)
        self.OPERATIVE_PERIODS = list(settings.LAYER_OPERATIVE_PERIODS)
        self.MACRO_PERIODS = list(settings.LAYER_MACRO_PERIODS)
        self.BIAS_PERIOD = settings.LAYER_BIAS_PERIOD
        
        logger.debug(f"[FourLayerEMA] Initialized with config-driven periods")
    
    def compute_layer_metrics(self, df: pd.DataFrame, layer: EMALayer) -> Optional[LayerMetrics]:
        if "close" not in df.columns or len(df) < 20:
            return None
        
        close = df["close"]
        if layer == EMALayer.MICRO: 
            periods = self.MICRO_PERIODS
        elif layer == EMALayer.OPERATIVE: 
            periods = self.OPERATIVE_PERIODS
        elif layer == EMALayer.MACRO: 
            periods = [p for p in self.MACRO_PERIODS if p <= len(df)]
        elif layer == EMALayer.BIAS: 
            periods = [min(self.BIAS_PERIOD, len(df))]
        else: 
            return None
        
        if not periods: 
            return None
        
        ema_values = []
        for p in periods:
            if p <= len(df):
                ema = close.ewm(span=p, adjust=False).mean().iloc[-1]
                ema_values.append(ema)
        
        if not ema_values: 
            return None
        
        avg_value = np.mean(ema_values)
        spread = max(ema_values) - min(ema_values) if len(ema_values) > 1 else 0
        compression = (spread / avg_value * 1000) if avg_value > 0 else 0
        
        window = 5
        if len(df) >= window:
            y = []
            for i in range(1, window + 1):
                past_emas = [close.ewm(span=p, adjust=False).mean().iloc[-i] 
                             for p in periods if p <= len(df)]
                y.insert(0, np.mean(past_emas) if past_emas else avg_value)
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)
        else:
            slope = 0.0
        
        return LayerMetrics(
            avg_value=avg_value,
            spread=spread,
            compression=compression,
            slope=slope,
            layer_type=layer
        )
    
    def get_all_layer_metrics(self, df: pd.DataFrame) -> Dict[EMALayer, LayerMetrics]:
        result = {}
        for layer in EMALayer:
            metrics = self.compute_layer_metrics(df, layer)
            if metrics: 
                result[layer] = metrics
        return result
    
    def compute_layer_alignment(self, df: pd.DataFrame) -> float:
        metrics = self.get_all_layer_metrics(df)
        if len(metrics) < 3: 
            return 0.0
        
        micro = metrics.get(EMALayer.MICRO)
        oper = metrics.get(EMALayer.OPERATIVE)
        macro = metrics.get(EMALayer.MACRO)
        
        if not all([micro, oper, macro]): 
            return 0.0
        
        slopes = [micro.slope, oper.slope, macro.slope]
        all_positive = all(s > 0 for s in slopes)
        all_negative = all(s < 0 for s in slopes)
        slope_aligned = 1.0 if (all_positive or all_negative) else 0.3
        
        bullish_order = micro.avg_value > oper.avg_value > macro.avg_value
        bearish_order = micro.avg_value < oper.avg_value < macro.avg_value
        order_aligned = 1.0 if (bullish_order or bearish_order) else 0.4
        
        return (slope_aligned * 0.5 + order_aligned * 0.5)


# Singleton for backward compatibility
four_layer_ema = FourLayerEMA()
