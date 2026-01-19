import numpy as np
import pandas as pd

class OrderFlowImbalance:
    """
    [MICROSTRUCTURE FLOW]
    Calculates Order Flow Imbalance (OFI) Proxy.
    
    Since we lack L2 Tick Data, we use the Candle Delta Proxy:
    OFI = (Close - Open) * Volume
    
    Positive OFI = Net Aggressive Buying
    Negative OFI = Net Aggressive Selling
    """
    
    def __init__(self, z_score_window: int = 50):
        self.window = z_score_window

    def calculate_proxy(self, opens: pd.Series, closes: pd.Series, volumes: pd.Series) -> float:
        """
        Returns the raw OFI for the latest candle.
        """
        try:
            c = closes.iloc[-1]
            o = opens.iloc[-1]
            v = volumes.iloc[-1]
            
            # Basic Delta
            # If C > O, we assume net buying.
            # If C < O, net selling.
            delta = (c - o)
            
            # We can normalize delta by ATR? No, keep raw for volume weighting.
            ofi = delta * v
            return float(ofi)
        except:
            return 0.0

    def calculate_cumulative_ofi(self, opens: pd.Series, closes: pd.Series, volumes: pd.Series, lookback: int = 10) -> float:
        """
        Cumulative OFI over N candles. Identifying persistent pressure.
        """
        if len(closes) < lookback:
            return 0.0
            
        o_slice = opens.iloc[-lookback:]
        c_slice = closes.iloc[-lookback:]
        v_slice = volumes.iloc[-lookback:]
        
        deltas = (c_slice - o_slice) * v_slice
        return float(deltas.sum())

    def get_ofi_z_score(self, current_ofi: float, history_ofi: np.ndarray) -> float:
        """
        Normalizes OFI to detect anomalies (Institutions stepping in).
        """
        if len(history_ofi) < 10:
            return 0.0
            
        mean = np.mean(history_ofi)
        std = np.std(history_ofi)
        
        if std == 0:
            return 0.0
            
        return (current_ofi - mean) / std
