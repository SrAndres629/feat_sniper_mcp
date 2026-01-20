import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def _calculate_ofi_tick(bid_prices: np.ndarray, ask_prices: np.ndarray, 
                        bid_sizes: np.ndarray, ask_sizes: np.ndarray) -> float:
    """
    Calculates OFI (Order Flow Imbalance) from tick data (L1).
    
    OFI = e_t * q_t
    where e_t is direction of best bid/ask change
    and q_t is the size change.
    """
    n = len(bid_prices)
    if n < 2:
        return 0.0
        
    ofi_sum = 0.0
    
    for i in range(1, n):
        # Bid Side
        if bid_prices[i] > bid_prices[i-1]:
            ofi_sum += bid_sizes[i]
        elif bid_prices[i] < bid_prices[i-1]:
            ofi_sum -= bid_sizes[i-1]
        else:
             # Price unchanged, size change
             ofi_sum += (bid_sizes[i] - bid_sizes[i-1])
             
        # Ask Side (Inverted logic for supply)
        if ask_prices[i] > ask_prices[i-1]:
            ofi_sum -= ask_sizes[i-1]
        elif ask_prices[i] < ask_prices[i-1]:
            ofi_sum += ask_sizes[i]
        else:
             ofi_sum -= (ask_sizes[i] - ask_sizes[i-1])
             
    return ofi_sum

@jit(nopython=True, cache=True)
def _calculate_proxy_ofi(opens: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> float:
    """
    JIT-optimized Proxy OFI for Candle Data.
    OFI ~= (Close - Open) * Volume
    """
    n = len(closes)
    if n == 0:
        return 0.0
        
    deltas = closes - opens
    ofi = np.sum(deltas * volumes)
    return ofi

class OrderFlowImbalance:
    """
    [MICROSTRUCTURE FLOW]
    Calculates Order Flow Imbalance (OFI).
    Supports both Tick (L1) and Candle Proxy modes.
    """
    
    def __init__(self, z_score_window: int = 50):
        self.window = z_score_window
        
        # OFI History for Z-Score
        self.history = [] 

    def calculate_tick_ofi(self, bid_p: np.ndarray, ask_p: np.ndarray, 
                           bid_s: np.ndarray, ask_s: np.ndarray) -> float:
        """High-precision Tick OFI."""
        return _calculate_ofi_tick(bid_p, ask_p, bid_s, ask_s)

    def calculate_proxy(self, opens: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> float:
        """
        Returns the raw OFI using Candle Proxy.
        Input: Numpy arrays.
        """
        return _calculate_proxy_ofi(opens, closes, volumes)

    def update_and_normalize(self, current_ofi: float) -> float:
        """
        Updates history and returns Z-Score.
        """
        self.history.append(current_ofi)
        if len(self.history) > self.window:
            self.history.pop(0)
            
        return self._get_z_score(current_ofi)

    def _get_z_score(self, current_ofi: float) -> float:
        """
        Normalizes OFI to detect anomalies.
        """
        if len(self.history) < 10:
            return 0.0
            
        arr = np.array(self.history)
        mean = np.mean(arr)
        std = np.std(arr)
        
        if std < 1e-9:
            return 0.0
            
        return (current_ofi - mean) / std
