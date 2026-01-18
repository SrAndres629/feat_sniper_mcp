import logging
import numpy as np
from collections import deque
from typing import Dict, Optional
from app.core.config import settings

logger = logging.getLogger("ML.Fractal")

class HurstBuffer:
    """
    Circular buffer for real-time Hurst coefficient calculation.
    Manages data sufficiency state and caching.
    """
    BUFFER_SIZE = settings.HURST_BUFFER_SIZE
    MIN_SAMPLES = settings.HURST_MIN_SAMPLES
    UPDATE_EVERY_N = settings.HURST_UPDATE_EVERY_N
    
    STATE_INSUFFICIENT = "DATA_INSUFFICIENT"
    STATE_READY = "READY"
    
    def __init__(self):
        self.buffers: Dict[str, deque] = {}
        self.cached_hurst: Dict[str, float] = {}
        self.update_counters: Dict[str, int] = {}
        self.state: Dict[str, str] = {}
        logger.info(f"Using Hurst Buffer: Size={self.BUFFER_SIZE}")
    
    def push(self, symbol: str, close_price: float) -> None:
        if symbol not in self.buffers:
            self.buffers[symbol] = deque(maxlen=self.BUFFER_SIZE)
            self.update_counters[symbol] = 0
            self.cached_hurst[symbol] = None
            self.state[symbol] = self.STATE_INSUFFICIENT
        
        self.buffers[symbol].append(close_price)
        self.update_counters[symbol] += 1
        
        if len(self.buffers[symbol]) >= self.MIN_SAMPLES:
            self.state[symbol] = self.STATE_READY
    
    def get_prices(self, symbol: str) -> np.ndarray:
        if symbol not in self.buffers or len(self.buffers[symbol]) < self.MIN_SAMPLES:
            raise ValueError(f"Insufficient data for {symbol}")
        return np.array(self.buffers[symbol])
    
    def should_recalculate(self, symbol: str) -> bool:
        if symbol not in self.update_counters: return False
        return self.update_counters[symbol] >= self.UPDATE_EVERY_N
    
    def reset_counter(self, symbol: str) -> None:
        self.update_counters[symbol] = 0
    
    def set_cached_hurst(self, symbol: str, hurst: float) -> None:
        self.cached_hurst[symbol] = hurst
    
    def get_cached_hurst(self, symbol: str) -> Optional[float]:
        return self.cached_hurst.get(symbol)
    
    def is_data_insufficient(self, symbol: str) -> bool:
        return self.state.get(symbol, self.STATE_INSUFFICIENT) == self.STATE_INSUFFICIENT
