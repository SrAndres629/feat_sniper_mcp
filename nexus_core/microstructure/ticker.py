"""
FEAT SNIPER: TICKER ENGINE
==========================
Handles high-frequency tick data streams and real-time buffers.
Replaces candle-based lag with sub-millisecond microstructure awareness.
"""

import numpy as np
import logging
from collections import deque
from typing import Dict, List, Optional
from threading import Lock

logger = logging.getLogger("FEAT.Ticker")

class TickBuffer:
    """
    High-performance Ring Buffer (deque) for raw market ticks.
    Ensures O(1) insertion and efficient sliding window math.
    """
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.lock = Lock()
        
        # Buffers
        self.times = deque(maxlen=max_size)
        self.bids = deque(maxlen=max_size)
        self.asks = deque(maxlen=max_size)
        self.bid_vols = deque(maxlen=max_size)
        self.ask_vols = deque(maxlen=max_size)
        self.last_prices = deque(maxlen=max_size) # (bid + ask) / 2
        
    def add_tick(self, tick: Dict):
        """
        Adds a new tick to the buffer.
        Expected keys: 'time', 'bid', 'ask', 'bid_vol', 'ask_vol'
        """
        with self.lock:
            self.times.append(tick.get('time', 0))
            bid = tick.get('bid', 0.0)
            ask = tick.get('ask', 0.0)
            self.bids.append(bid)
            self.asks.append(ask)
            self.bid_vols.append(tick.get('bid_vol', 0.0))
            self.ask_vols.append(tick.get('ask_vol', 0.0))
            self.last_prices.append((bid + ask) / 2.0)

    def get_arrays(self) -> Dict[str, np.ndarray]:
        """Returns buffers as numpy arrays for JIT math."""
        with self.lock:
            return {
                "prices": np.array(self.last_prices, dtype=np.float64),
                "bids": np.array(self.bids, dtype=np.float64),
                "asks": np.array(self.asks, dtype=np.float64),
                "bid_vols": np.array(self.bid_vols, dtype=np.float64),
                "ask_vols": np.array(self.ask_vols, dtype=np.float64)
            }

    @property
    def ready(self) -> bool:
        return len(self.last_prices) >= 50

# Global Ticker instance
tick_buffer = TickBuffer(max_size=1000)
