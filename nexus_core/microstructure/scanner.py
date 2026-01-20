"""
FEAT SNIPER: MICROSTRUCTURE SCANNER
===================================
Orchestrates the Physics Layer sensors (Entropy, OFI, Hurst).
Acts as the central point for Microstructure State generation.

This component 'wires' the raw sensors to the Nexus Engine.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List

# Sensors
from .shannon_entropy import entropy_analyzer
from .ofi import OrderFlowImbalance
from .hurst import HurstExponent

@dataclass
class MicrostructureState:
    entropy_score: float
    ofi_z_score: float
    hurst: float
    is_drunk: bool # High Entropy
    is_trending: bool # High Hurst
    buying_pressure: float # OFI

class MicrostructureScanner:
    def __init__(self):
        self.ofi_engine = OrderFlowImbalance(z_score_window=100)
        self.hurst_engine = HurstExponent()
        # entropy_analyzer is a singleton already
        
        self.current_state = MicrostructureState(
            entropy_score=0.5,
            ofi_z_score=0.0,
            hurst=0.5,
            is_drunk=False,
            is_trending=False,
            buying_pressure=0.0
        )
        
    def process_tick_batch(self, 
                           ticks: List[Dict], 
                           prices: np.ndarray) -> MicrostructureState:
        """
        Processes a batch of ticks/prices to update state.
        
        Args:
            ticks: List of {'bid':,'ask':,'bid_vol':,'ask_vol':} (For OFI)
            prices: Array of recent prices (For Entropy/Hurst)
        """
        
        # 1. Update Entropy (Market Noise)
        entropy = entropy_analyzer.calculate_returns_entropy(prices)
        regime, safe = entropy_analyzer.get_regime_signal(entropy)
        
        # 2. Update OFI (Order Flow)
        # Assuming ticks have bid/ask arrays ready or we construct them
        # For simplicity in this wiring phase, we use the proxy if no ticks, 
        # but the infrastructure is ready for ticks.
        
        current_ofi = 0.0
        if len(ticks) > 10:
             # Convert ticks to numpy for JIT
             bids = np.array([t['bid'] for t in ticks], dtype=np.float64)
             asks = np.array([t['ask'] for t in ticks], dtype=np.float64)
             bid_v = np.array([t['bid_vol'] for t in ticks], dtype=np.float64)
             ask_v = np.array([t['ask_vol'] for t in ticks], dtype=np.float64)
             
             current_ofi = self.ofi_engine.calculate_tick_ofi(bids, asks, bid_v, ask_v)
        else:
             # Fallback to Candle Proxy (using prices as candles for now)
             # Simple estimation: Close=prices[-1], Open=prices[0]
             mock_vol = np.ones_like(prices) # We don't have vol in this sig yet
             current_ofi = self.ofi_engine.calculate_proxy(prices[:-1], prices[1:], mock_vol[1:])
             
        ofi_z = self.ofi_engine.update_and_normalize(current_ofi)
        
        # 3. Update Hurst (Fractal Memory)
        hurst = self.hurst_engine.compute(prices)
        
        # 4. Compile State
        self.current_state = MicrostructureState(
            entropy_score=entropy,
            ofi_z_score=ofi_z,
            hurst=hurst,
            is_drunk=not safe,
            is_trending=hurst > 0.6,
            buying_pressure=ofi_z
        )
        
        return self.current_state

    def get_dict(self) -> dict:
        """Returns state as dictionary for StateEncoder."""
        return {
            "entropy_score": self.current_state.entropy_score,
            "ofi_z_score": self.current_state.ofi_z_score,
            "hurst": self.current_state.hurst
        }

# Singleton
micro_scanner = MicrostructureScanner()
