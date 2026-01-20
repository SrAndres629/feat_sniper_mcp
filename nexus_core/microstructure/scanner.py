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
from .ticker import tick_buffer

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
        
        self.current_state = MicrostructureState(
            entropy_score=0.5,
            ofi_z_score=0.0,
            hurst=0.5,
            is_drunk=False,
            is_trending=False,
            buying_pressure=0.0
        )
        
    def live_scan(self) -> MicrostructureState:
        """
        [REAL-TIME] Performs analysis using the live TickBuffer.
        Zero-Lag Microstructure Awareness.
        """
        if not tick_buffer.ready:
            return self.current_state
            
        data = tick_buffer.get_arrays()
        return self._calculate_from_arrays(data["prices"], data["bids"], data["asks"], data["bid_vols"], data["ask_vols"])

    def process_tick_batch(self, 
                           ticks: List[Dict], 
                           prices: np.ndarray) -> MicrostructureState:
        """
        Processes a batch of ticks/prices (e.g. for backtesting or startup).
        """
        if len(ticks) > 10:
             bids = np.array([t['bid'] for t in ticks], dtype=np.float64)
             asks = np.array([t['ask'] for t in ticks], dtype=np.float64)
             bid_v = np.array([t['bid_vol'] for t in ticks], dtype=np.float64)
             ask_v = np.array([t['ask_vol'] for t in ticks], dtype=np.float64)
             return self._calculate_from_arrays(prices, bids, asks, bid_v, ask_v)
        else:
             # Fallback to Proxy
             mock_vol = np.ones_like(prices)
             ofi = self.ofi_engine.calculate_proxy(prices[:-1], prices[1:], mock_vol[1:])
             ofi_z = self.ofi_engine.update_and_normalize(ofi)
             entropy = entropy_analyzer.calculate_returns_entropy(prices)
             hurst = self.hurst_engine.compute(prices)
             return self._update_state(entropy, ofi_z, hurst)

    def _calculate_from_arrays(self, prices, bids, asks, bid_v, ask_v) -> MicrostructureState:
        """Centralized calculation logic."""
        # 1. Entropy
        entropy = entropy_analyzer.calculate_returns_entropy(prices)
        
        # 2. OFI
        ofi = self.ofi_engine.calculate_tick_ofi(bids, asks, bid_v, ask_v)
        ofi_z = self.ofi_engine.update_and_normalize(ofi)
        
        # 3. Hurst
        hurst = self.hurst_engine.compute(prices)
        
        return self._update_state(entropy, ofi_z, hurst)

    def _update_state(self, entropy, ofi_z, hurst) -> MicrostructureState:
        _, safe = entropy_analyzer.get_regime_signal(entropy)
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
