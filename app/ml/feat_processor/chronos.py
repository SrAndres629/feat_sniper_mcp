import numpy as np
import pandas as pd
from typing import Dict, List
import datetime

from nexus_core.microstructure import HurstExponent, PriceImpactModel, OrderFlowImbalance
from nexus_core.chronos_engine.phaser import GoldCyclePhaser, MicroPhase, Intent

class ChronosTensorFactory:
    """
    [CHRONOS SUPREMACY: LAYER 3]
    The Neural Bridge.
    Aggregates:
    1. Physics (Hurst, OFI, Impact)
    2. Logic (Gold Micro-Phases)
    Into Tensors for the Synapse.
    """
    
    def __init__(self):
        self.phaser = GoldCyclePhaser()
        self.hurst = HurstExponent()
        self.impact = PriceImpactModel()
        self.ofi = OrderFlowImbalance()

    def process_state(self, timestamp_utc: datetime.datetime, 
                      closes: pd.Series, 
                      volumes: pd.Series, 
                      opens: pd.Series) -> Dict[str, np.ndarray]:
        """
        Generates the 'Liquidity Tensor' payload.
        """
        # 1. Get Session Logic
        state = self.phaser.get_current_state(timestamp_utc)
        
        # 2. Discrete Phase Tensor (One-Hot for Key Phases)
        # We focus on the big 3 types: ACCUM, MANIP, EXPAN
        # [Is_Accum, Is_Manip, Is_Expan, Is_Off]
        intent_tensor = [0.0, 0.0, 0.0, 0.0]
        if state.intent == Intent.ACCUMULATION: intent_tensor[0] = 1.0
        elif state.intent == Intent.MANIPULATION: intent_tensor[1] = 1.0
        elif state.intent == Intent.EXPANSION: intent_tensor[2] = 1.0
        else: intent_tensor[3] = 1.0 # Distribution/Off
        
        # 3. Physics Metrics
        h_val = self.hurst.calculate(closes)
        eta_val = self.impact.calculate_impact_coefficient(closes, volumes)
        ofi_val = self.ofi.calculate_proxy(opens, closes, volumes)
        
        # Normalize OFI (basic log scale or sign for now, depends on model scaling)
        # Using sign * log1p(abs) for safe neural input
        ofi_norm = np.sign(ofi_val) * np.log1p(np.abs(ofi_val))

        return {
            "liquidity_cycle_tensor": np.array(intent_tensor, dtype=np.float32),
            "hurst_regime": np.array([h_val], dtype=np.float32),
            "impact_potential": np.array([eta_val], dtype=np.float32),
            "ofi_flow": np.array([ofi_norm], dtype=np.float32),
            "raw_phase": state.micro_phase.value, # For logging, not tensor
            "action_warning": state.warning_label
        }
