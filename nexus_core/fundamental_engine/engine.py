"""
[MACRO SENTINEL - Fundamental Engine]
======================================
Central Orchestrator for Macro-Economic Awareness.
Integrates Calendar Data and Risk Modulation.
"""

import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from .calendar_client import CalendarClient, EconomicEvent, EventImpact
from .risk_modulator import RiskModulator, DEFCON

class FundamentalEngine:
    """
    [MACRO SENTINEL - Core]
    The "Financial Logic Auditor" of the system.
    
    Responsibilities:
    1. Monitor upcoming economic events.
    2. Calculate DEFCON levels.
    3. Generate Kill Signals when appropriate.
    4. Provide macro_regime_tensor for Neural Network.
    """
    
    def __init__(self, calendar_provider: str = "mock"):
        """
        Args:
            calendar_provider: Source for economic calendar data.
        """
        self.calendar = CalendarClient(provider=calendar_provider)
        self.risk_modulator = RiskModulator(tau_minutes=60.0)
        self.current_defcon = DEFCON.DEFCON_5
        self.last_check = None
    
    def check_event_proximity(self, currencies: List[str] = None) -> Dict:
        """
        [CORE METHOD] Checks for imminent events and updates system state.
        
        Args:
            currencies: Filter by relevant currencies.
            
        Returns:
            Dict containing:
                - defcon: Current DEFCON level.
                - kill_switch: Boolean (True = HALT trading).
                - position_multiplier: Float (0.0 to 1.0).
                - next_event: The most impactful imminent event (or None).
                - minutes_until: Time to next event.
        """
        now = datetime.datetime.now()
        
        # Get next HIGH impact event
        next_high = self.calendar.get_next_high_impact(currencies=currencies)
        
        if next_high is None:
            # No high impact events on the horizon
            self.current_defcon = DEFCON.DEFCON_5
            return {
                "defcon": DEFCON.DEFCON_5,
                "kill_switch": False,
                "position_multiplier": 1.0,
                "next_event": None,
                "minutes_until": float("inf"),
                "macro_regime": "SAFE"
            }
        
        # Calculate time until event
        time_delta = next_high.timestamp - now
        minutes_until = time_delta.total_seconds() / 60.0
        
        # Get DEFCON level
        defcon = self.risk_modulator.get_defcon_level(
            minutes_until=minutes_until,
            impact=next_high.impact
        )
        self.current_defcon = defcon
        
        # Get Kill Signal
        kill_switch = self.risk_modulator.generate_kill_signal(defcon)
        
        # Get Position Multiplier
        if kill_switch:
            position_multiplier = 0.0  # ZERO position
        else:
            position_multiplier = self.risk_modulator.get_position_multiplier(
                event=next_high,
                minutes_until=minutes_until
            )
        
        # Determine Macro Regime
        if defcon == DEFCON.DEFCON_1:
            macro_regime = "DANGER"
        elif defcon in [DEFCON.DEFCON_2, DEFCON.DEFCON_3]:
            macro_regime = "CAUTION"
        else:
            macro_regime = "SAFE"
        
        self.last_check = now
        
        return {
            "defcon": defcon,
            "kill_switch": kill_switch,
            "position_multiplier": position_multiplier,
            "next_event": next_high,
            "minutes_until": minutes_until,
            "macro_regime": macro_regime
        }
    
    def get_macro_regime_tensor(self, currencies: List[str] = None) -> Dict[str, float]:
        """
        [NEURAL INTEGRATION]
        Returns One-Hot encoded macro regime for Neural Network input.
        
        Uses Fuzzy Logic (validated by Stochastic Architect):
            - SAFE: No imminent risk.
            - CAUTION: Event approaching (1-4 hours).
            - DANGER: Event imminent (< 30 min).
        
        Returns:
            Dict with keys: "macro_safe", "macro_caution", "macro_danger"
        """
        result = self.check_event_proximity(currencies=currencies)
        regime = result["macro_regime"]
        
        # One-Hot Encoding
        tensor = {
            "macro_safe": 1.0 if regime == "SAFE" else 0.0,
            "macro_caution": 1.0 if regime == "CAUTION" else 0.0,
            "macro_danger": 1.0 if regime == "DANGER" else 0.0,
            "position_multiplier": result["position_multiplier"],
            "minutes_to_event": min(result["minutes_until"], 1440.0) / 1440.0  # Normalized (0-1, capped at 24h)
        }
        
        return tensor
    
    def apply_kill_switch(self, neural_signal: np.ndarray) -> np.ndarray:
        """
        [KILL SWITCH PROTOCOL]
        If DEFCON_1 is active, multiplies the neural signal by zero.
        
        Args:
            neural_signal: The output from the Neural Network (e.g., logits).
            
        Returns:
            Modified signal (zeroed if kill switch active).
        """
        if self.risk_modulator.generate_kill_signal(self.current_defcon):
            return np.zeros_like(neural_signal)
        return neural_signal
