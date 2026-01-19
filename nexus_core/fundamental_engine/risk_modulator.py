"""
[MATH SENIOR FULLSTACK - subskill_financial]
Risk Modulator: Position Sizing Under Event Risk
=================================================
Implements the Doctoral Formula for Event-Adjusted Position Sizing.
"""

import math
from typing import Dict
from enum import Enum
from .calendar_client import EconomicEvent, EventImpact

class DEFCON(Enum):
    """
    Defense Condition Levels (Inspired by US Military).
    Lower number = Higher Alert.
    """
    DEFCON_5 = 5  # Normal Operations
    DEFCON_4 = 4  # Increased Readiness (Event in 4+ hours)
    DEFCON_3 = 3  # Round-the-Clock Readiness (Event in 1-4 hours)
    DEFCON_2 = 2  # Next Step to War (Event in 30-60 min)
    DEFCON_1 = 1  # Maximum Readiness (Event IMMINENT < 30 min)

class RiskModulator:
    """
    [MATH SENIOR FULLSTACK - subskill_financial]
    Calculates event-adjusted risk multipliers.
    
    Core Formula:
        Position_Size = Base_Risk * (1 / (1 + Event_Impact_Score))
    
    Where:
        Event_Impact_Score = Impact_Weight * Proximity_Decay
        Proximity_Decay = exp(-t / τ)  (Exponential decay towards event)
        τ (tau) = Time constant (Default: 60 minutes)
    """
    
    def __init__(self, tau_minutes: float = 60.0):
        """
        Args:
            tau_minutes: Time constant for exponential decay.
                         Lower = faster ramp-up of risk.
        """
        self.tau = tau_minutes
        self.impact_weights = {
            EventImpact.LOW: 0.2,
            EventImpact.MEDIUM: 0.5,
            EventImpact.HIGH: 1.0
        }
    
    def calculate_event_impact_score(self, event: EconomicEvent, minutes_until: float) -> float:
        """
        [subskill_financial] Calculates the weighted impact score.
        
        Args:
            event: The economic event.
            minutes_until: Time remaining until event.
            
        Returns:
            Impact score (0.0 to ~1.0+).
        """
        impact_weight = self.impact_weights.get(event.impact, 0.0)
        
        # [MATH SENIOR FULLSTACK - subskill_computational]
        # Exponential Decay: As we approach the event, the risk "decays" towards 1.0 (full impact).
        # Actually, we want risk to INCREASE as time decreases. So:
        # proximity_factor = 1 - exp(-t/τ)? No, that goes 0 -> 1 as t -> inf.
        # We want the opposite: high impact near t=0.
        # proximity_factor = exp(-t/τ) -> Goes 1 -> 0 as t -> inf.
        # High when t is low (close to event). This is what we want.
        
        if minutes_until <= 0:
            # Event has passed or is NOW
            proximity_factor = 1.0
        else:
            # ε = 1e-9 for numerical stability (never divide by zero if self.tau is somehow 0)
            proximity_factor = math.exp(-minutes_until / (self.tau + 1e-9))
        
        return impact_weight * proximity_factor
    
    def get_position_multiplier(self, event: EconomicEvent, minutes_until: float) -> float:
        """
        [DOCTORAL FORMULA]
        Position_Size_Multiplier = 1 / (1 + Event_Impact_Score)
        
        Result is in range (0, 1]:
            - 1.0 = No event risk (Full Position).
            - 0.5 = Moderate event risk (Half Position).
            - ~0.0 = Imminent HIGH event (Near-Zero Position).
        """
        score = self.calculate_event_impact_score(event, minutes_until)
        return 1.0 / (1.0 + score)
    
    def get_defcon_level(self, minutes_until: float, impact: EventImpact) -> DEFCON:
        """
        Maps time-to-event and impact to a DEFCON level.
        """
        if impact != EventImpact.HIGH:
            # Only HIGH impact events trigger elevated DEFCON
            if minutes_until < 60:
                return DEFCON.DEFCON_3
            return DEFCON.DEFCON_5
        
        # HIGH IMPACT EVENT LOGIC
        if minutes_until <= 0:
            return DEFCON.DEFCON_1  # EVENT IS NOW
        elif minutes_until < 30:
            return DEFCON.DEFCON_1  # IMMINENT
        elif minutes_until < 60:
            return DEFCON.DEFCON_2  # Near
        elif minutes_until < 240:  # 4 hours
            return DEFCON.DEFCON_3
        else:
            return DEFCON.DEFCON_4
    
    def generate_kill_signal(self, defcon: DEFCON) -> bool:
        """
        [KILL SWITCH PROTOCOL]
        Returns True if trading should be HALTED.
        """
        return defcon == DEFCON.DEFCON_1

    def apply_chronos_factor(self, base_size: float, ipda: 'IPDAGlobalState', kill_zone_active: bool) -> float:
        """
        [CHRONOS TIME SHIELD]
        Modulates position size based on Institutional Time.
        """
        # Base Multipliers per Phase
        phase_mult = 0.1
        if str(ipda) == "IPDAGlobalState.ACCUMULATION": phase_mult = 0.1
        elif str(ipda) == "IPDAGlobalState.MANIPULATION": phase_mult = 0.5
        elif str(ipda) == "IPDAGlobalState.EXPANSION": phase_mult = 1.0
        elif str(ipda) == "IPDAGlobalState.DISTRIBUTION": phase_mult = 0.3
        
        # Kill Zone Boost
        kz_mult = 1.2 if kill_zone_active else 0.8
        
        # Combined Factor
        final_mult = phase_mult * kz_mult
        
        # Apply to risk
        return base_size * final_mult
