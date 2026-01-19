from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

class ZoneType(Enum):
    TARGET = "TARGET"
    BOUNCE = "BOUNCE"
    BREAKOUT = "BREAKOUT"
    RETRACEMENT = "RETRACEMENT"
    LIQUIDITY = "LIQUIDITY"

class VolatilityState(Enum):
    EXTREME = "EXTREME"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"

@dataclass
class ProjectedZone:
    zone_type: ZoneType
    price_high: float
    price_low: float
    probability: float
    distance_pips: float
    volatility_factor: float = 1.0
    is_high_vol_target: bool = False
    reasoning: str = ""
    action_if_reached: str = ""
    suggested_sl: float = 0.0
    suggested_tp: float = 0.0
    
    @property
    def midpoint(self) -> float:
        return (self.price_high + self.price_low) / 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.zone_type.value,
            "high": round(self.price_high, 5),
            "low": round(self.price_low, 5),
            "mid": round(self.midpoint, 5),
            "probability": round(self.probability, 2),
            "distance": round(self.distance_pips, 1),
            "action": self.action_if_reached,
            "reasoning": self.reasoning
        }

@dataclass
class ActionPlan:
    current_price: float
    current_structure: str
    volatility_state: VolatilityState
    in_killzone: bool
    killzone_name: str
    immediate_target: Optional[ProjectedZone] = None
    bounce_zone: Optional[ProjectedZone] = None
    breakout_level: Optional[ProjectedZone] = None
    all_zones: List[ProjectedZone] = field(default_factory=list)
    bias: str = "NEUTRAL"
    recommendation: str = "WAIT"
    recommendation: str = "WAIT"
    confidence_score: float = 0.0
    reasoning: List[str] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "price": self.current_price,
            "structure": self.current_structure,
            "volatility": self.volatility_state.value,
            "killzone": {"active": self.in_killzone, "name": self.killzone_name},
            "immediate_target": self.immediate_target.to_dict() if self.immediate_target else None,
            "bounce_zone": self.bounce_zone.to_dict() if self.bounce_zone else None,
            "breakout_level": self.breakout_level.to_dict() if self.breakout_level else None,
            "all_zones": [z.to_dict() for z in self.all_zones],
            "bias": self.bias,
            "recommendation": self.recommendation,
            "confidence": round(self.confidence_score, 2),
            "reasoning": self.reasoning
        }
