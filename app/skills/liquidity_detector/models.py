from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class OrderBlock:
    """Institutional Order Block zone."""
    zone_type: str              # "BULLISH_OB" or "BEARISH_OB"
    top: float                  # Zone top price
    bottom: float               # Zone bottom price
    midpoint: float             # Zone midpoint
    time_index: int             # Bar index where OB formed
    strength: float             # 0.0-1.0 strength score
    mitigated: bool = False     # Has price returned to zone?
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.zone_type,
            "top": self.top,
            "bottom": self.bottom,
            "midpoint": self.midpoint,
            "strength": round(self.strength, 3),
            "mitigated": self.mitigated
        }

@dataclass
class SpaceConfidence:
    """Probabilistic result for Space/Liquidity analysis."""
    fvg_confidence: float = 0.0        # 0.0-1.0: Valid FVG present
    ob_confidence: float = 0.0         # 0.0-1.0: Price at OrderBlock
    liquidity_confidence: float = 0.0  # 0.0-1.0: Near liquidity pool
    confluence_confidence: float = 0.0 # 0.0-1.0: Multiple zones overlap
    
    overall_space_score: float = 0.0   # Weighted combination
    reasoning: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fvg_confidence": round(self.fvg_confidence, 3),
            "ob_confidence": round(self.ob_confidence, 3),
            "liquidity_confidence": round(self.liquidity_confidence, 3),
            "confluence_confidence": round(self.confluence_confidence, 3),
            "overall_space_score": round(self.overall_space_score, 3),
            "reasoning": self.reasoning
        }

@dataclass
class ConfluenceZone:
    """Zone where multiple signals overlap."""
    top: float
    bottom: float
    zone_types: List[str]       # What overlaps: ["FVG", "OB", "LIQUIDITY"]
    direction: str              # "BULLISH" or "BEARISH"
    strength: float             # 0.0-2.0 (confluence adds strength)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "top": self.top,
            "bottom": self.bottom,
            "overlapping": self.zone_types,
            "direction": self.direction,
            "strength": round(self.strength, 3)
        }
