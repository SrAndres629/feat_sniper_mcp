from dataclasses import dataclass, field
from typing import Dict, Any, List
from enum import Enum

class EMALayer(Enum):
    """EMA Layer classification from CEMAs.mqh"""
    MICRO = "micro"           # Layer 1: Intent/Gas (fast EMAs)
    OPERATIVE = "operative"   # Layer 2: Structure/Water (medium EMAs)
    MACRO = "macro"           # Layer 3: Memory/Wall (slow EMAs)
    BIAS = "bias"             # Layer 4: Regime/Bedrock (SMMA 2048)

@dataclass
class LayerMetrics:
    """Metrics for a single EMA layer."""
    avg_value: float          # Average of all EMAs in layer
    spread: float             # Max - Min of layer
    compression: float        # Spread/Price * 1000 (squeeze indicator)
    slope: float              # Direction of layer movement
    layer_type: EMALayer

@dataclass
class StructuralConfidence:
    """Probabilistic result from structure analysis."""
    bos_confidence: float = 0.0       # 0.0-1.0: Break of Structure probability
    choch_confidence: float = 0.0     # 0.0-1.0: Change of Character probability
    zone_confidence: float = 0.0      # 0.0-1.0: Price at valid zone probability
    mae_confidence: float = 0.0       # 0.0-1.0: MAE pattern valid probability
    layer_alignment: float = 0.0      # 0.0-1.0: EMA layers aligned probability
    
    # [FEAT TENSORS]
    space_quality: float = 0.0        # E - Empty Space to Operative Layer
    time_sync_score: float = 0.0      # T - Fractal Time Alignment (M5/H4)
    kinetic_elasticity: float = 0.0   # A - Acceleration / Elasticity State
    
    overall_form_score: float = 0.0   # Weighted combination
    reasoning: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bos_confidence": round(self.bos_confidence, 3),
            "choch_confidence": round(self.choch_confidence, 3),
            "zone_confidence": round(self.zone_confidence, 3),
            "mae_confidence": round(self.mae_confidence, 3),
            "layer_alignment": round(self.layer_alignment, 3),
            "space_quality": round(self.space_quality, 3),
            "time_sync_score": round(self.time_sync_score, 3),
            "kinetic_elasticity": round(self.kinetic_elasticity, 3),
            "overall_form_score": round(self.overall_form_score, 3),
            "reasoning": self.reasoning
        }
