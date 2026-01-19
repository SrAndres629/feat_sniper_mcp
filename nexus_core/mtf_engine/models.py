from dataclasses import dataclass, field
from typing import Dict, Any, List
from enum import Enum
from app.core.config import settings

# =============================================================================
# TIMEFRAME DEFINITIONS
# =============================================================================

class Timeframe(Enum):
    """Timeframe hierarchy from macro to micro."""
    W1 = "W1"    # Weekly - Regime
    D1 = "D1"    # Daily - Macro Flow
    H4 = "H4"    # 4-Hour - Structural Bias
    H1 = "H1"    # 1-Hour - Intraday Flow
    M30 = "M30"  # 30-Min - Context
    M15 = "M15"  # 15-Min - Tactical
    M5 = "M5"    # 5-Min - Momentum
    M1 = "M1"    # 1-Min - SNIPER EXECUTION

def get_tf_weight(tf: Timeframe) -> float:
    """Retrieves timeframe weight from centralized settings."""
    return settings.MTF_WEIGHTS.get(tf.value, 0.0)

# =============================================================================
# RESULT STRUCTURES (INSTITUTIONAL GRADE)
# =============================================================================

@dataclass
class TimeframeScore:
    """Score for a single timeframe based on Hydrodynamics."""
    timeframe: str
    score: float                    # 0.0-1.0 Probability of Valid Move
    direction: int                  # 1=Long, -1=Short, 0=Neutral
    trend: str                      # "ACCUMULATION", "DISTRIBUTION", "VACUUM_RUN"
    
    # Microstructure Metrics
    impact_pressure: float = 0.0    # Effective Force
    is_vacuum: bool = False         # Low Liquidity State
    is_absorption: bool = False     # High Effort / Low Result
    ofi_z_score: float = 0.0        # Order Flow Imbalance Significance
    
    reasoning: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "score": round(self.score, 3),
            "direction": self.direction,
            "trend": self.trend,
            "impact": round(self.impact_pressure, 2),
            "vacuum": self.is_vacuum,
            "absorption": self.is_absorption,
            "ofi_z": round(self.ofi_z_score, 2)
        }

@dataclass
class MTFCompositeScore:
    """
    Multi-Timeframe Composite Score.
    Aggregates Flow Dynamics across time.
    """
    # Individual TF scores (0.0-1.0)
    w1_score: float = 0.0
    d1_score: float = 0.0
    h4_score: float = 0.0
    h1_score: float = 0.0
    m30_score: float = 0.0
    m15_score: float = 0.0
    m5_score: float = 0.0
    m1_score: float = 0.0
    
    # Weighted composite
    composite_score: float = 0.0
    
    # Alignment analysis
    alignment_percentage: float = 0.0
    primary_direction: int = 0      # 1=Long, -1=Short
    
    # Trading decision
    action: str = "HOLD"            # BUY/SELL/HOLD
    entry_type: str = "NONE"        # AGGRESSIVE/PASSIVE
    suggested_entry: float = 0.0
    suggested_sl: float = 0.0
    suggested_tp: float = 0.0
    
    # Metadata
    tf_details: Dict[str, TimeframeScore] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)
    
    # Thresholds (Now centralized in Settings)
    @property
    def THRESHOLD_SIGNAL(self) -> float: return settings.MTF_THRESHOLD_SIGNAL

    @property
    def THRESHOLD_SNIPER(self) -> float: return settings.MTF_THRESHOLD_SNIPER

    @property
    def THRESHOLD_SNIPER_TRIGGER(self) -> float: return settings.MTF_THRESHOLD_SNIPER_TRIGGER
    
    @property
    def is_valid_setup(self) -> bool:
        return self.composite_score >= self.THRESHOLD_SIGNAL and self.action != "HOLD"
    
    @property
    def is_sniper_entry(self) -> bool:
        return self.composite_score >= self.THRESHOLD_SNIPER
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "composite_score": round(self.composite_score, 3),
            "action": self.action,
            "entry_type": self.entry_type,
            "entry": self.suggested_entry,
            "sl": self.suggested_sl,
            "tp": self.suggested_tp,
            "reasoning": self.reasoning
        }
