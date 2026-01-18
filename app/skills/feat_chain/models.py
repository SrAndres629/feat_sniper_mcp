from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import collections

@dataclass
class ValidationResult:
    """Objeto inmutable para auditoria de decisiones."""
    is_valid: bool
    rule_name: str
    message: str
    data: Dict[str, Any]

@dataclass
class FEATDecision:
    """Unified probabilistic decision from FEAT analysis."""
    form_confidence: float = 0.0
    space_confidence: float = 0.0
    accel_confidence: float = 0.0
    time_confidence: float = 0.0
    composite_score: float = 0.0
    action: str = "HOLD"
    direction: int = 0
    reasoning: list = None
    black_swan_multiplier: float = 1.0
    layer_alignment: float = 0.0

    def __post_init__(self):
        if self.reasoning is None:
            self.reasoning = []

    def to_dict(self):
        return {
            "form": round(self.form_confidence, 3),
            "space": round(self.space_confidence, 3),
            "accel": round(self.accel_confidence, 3),
            "time": round(self.time_confidence, 3),
            "composite": round(self.composite_score, 3),
            "action": self.action,
            "direction": self.direction,
            "black_swan": self.black_swan_multiplier,
            "layer_alignment": self.layer_alignment,
            "reasoning": self.reasoning
        }

    @property
    def is_valid_setup(self) -> bool:
        return self.composite_score >= 0.60

class MicroStructure:
    """Memoria de corto plazo para detectar BOS/CHOCH en ticks."""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.highs = collections.deque(maxlen=10)
        self.lows = collections.deque(maxlen=10)
        self.tick_count = 0
        self.warmup_period = 20

    def update(self, price: float):
        self.tick_count += 1
        self.highs.append(price)
        self.lows.append(price)

    def is_warmed_up(self) -> bool:
        return self.tick_count >= self.warmup_period

    def check_bos(self, current_price: float, trend: str) -> bool:
        if not self.is_warmed_up(): return False
        if trend == "BULLISH":
            return current_price > max(self.highs)
        elif trend == "BEARISH":
            return current_price < min(self.lows)
        return False

    def get_status(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "warmed_up": self.is_warmed_up(),
            "ticks": self.tick_count
        }
