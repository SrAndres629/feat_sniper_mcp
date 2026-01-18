from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from .models import ValidationResult

class FEATRule(ABC):
    """Clase base para todas las reglas de la estrategia FEAT."""
    def __init__(self):
        self.next_rule: Optional['FEATRule'] = None

    def set_next(self, rule: 'FEATRule') -> 'FEATRule':
        self.next_rule = rule
        return rule

    @abstractmethod
    async def validate(self, market_data: Dict, physics_output: Optional[Any]) -> ValidationResult:
        """Abstract validation logic to be implemented by sub-classes."""
        pass

    async def pass_next(self, market_data: Dict, physics_output: Optional[Any], current_result: ValidationResult) -> ValidationResult:
        """Helper to pass to next rule or return success."""
        if not current_result.is_valid or self.next_rule is None:
            return current_result
        return await self.next_rule.validate(market_data, physics_output)
