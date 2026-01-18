from typing import Optional, Any, Dict
from ..base import FEATRule
from ..models import ValidationResult

class TimeRule(FEATRule):
    """Regla T: Valida Tiempo (Killzones NY)."""
    async def validate(self, market_data: Dict, physics_output: Optional[Any]) -> ValidationResult:
        try:
            from app.skills.calendar import chronos_engine
            t_val = chronos_engine.validate_window()
            
            result = ValidationResult(
                is_valid=t_val["is_valid"],
                rule_name="Tiempo",
                message=f"Session: {t_val['session']} ({t_val['ny_time']} NY)",
                data=t_val
            )
            
            return await self.pass_next(market_data, physics_output, result)
        except Exception as e:
            return ValidationResult(False, "Tiempo", f"Error: {e}", {})
