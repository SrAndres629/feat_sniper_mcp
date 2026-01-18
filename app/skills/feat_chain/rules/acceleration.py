import pandas as pd
from typing import Optional, Any, Dict
from ..base import FEATRule
from ..models import ValidationResult

class AccelerationRule(FEATRule):
    """Regla A: Valida Física de Mercado (Newtonian Acceleration)."""
    async def validate(self, market_data: Dict, physics_output: Optional[Any]) -> ValidationResult:
        try:
            from nexus_core.acceleration import acceleration_engine
            from app.skills.market_physics import market_physics
            
            if not hasattr(market_physics, 'price_window') or len(market_physics.price_window) < 10:
                 return ValidationResult(False, "Aceleración", "Warmup", {})

            df = pd.DataFrame({'close': list(market_physics.price_window)})
            vec = acceleration_engine.calculate_momentum_vector(df)
            
            is_valid = vec['is_valid']
            
            result = ValidationResult(
                is_valid=is_valid,
                rule_name="Aceleración",
                message=f"Newtonian: {vec['status']} (Acc:{vec['acceleration']:.2f})",
                data=vec
            )
            
            return await self.pass_next(market_data, physics_output, result)
        except Exception as e:
            return ValidationResult(False, "Aceleración", f"Error: {e}", {})
