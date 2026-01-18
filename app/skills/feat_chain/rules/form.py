import pandas as pd
from typing import Optional, Any, Dict
from ..base import FEATRule
from ..models import ValidationResult

class FormRule(FEATRule):
    """Regla F: Valida Estructura (BOS/CHOCH/MAE)."""
    async def validate(self, market_data: Dict, physics_output: Optional[Any]) -> ValidationResult:
        try:
            from nexus_core.structure_engine import structure_engine
            from app.skills.market_physics import market_physics
            
            if not hasattr(market_physics, 'price_window') or len(market_physics.price_window) < 10:
                return ValidationResult(False, "Forma", "Warmup (Prices < 10)", {})

            df = pd.DataFrame({'close': list(market_physics.price_window)})
            if hasattr(market_physics, 'volume_window'):
                df['volume'] = list(market_physics.volume_window)
            
            df['high'] = df['close']
            df['low'] = df['close']
            df['open'] = df['close']
            
            feat_index = structure_engine.get_structural_score(df)
            health = structure_engine.get_structural_health(df)
            
            is_valid = health["health_score"] > 0.4 and feat_index > 50.0
            
            result = ValidationResult(
                is_valid=is_valid,
                rule_name="Forma",
                message=f"Estructura: {health['status']} (Score:{feat_index:.1f}, Health:{health['health_score']:.2f})",
                data={"feat_index": feat_index, "health": health}
            )
            
            return await self.pass_next(market_data, physics_output, result)
        except Exception as e:
            return ValidationResult(False, "Forma", f"Error: {e}", {})
