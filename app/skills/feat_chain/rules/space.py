from typing import Optional, Any, Dict
from ..base import FEATRule
from ..models import ValidationResult

class SpaceRule(FEATRule):
    """Regla E: Valida Espacio (Liquidez y POI)."""
    async def validate(self, market_data: Dict, physics_output: Optional[Any]) -> ValidationResult:
        try:
            from app.skills.liquidity import liquidity_map
            price = float(market_data.get('bid', 0) or market_data.get('close', 0))
            
            poi = liquidity_map.get_nearest_poi(price)
            if not poi:
                return ValidationResult(False, "Espacio", "No POI Mapped", {})
                
            dist = abs(poi["price"] - price)
            atr = getattr(physics_output, 'atr', 0.001)
            is_near = dist < (atr * 2.0)
            
            result = ValidationResult(
                is_valid=is_near,
                rule_name="Espacio",
                message=f"POI: {poi['name']} at {dist:.4f} dist",
                data={"poi": poi, "dist": dist}
            )
            
            return await self.pass_next(market_data, physics_output, result)
        except Exception as e:
            return ValidationResult(False, "Espacio", f"Error: {e}", {})
