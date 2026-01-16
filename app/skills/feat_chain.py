from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging
import collections
from datetime import datetime, timezone, time as dt_time
import pandas as pd

# Logger
logger = logging.getLogger("feat.chain")

# Black Swan Protection (Imported dynamically to avoid circular deps)
_black_swan_guard = None

def _get_black_swan_guard():
    """Lazy import of BlackSwanGuard to avoid circular dependencies."""
    global _black_swan_guard
    if _black_swan_guard is None:
        try:
            from app.skills.black_swan_guard import get_black_swan_guard
            _black_swan_guard = get_black_swan_guard()
        except ImportError:
            logger.warning("[FEAT] BlackSwanGuard not available")
    return _black_swan_guard


class MicroStructure:
    """
    Memoria de corto plazo para detectar BOS/CHOCH en ticks.
    
    [P0 FIX] Now instantiated PER SYMBOL to avoid cross-contamination.
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.prices = collections.deque(maxlen=100)  # Last 100 ticks
        self.fractal_high = -1.0
        self.fractal_low = float('inf')
        self._warmup_complete = False
        logger.debug(f"[STRUCTURE] MicroStructure created for {symbol}")
    
    def update(self, price: float) -> None:
        self.prices.append(price)
        
        # Update Fractals (Simple High/Low of last N)
        if len(self.prices) >= 3:
            # Swing High (Simple): P[i-1] > P[i] and P[i-1] > P[i-2]
            prev = self.prices[-2]
            if prev > self.prices[-1] and prev > self.prices[-3]:
                self.fractal_high = prev
            
            # Swing Low
            if prev < self.prices[-1] and prev < self.prices[-3]:
                self.fractal_low = prev
        
        # Mark warmup complete after 10 ticks (not 5)
        if len(self.prices) >= 10:
            self._warmup_complete = True

    def is_warmed_up(self) -> bool:
        """[P0 FIX] Explicit warmup check - no more forced TRUE."""
        return self._warmup_complete

    def check_bos(self, current_price: float, trend: str) -> tuple:
        """
        Check for Break of Structure.
        """
        if not self._warmup_complete:
            return False, f"WARMING_UP ({len(self.prices)}/10 ticks)"
        
        if trend == "BULLISH" and self.fractal_high > 0 and current_price > self.fractal_high:
            return True, f"BOS Alcista (> {self.fractal_high:.2f})"
        if trend == "BEARISH" and self.fractal_low < float('inf') and current_price < self.fractal_low:
            return True, f"BOS Bajista (< {self.fractal_low:.2f})"
        return False, "Rango (Inside)"

    def get_status(self) -> Dict[str, Any]:
        """Get current structure status for monitoring."""
        return {
            "symbol": self.symbol,
            "tick_count": len(self.prices),
            "warmed_up": self._warmup_complete,
            "fractal_high": self.fractal_high if self.fractal_high > 0 else None,
            "fractal_low": self.fractal_low if self.fractal_low < float('inf') else None
        }


@dataclass(frozen=True)
class ValidationResult:
    """Objeto inmutable para auditoria de decisiones."""
    is_valid: bool
    rule_name: str
    message: str
    data: Dict[str, Any]


class FEATRule(ABC):
    """Clase base para todas las reglas de la estrategia FEAT."""
    def __init__(self):
        self.next_rule: Optional['FEATRule'] = None

    def set_next(self, rule: 'FEATRule') -> 'FEATRule':
        self.next_rule = rule
        return rule

    @abstractmethod
    async def validate(self, market_data: Dict, physics_output: Optional[Any]) -> ValidationResult:
        pass

    async def pass_next(self, market_data: Dict, physics_output: Optional[Any], current_result: ValidationResult) -> ValidationResult:
        """Helper to pass to next rule or return success."""
        if self.next_rule:
            return await self.next_rule.validate(market_data, physics_output)
        return current_result


class FormRule(FEATRule):
    """
    Regla F: Valida Estructura (BOS/CHOCH/MAE).
    """
    async def validate(self, market_data: Dict, physics_output: Optional[Any]) -> ValidationResult:
        try:
            from nexus_core.structure_engine import structure_engine
            from app.skills.market_physics import market_physics
            
            if not hasattr(market_physics, 'price_window') or len(market_physics.price_window) < 10:
                return ValidationResult(False, "Forma", "Warmup (Prices < 10)", {})

            df = pd.DataFrame({'close': list(market_physics.price_window)})
            if hasattr(market_physics, 'volume_window'):
                df['volume'] = list(market_physics.volume_window)
            
            # Simple OHLC reconstruction for structure engine
            df['high'] = df['close']
            df['low'] = df['close']
            df['open'] = df['close']
            
            res = structure_engine.compute_feat_index(df)
            last = res.iloc[-1]
            
            is_valid = last['feat_index'] > 60.0
            
            result = ValidationResult(
                is_valid=is_valid,
                rule_name="Forma",
                message=f"Estructura: {last['structure_status']} (FEAT:{last['feat_index']})",
                data={"feat_index": last['feat_index']}
            )
            
            if result.is_valid:
                return await self.pass_next(market_data, physics_output, result)
            return result
        except Exception as e:
            return ValidationResult(False, "Forma", f"Error: {e}", {})


class AccelerationRule(FEATRule):
    """
    Regla A: Valida Física de Mercado (Newtonian Acceleration).
    """
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
            
            if result.is_valid:
                return await self.pass_next(market_data, physics_output, result)
            return result
        except Exception as e:
            return ValidationResult(False, "Aceleración", f"Error: {e}", {})


class SpaceRule(FEATRule):
    """
    Regla E: Valida Espacio (Liquidez y POI).
    """
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
            
            if result.is_valid:
                return await self.pass_next(market_data, physics_output, result)
            return result
        except Exception as e:
            return ValidationResult(False, "Espacio", f"Error: {e}", {})


class TimeRule(FEATRule):
    """
    Regla T: Valida Tiempo (Killzones NY).
    """
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
            
            if result.is_valid:
                return await self.pass_next(market_data, physics_output, result)
            return result
        except Exception as e:
            return ValidationResult(False, "Tiempo", f"Error: {e}", {})


class FEATChain:
    """
    Coordinador de la cadena de decisión con protección Black Swan.
    """
    def __init__(self):
        # [P0 FIX] Per-symbol structure memories
        self._structure_memories: Dict[str, MicroStructure] = {}
        
        # Chain Assembly: F -> E -> A -> T
        self.form = FormRule()
        self.space = SpaceRule()
        self.accel = AccelerationRule()
        self.time = TimeRule()

        # Linking
        self.form.set_next(self.space)\
                 .set_next(self.accel)\
                 .set_next(self.time)
        
        self.head = self.form
        
        # Black Swan state cache
        self._last_black_swan_decision = None
        
        logger.info("[BRAIN] FEAT Logic Chain Assembled (Institutional Mode)")

    def _get_structure(self, symbol: str) -> MicroStructure:
        if symbol not in self._structure_memories:
            self._structure_memories[symbol] = MicroStructure(symbol)
        return self._structure_memories[symbol]

    async def check_black_swan(self, market_data: Dict, physics_output: Optional[Any] = None) -> tuple:
        guard = _get_black_swan_guard()
        if not guard:
            return (True, 1.0, None)
        
        atr = 0.0
        if physics_output:
            atr = getattr(physics_output, 'atr', 0)
        
        spread = None
        if 'ask' in market_data and 'bid' in market_data:
            spread = float(market_data.get('ask', 0)) - float(market_data.get('bid', 0))
        
        equity = market_data.get('equity', 20.0)
        
        if atr <= 0:
            return (True, 1.0, None)
        
        try:
            decision = guard.evaluate(current_atr=atr, current_equity=equity, current_spread=spread)
            self._last_black_swan_decision = decision
            if not decision.can_trade:
                return (False, 0.0, "; ".join(decision.rejection_reasons))
            return (True, decision.lot_multiplier, None)
        except Exception as e:
            logger.critical(f"🚨 [FAIL-SAFE] BlackSwanGuard Evaluation Failure: {e}")
            return (False, 0.0, f"INTERNAL_ERROR: {e}")

    async def analyze(self, market_data: Dict, current_price: float, precomputed_physics: Optional[Any] = None) -> bool:
        if 'symbol' not in market_data:
            market_data['symbol'] = 'UNKNOWN'
        
        can_trade, lot_mult, rejection = await self.check_black_swan(market_data, precomputed_physics)
        if not can_trade:
            return False
        
        market_data['_black_swan_lot_multiplier'] = lot_mult
        
        physics_output = precomputed_physics
        if not physics_output:
            try:
                from app.skills.market_physics import market_physics
                physics_output = market_physics.ingest_tick(market_data)
            except Exception as e:
                logger.error(f"🚨 [FAIL-SAFE] Physics Ingestion Failure: {e}")
                physics_output = None

        result = await self.head.validate(market_data, physics_output)
        
        if result.is_valid:
            logger.info(f"✅ FEAT SETUP CONFIRMED: {result.message}")
            return True
        return False

    def get_structure_status(self, symbol: str = None) -> Dict[str, Any]:
        if symbol and symbol in self._structure_memories:
            return self._structure_memories[symbol].get_status()
        return {s: m.get_status() for s, m in self._structure_memories.items()}

# Global singleton
feat_full_chain_institucional = FEATChain()
