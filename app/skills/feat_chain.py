from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging
import collections
from datetime import datetime

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
    """Memoria de corto plazo para detectar BOS/CHOCH en ticks."""
    def __init__(self):
        self.prices = collections.deque(maxlen=100) # Last 100 ticks
        self.fractal_high = -1.0
        self.fractal_low = 999999.0
    
    def update(self, price):
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

    def check_bos(self, current_price, trend):
        if trend == "BULLISH" and self.fractal_high > 0 and current_price > self.fractal_high:
             return True, f"BOS Alcista (> {self.fractal_high})"
        if trend == "BEARISH" and self.fractal_low < 999999 and current_price < self.fractal_low:
             return True, f"BOS Bajista (< {self.fractal_low})"
        return False, "Rango (Inside)"

# Estado Global de Estructura (Singleton per module load)
structure_memory = MicroStructure()

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
    """Regla F: Valida Estructura (BOS/CHOCH Real)."""
    async def validate(self, market_data: Dict, physics_output: Optional[Any]) -> ValidationResult:
        price = float(market_data.get('bid', 0) or market_data.get('close', 0))
        trend = physics_output.trend if physics_output else "NEUTRAL"
        
        # 1. Update Memory
        structure_memory.update(price)
        
        # 2. Check Break of Structure
        is_bos, reason = structure_memory.check_bos(price, trend)
        
        # Override para demo/simulacion si no hay suficiente historia aun
        if len(structure_memory.prices) < 5:
             is_bos = True 
             reason = "Warmup Structure"

        result = ValidationResult(
            is_valid=is_bos,
            rule_name="Forma",
            message=reason,
            data={"trend": trend, "last_high": structure_memory.fractal_high}
        )
        
        if result.is_valid:
            return await self.pass_next(market_data, physics_output, result)
        return result

class AccelerationRule(FEATRule):
    """Regla A: Valida Fisica de Mercado (Modulo 2)."""
    async def validate(self, market_data: Dict, physics_output: Optional[Any]) -> ValidationResult:
        if not physics_output:
             return ValidationResult(False, "Aceleración", "No Physics Data", {})

        # Strict User Requirement: Threshold 2.0
        accel_val = getattr(physics_output, 'acceleration_score', 0.0)
        is_valid = accel_val >= 2.0

        result = ValidationResult(
            is_valid=is_valid,
            rule_name="Aceleración",
            message=f"Aceleración Confirmada: {accel_val:.2f}x (>= 2.0)" if is_valid else f"Inercia Insuficiente: {accel_val:.2f}x (< 2.0)",
            data={"acceleration": accel_val, "z_score": getattr(physics_output, 'vol_z_score', 0.0)}
        )
        
        if result.is_valid:
             return await self.pass_next(market_data, physics_output, result)
        return result

class SpaceRule(FEATRule):
    """Regla E: Valida Espacio (Liquidez y FVG)."""
    async def validate(self, market_data: Dict, physics_output: Optional[Any]) -> ValidationResult:
        # FVG Proxy: Volumen Alto + Aceleración implica 'Displacement' (Void)
        # Check Liquidity
        is_displacement = False
        fv_gap = 0.0
        
        if physics_output:
             # Si hay alta aceleracion (>1.5) y alto volumen (>1.5 sigma), hay gap.
             accel = getattr(physics_output, 'acceleration_score', 0.0)
             vol_z = getattr(physics_output, 'vol_z_score', 0.0)
             
             if accel > 1.5 and vol_z > 1.0:
                 is_displacement = True
                 fv_gap = accel * 10 # Mock gap size in points
        
        result = ValidationResult(
            is_valid=is_displacement,
            rule_name="Espacio",
            message="FVG Detectado (Displacement)" if is_displacement else "Low Volatility/Congestion",
            data={"fvg_size": fv_gap, "is_displacement": is_displacement}
        )

        if result.is_valid:
             return await self.pass_next(market_data, physics_output, result)
        return result

class TimeRule(FEATRule):
    """Regla T: Valida Tiempo (Killzones)."""
    async def validate(self, market_data: Dict, physics_output: Optional[Any]) -> ValidationResult:
        from datetime import time as dt_time
        
        # Testability: Allow simulated time injection validation
        if 'simulated_time' in market_data:
            now = market_data['simulated_time']
        else:
            now = datetime.now()
            
        t = now.time()
        
        # Killzones: London (02-05), NY (07-11), Asia (20-00 - Opcional)
        is_london = dt_time(2,0) <= t <= dt_time(5,0)
        is_ny = dt_time(7,0) <= t <= dt_time(11,0)
        
        is_valid = is_london or is_ny

        result = ValidationResult(
            is_valid=is_valid,
            rule_name="Tiempo",
            message=f"Killzone Activa ({t.strftime('%H:%M')})" if is_valid else f"Fuera de Horario ({t.strftime('%H:%M')})",
            data={"session": "LONDON" if is_london else "NY" if is_ny else "NONE"}
        )

        if result.is_valid:
             return await self.pass_next(market_data, physics_output, result)
        return result

class FEATChain:
    """Coordinador de la cadena de decisión con protección Black Swan."""
    def __init__(self):
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
        
        logger.info("[BRAIN] FEAT Logic Chain Assembled (BlackSwan->F->E->A->T)")

    async def check_black_swan(self, market_data: Dict, physics_output: Optional[Any] = None) -> tuple:
        """
        Pre-validate market conditions for Black Swan events.
        
        Returns:
            Tuple of (can_trade: bool, lot_multiplier: float, rejection_reason: str)
        """
        guard = _get_black_swan_guard()
        if not guard:
            return (True, 1.0, None)  # Guard not available, proceed
        
        # Extract ATR from physics or market_data
        atr = 0.0
        if physics_output:
            # Try multiple ATR sources
            atr = getattr(physics_output, 'atr', 0) or getattr(physics_output, 'atr_14', 0)
        if atr == 0:
            atr = market_data.get('atr', 0) or market_data.get('atr_14', 0)
        
        # Extract spread if available
        spread = None
        if 'ask' in market_data and 'bid' in market_data:
            spread = float(market_data.get('ask', 0)) - float(market_data.get('bid', 0))
        
        # Extract equity (may come from account data)
        equity = market_data.get('equity', 0) or market_data.get('account_equity', 20.0)
        
        # Skip if no ATR data available
        if atr <= 0:
            logger.debug("[BLACK_SWAN] No ATR data, skipping guard check")
            return (True, 1.0, None)
        
        # Evaluate all guards
        try:
            decision = guard.evaluate(
                current_atr=atr,
                current_equity=equity,
                current_spread=spread
            )
            
            self._last_black_swan_decision = decision
            
            if not decision.can_trade:
                reasons = "; ".join(decision.rejection_reasons)
                logger.warning(f"🛡️ [BLACK_SWAN] Trading BLOCKED: {reasons}")
                return (False, 0.0, reasons)
            
            if decision.lot_multiplier < 1.0:
                logger.info(f"⚠️ [BLACK_SWAN] Lot reduced to {decision.lot_multiplier*100:.0f}%")
            
            return (True, decision.lot_multiplier, None)
            
        except Exception as e:
            logger.error(f"[BLACK_SWAN] Guard evaluation failed: {e}")
            return (True, 0.5, None)  # Proceed with caution

    async def analyze(self, market_data: Dict, current_price: float, precomputed_physics: Optional[Any] = None) -> bool:
        """Interfaz publica compatible con mcp_server.py"""
        
        # =====================================================
        # PHASE 0: BLACK SWAN GUARD (First line of defense)
        # =====================================================
        can_trade, lot_mult, rejection = await self.check_black_swan(market_data, precomputed_physics)
        
        if not can_trade:
            logger.warning(f"🚨 [BLACK_SWAN] TRADING HALTED: {rejection}")
            return False
        
        # Store lot multiplier for downstream use
        market_data['_black_swan_lot_multiplier'] = lot_mult
        
        # =====================================================
        # PHASE 1: PHYSICS ENGINE
        # =====================================================
        try:
            from app.skills.market_physics import market_physics
            
            if precomputed_physics:
                physics_output = precomputed_physics
            else:
                if 'simulated_time' in market_data:
                    ts = market_data['simulated_time'].timestamp()
                else:
                    ts = datetime.now().timestamp()
                    
                physics_output = market_physics.ingest_tick(market_data, force_timestamp=ts)
            
        except ImportError:
            physics_output = None
            logger.warning("Physics Engine Missing during analysis")

        # =====================================================
        # PHASE 2: FEAT CHAIN VALIDATION (F->E->A->T)
        # =====================================================
        result = await self.head.validate(market_data, physics_output)
        
        if result.is_valid:
            if lot_mult < 1.0:
                logger.info(f"✅ FEAT SETUP CONFIRMED (Lot: {lot_mult*100:.0f}%): {result.message}")
            else:
                logger.info(f"✅ FEAT SETUP CONFIRMED: {result.message}")
            return True
        else:
            logger.debug(f"❌ FEAT REJECT: [{result.rule_name}] {result.message}")
            return False
    
    def get_black_swan_status(self) -> Dict[str, Any]:
        """Get current Black Swan guard status for monitoring."""
        guard = _get_black_swan_guard()
        if guard:
            return guard.get_status()
        return {"status": "guard_not_available"}
    
    def get_last_decision(self) -> Optional[Any]:
        """Get the last Black Swan decision for inspection."""
        return self._last_black_swan_decision

feat_full_chain_institucional = FEATChain()

