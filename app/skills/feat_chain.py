"""
FEAT Chain - Institutional Trading Logic (P0 REPAIR)
=====================================================
Chain of Responsibility pattern for FEAT validation.

[P0 REPAIR] Fixes:
- Symbol Isolation: Each asset has its own MicroStructure memory
- Threshold Unification: Uses is_accelerating (σ dynamic) instead of 2.0 fixed
- UTC Enforcement: Killzones use UTC time, not local server time
- Warmup Safety: Returns False during warmup, not forced True
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging
import collections
from datetime import datetime, timezone, time as dt_time

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
        
        [P0 FIX] Returns (False, "WARMING_UP") during warmup instead of forced True.
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
    Regla F: Valida Estructura (BOS/CHOCH Real).
    
    [P0 FIX] Now receives MicroStructure from FEATChain, not global singleton.
    """
    def __init__(self, get_structure_fn):
        super().__init__()
        self._get_structure = get_structure_fn
    
    async def validate(self, market_data: Dict, physics_output: Optional[Any]) -> ValidationResult:
        symbol = market_data.get('symbol', 'UNKNOWN')
        price = float(market_data.get('bid', 0) or market_data.get('close', 0))
        trend = physics_output.trend if physics_output else "NEUTRAL"
        
        # [P0 FIX] Get symbol-specific structure memory
        structure = self._get_structure(symbol)
        
        # Update Memory
        structure.update(price)
        
        # Check Break of Structure
        is_bos, reason = structure.check_bos(price, trend)

        result = ValidationResult(
            is_valid=is_bos,
            rule_name="Forma",
            message=reason,
            data={"trend": trend, "last_high": structure.fractal_high, "symbol": symbol}
        )
        
        if result.is_valid:
            return await self.pass_next(market_data, physics_output, result)
        return result


class AccelerationRule(FEATRule):
    """
    Regla A: Valida Física de Mercado (Módulo 2).
    
    [P0 FIX] Uses is_accelerating (σ-based) instead of fixed 2.0 threshold.
    """
    async def validate(self, market_data: Dict, physics_output: Optional[Any]) -> ValidationResult:
        if not physics_output:
            return ValidationResult(False, "Aceleración", "No Physics Data", {})

        # [P0 FIX] Use the σ-dynamic is_accelerating, not fixed threshold
        is_valid = getattr(physics_output, 'is_accelerating', False)
        accel_val = getattr(physics_output, 'acceleration_score', 0.0)

        result = ValidationResult(
            is_valid=is_valid,
            rule_name="Aceleración",
            message=f"Aceleración Confirmada: {accel_val:.2f} (σ-dynamic)" if is_valid else f"Inercia: {accel_val:.2f} (below σ threshold)",
            data={
                "acceleration": accel_val, 
                "z_score": getattr(physics_output, 'vol_z_score', 0.0),
                "is_accelerating": is_valid
            }
        )
        
        if result.is_valid:
            return await self.pass_next(market_data, physics_output, result)
        return result


class SpaceRule(FEATRule):
    """
    Regla E: Valida Espacio (Liquidez y FVG).
    
    [LAST MILE] Real FVG Calculation:
    - FVG = distance between fractals from MicroStructure
    - If no clear gap, use distance to nearest fractal
    """
    def __init__(self, get_structure_fn):
        super().__init__()
        self._get_structure = get_structure_fn
    
    async def validate(self, market_data: Dict, physics_output: Optional[Any]) -> ValidationResult:
        symbol = market_data.get('symbol', 'UNKNOWN')
        current_price = float(market_data.get('bid', 0) or market_data.get('close', 0))
        
        # Get symbol-specific structure for fractal data
        structure = self._get_structure(symbol)
        
        is_displacement = False
        fv_gap = 0.0
        gap_type = "NONE"
        
        if physics_output:
            accel = getattr(physics_output, 'acceleration_score', 0.0)
            vol_z = getattr(physics_output, 'vol_z_score', 0.0)
            
            # Displacement detection: high acceleration + high volume
            if accel > 1.5 and vol_z > 1.0:
                is_displacement = True
                
                # [LAST MILE] Real FVG Calculation
                # Use fractals from MicroStructure instead of placeholder
                fractal_high = structure.fractal_high if structure.fractal_high > 0 else current_price
                fractal_low = structure.fractal_low if structure.fractal_low < float('inf') else current_price
                
                if fractal_high > 0 and fractal_low < float('inf'):
                    # FVG = range between recent fractals
                    fv_gap = abs(fractal_high - fractal_low)
                    
                    # Determine gap type based on current price position
                    if current_price > fractal_high:
                        gap_type = "BULLISH_BREAKOUT"
                    elif current_price < fractal_low:
                        gap_type = "BEARISH_BREAKOUT"
                    else:
                        gap_type = "INSIDE_RANGE"
                else:
                    # Fallback: distance from current price to ATR (if available)
                    atr = getattr(physics_output, 'atr', 0)
                    fv_gap = atr if atr > 0 else 0
                    gap_type = "ATR_FALLBACK"
        
        result = ValidationResult(
            is_valid=is_displacement,
            rule_name="Espacio",
            message=f"FVG Detectado: {fv_gap:.2f} ({gap_type})" if is_displacement else "Low Volatility/Congestion",
            data={
                "fvg_size": round(fv_gap, 4), 
                "is_displacement": is_displacement,
                "gap_type": gap_type
            }
        )

        if result.is_valid:
            return await self.pass_next(market_data, physics_output, result)
        return result


class TimeRule(FEATRule):
    """
    Regla T: Valida Tiempo (Killzones).
    
    [P0 FIX] Uses UTC time, not local server time.
    """
    
    # Killzones in UTC
    KILLZONES = {
        "LONDON_OPEN": (dt_time(7, 0), dt_time(10, 0)),   # 07:00-10:00 UTC
        "NY_OPEN": (dt_time(12, 0), dt_time(15, 0)),      # 12:00-15:00 UTC
        "LONDON_CLOSE": (dt_time(15, 0), dt_time(17, 0)), # 15:00-17:00 UTC (overlap)
        "ASIA": (dt_time(0, 0), dt_time(3, 0)),           # 00:00-03:00 UTC (optional)
    }
    
    async def validate(self, market_data: Dict, physics_output: Optional[Any]) -> ValidationResult:
        # [P0 FIX] Use UTC time, not local time
        if 'simulated_time' in market_data:
            now = market_data['simulated_time']
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
        else:
            now = datetime.now(timezone.utc)
            
        t = now.time()
        
        # Check which killzone we're in
        active_kz = None
        for kz_name, (start, end) in self.KILLZONES.items():
            if start <= t <= end:
                active_kz = kz_name
                break
        
        is_valid = active_kz is not None

        result = ValidationResult(
            is_valid=is_valid,
            rule_name="Tiempo",
            message=f"Killzone Activa: {active_kz} ({t.strftime('%H:%M')} UTC)" if is_valid else f"Fuera de Horario ({t.strftime('%H:%M')} UTC)",
            data={"session": active_kz or "NONE", "time_utc": t.isoformat()}
        )

        if result.is_valid:
            return await self.pass_next(market_data, physics_output, result)
        return result


class FEATChain:
    """
    Coordinador de la cadena de decisión con protección Black Swan.
    
    [P0 FIX] Symbol-isolated MicroStructure memories.
    """
    def __init__(self):
        # [P0 FIX] Per-symbol structure memories
        self._structure_memories: Dict[str, MicroStructure] = {}
        
        # Chain Assembly: F -> E -> A -> T
        # Note: Order could be optimized to T -> A -> E -> F for efficiency
        # but keeping F first for compatibility with existing logic flow
        self.form = FormRule(self._get_structure)
        self.space = SpaceRule(self._get_structure)  # [LAST MILE] Now uses real fractals
        self.accel = AccelerationRule()
        self.time = TimeRule()

        # Linking
        self.form.set_next(self.space)\
                 .set_next(self.accel)\
                 .set_next(self.time)
        
        self.head = self.form
        
        # Black Swan state cache
        self._last_black_swan_decision = None
        
        logger.info("[BRAIN] FEAT Logic Chain Assembled (Symbol-Isolated, UTC-Enforced)")

    def _get_structure(self, symbol: str) -> MicroStructure:
        """
        [P0 FIX] Get or create symbol-specific structure memory.
        
        Each asset now has its own isolated geometric brain.
        """
        if symbol not in self._structure_memories:
            self._structure_memories[symbol] = MicroStructure(symbol)
            logger.info(f"[FEAT] Created isolated MicroStructure for {symbol}")
        return self._structure_memories[symbol]

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
            atr = getattr(physics_output, 'atr', 0) or getattr(physics_output, 'atr_14', 0)
        if atr == 0:
            atr = market_data.get('atr', 0) or market_data.get('atr_14', 0)
        
        # Extract spread if available
        spread = None
        if 'ask' in market_data and 'bid' in market_data:
            spread = float(market_data.get('ask', 0)) - float(market_data.get('bid', 0))
        
        # Extract equity
        equity = market_data.get('equity', 0) or market_data.get('account_equity', 20.0)
        
        # Skip if no ATR data available
        if atr <= 0:
            logger.debug("[BLACK_SWAN] No ATR data, skipping guard check")
            return (True, 1.0, None)
        
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
            return (True, 0.5, None)

    async def analyze(self, market_data: Dict, current_price: float, precomputed_physics: Optional[Any] = None) -> bool:
        """Interfaz pública compatible con mcp_server.py"""
        
        # Ensure symbol is in market_data
        if 'symbol' not in market_data:
            market_data['symbol'] = 'UNKNOWN'
        
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
                    ts = datetime.now(timezone.utc).timestamp()
                    
                physics_output = market_physics.ingest_tick(market_data, force_timestamp=ts)
            
        except ImportError:
            physics_output = None
            logger.warning("Physics Engine Missing during analysis")

        # =====================================================
        # PHASE 2: FEAT CHAIN VALIDATION (F->E->A->T)
        # =====================================================
        result = await self.head.validate(market_data, physics_output)
        
        if result.is_valid:
            symbol = market_data.get('symbol', 'UNKNOWN')
            if lot_mult < 1.0:
                logger.info(f"✅ [{symbol}] FEAT SETUP CONFIRMED (Lot: {lot_mult*100:.0f}%): {result.message}")
            else:
                logger.info(f"✅ [{symbol}] FEAT SETUP CONFIRMED: {result.message}")
            return True
        else:
            logger.debug(f"❌ FEAT REJECT: [{result.rule_name}] {result.message}")
            return False
    
    def get_structure_status(self, symbol: str = None) -> Dict[str, Any]:
        """Get structure memory status for monitoring."""
        if symbol:
            if symbol in self._structure_memories:
                return self._structure_memories[symbol].get_status()
            return {"error": f"No structure for {symbol}"}
        return {s: m.get_status() for s, m in self._structure_memories.items()}
    
    def get_black_swan_status(self) -> Dict[str, Any]:
        """Get current Black Swan guard status for monitoring."""
        guard = _get_black_swan_guard()
        if guard:
            return guard.get_status()
        return {"status": "guard_not_available"}
    
    def get_last_decision(self) -> Optional[Any]:
        """Get the last Black Swan decision for inspection."""
        return self._last_black_swan_decision


# Global singleton
feat_full_chain_institucional = FEATChain()
