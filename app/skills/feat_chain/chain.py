import logging
import pandas as pd
from typing import Dict, Any, Optional
from app.core.config import settings

from .models import MicroStructure, FEATDecision
from .rules.form import FormRule
from .rules.space import SpaceRule
from .rules.acceleration import AccelerationRule
from .rules.time import TimeRule

logger = logging.getLogger("feat.chain")

# Black Swan Protection
_black_swan_guard = None

def _get_black_swan_guard():
    global _black_swan_guard
    if _black_swan_guard is None:
        try:
            from app.services.black_swan_guard import black_swan_guard
            _black_swan_guard = black_swan_guard
        except Exception as e:
            logger.error(f"Failed to load BlackSwanGuard: {e}")
    return _black_swan_guard

class FEATChain:
    """Coordinador de la cadena de decisión con protección Black Swan."""
    def __init__(self):
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
        self._last_black_swan_decision = None
        logger.info("[BRAIN] FEAT Logic Chain Assembled (Institutional Mode)")

    def _get_structure(self, symbol: str) -> MicroStructure:
        if symbol not in self._structure_memories:
            self._structure_memories[symbol] = MicroStructure(symbol)
        return self._structure_memories[symbol]

    async def check_black_swan(self, market_data: Dict, physics_output: Optional[Any] = None) -> tuple:
        guard = _get_black_swan_guard()
        if not guard: return (True, 1.0, None)
        
        atr = getattr(physics_output, 'atr', 0) if physics_output else 0
        if atr <= 0: return (True, 1.0, None)
        
        spread = float(market_data.get('ask', 0)) - float(market_data.get('bid', 0)) if 'ask' in market_data else None
        equity = market_data.get('equity', 20.0)
        
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
        symbol = market_data.get('symbol', 'UNKNOWN')
        can_trade, lot_mult, rejection = await self.check_black_swan(market_data, precomputed_physics)
        if not can_trade: return False
        
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

    async def analyze_probabilistic(
        self, 
        market_data: Dict, 
        candles: pd.DataFrame = None,
        current_price: float = None
    ) -> FEATDecision:
        reasoning = []
        if current_price is None:
            current_price = float(market_data.get('bid', 0) or market_data.get('close', 0))
        
        can_trade, lot_mult, rejection = await self.check_black_swan(market_data)
        if not can_trade:
            return FEATDecision(action="HOLD", reasoning=[f"BLACK_SWAN: {rejection}"], black_swan_multiplier=lot_mult)
        
        form_conf, space_conf, accel_conf, time_conf, layer_alignment, direction = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        # 2. FORM (Structure) Analysis
        try:
            from nexus_core.structure_engine import structure_engine, four_layer_ema
            if candles is not None and len(candles) >= 20:
                health = structure_engine.get_structural_health(candles)
                zone_status = structure_engine.get_zone_status(candles)
                form_conf = health["health_score"]
                reasoning.append(f"Structure Health: {health['status']} ({health['health_score']:.2f})")
                
                # Direction fallback logic removed for space, but we keep core intent
                if health["trend_alignment"] > 0.6: direction = 1 # Simplified for POC
                
                if zone_status["is_in_zone"]:
                    form_conf += zone_status["zone_strength"] * 0.2
                
                layer_alignment = four_layer_ema.compute_layer_alignment(candles)
                if layer_alignment > 0.7: form_conf += 0.1
        except Exception as e:
            logger.warning(f"[FORM] Analysis error: {e}")

        # 3. SPACE (Liquidity) Analysis
        try:
            from app.skills.liquidity_detector import compute_space_confidence
            if candles is not None and len(candles) >= 20:
                space_result = compute_space_confidence(candles, current_price)
                space_conf = space_result.overall_space_score
                reasoning.extend(space_result.reasoning)
        except Exception as e:
            logger.warning(f"[SPACE] Analysis error: {e}")

        # 4. ACCELERATION Analysis
        try:
            from nexus_core.acceleration import acceleration_engine
            if candles is not None and len(candles) >= 20:
                accel_features = acceleration_engine.compute_acceleration_features(candles)
                if not accel_features.empty:
                    last = accel_features.iloc[-1]
                    if last.get('is_initiative', 0) > 0: accel_conf += 0.4
                    accel_conf += last.get('accel_score', 0) * 0.3
                    if last.get('is_trap', 0) > 0: accel_conf *= 0.5
        except Exception as e:
            logger.warning(f"[ACCEL] Analysis error: {e}")

        # 5. TIME Analysis
        try:
            from app.skills.liquidity_detector import get_current_kill_zone
            kz = get_current_kill_zone()
            if kz in ["NY", "LONDON"]: time_conf = 0.9 if kz == "NY" else 0.8
            else: time_conf = 0.3
        except Exception as e:
            time_conf = 0.3

        weights = {"F": 0.30, "S": 0.25, "A": 0.25, "T": 0.20}
        composite = (weights["F"] * min(1.0, form_conf) + weights["S"] * min(1.0, space_conf) +
                     weights["A"] * min(1.0, accel_conf) + weights["T"] * min(1.0, time_conf))
        
        action = "HOLD"
        if composite >= 0.60:
            action = "BUY" if direction >= 0 else "SELL" # Simplified

        return FEATDecision(
            form_confidence=min(1.0, form_conf), space_confidence=min(1.0, space_conf),
            accel_confidence=min(1.0, accel_conf), time_confidence=min(1.0, time_conf),
            composite_score=composite, action=action, direction=int(direction),
            reasoning=reasoning, black_swan_multiplier=lot_mult, layer_alignment=layer_alignment
        )
