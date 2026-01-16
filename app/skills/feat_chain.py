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


@dataclass
class FEATDecision:
    """
    Unified probabilistic decision from FEAT analysis.
    
    Each component returns a confidence score (0.0-1.0):
    - Form: Structure analysis (BOS/CHOCH/MAE/Layer alignment)
    - Space: Liquidity analysis (FVG/OB/Confluence)
    - Acceleration: Physics analysis (Velocity/Initiative)
    - Time: Session analysis (Killzones)
    """
    form_confidence: float = 0.0       # From StructureEngine
    space_confidence: float = 0.0      # From LiquidityDetector
    accel_confidence: float = 0.0      # From AccelerationEngine
    time_confidence: float = 0.0       # From ChronosEngine
    
    composite_score: float = 0.0       # Bayesian combination
    action: str = "HOLD"               # BUY/SELL/HOLD
    direction: int = 0                 # 1=Bullish, -1=Bearish, 0=Neutral
    reasoning: list = None
    
    # Risk modifiers
    black_swan_multiplier: float = 1.0
    layer_alignment: float = 0.0
    
    def __post_init__(self):
        if self.reasoning is None:
            object.__setattr__(self, 'reasoning', [])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "form_confidence": round(self.form_confidence, 3),
            "space_confidence": round(self.space_confidence, 3),
            "accel_confidence": round(self.accel_confidence, 3),
            "time_confidence": round(self.time_confidence, 3),
            "composite_score": round(self.composite_score, 3),
            "action": self.action,
            "direction": self.direction,
            "reasoning": self.reasoning,
            "black_swan_multiplier": self.black_swan_multiplier,
            "layer_alignment": round(self.layer_alignment, 3)
        }
    
    @property
    def is_valid_setup(self) -> bool:
        """True if composite score exceeds threshold for trading."""
        return self.composite_score >= 0.60 and self.action != "HOLD"


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
            
            feat_index = structure_engine.get_structural_score(df)
            health = structure_engine.get_structural_health(df)
            
            # Form Rule is valid if structural health is healthy/neutral AND score is high enough
            is_valid = health["health_score"] > 0.4 and feat_index > 50.0
            
            result = ValidationResult(
                is_valid=is_valid,
                rule_name="Forma",
                message=f"Estructura: {health['status']} (Score:{feat_index:.1f}, Health:{health['health_score']:.2f})",
                data={"feat_index": feat_index, "health": health}
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

    async def analyze_probabilistic(
        self, 
        market_data: Dict, 
        candles: pd.DataFrame = None,
        current_price: float = None
    ) -> FEATDecision:
        """
        Main probabilistic analysis entry point.
        
        Returns a FEATDecision with confidence scores from each component:
        - Form: Structure analysis (BOS/CHOCH/MAE/Layer alignment)
        - Space: Liquidity analysis (FVG/OB/Confluence)  
        - Acceleration: Physics analysis (Velocity/Initiative)
        - Time: Session analysis (Killzones)
        
        Args:
            market_data: Dict with bid, ask, symbol, etc.
            candles: Optional DataFrame with OHLCV data
            current_price: Current market price
            
        Returns:
            FEATDecision with all confidence scores and action
        """
        reasoning = []
        
        # Get current price
        if current_price is None:
            current_price = float(market_data.get('bid', 0) or market_data.get('close', 0))
        
        symbol = market_data.get('symbol', 'UNKNOWN')
        
        # 1. Black Swan check first
        can_trade, lot_mult, rejection = await self.check_black_swan(market_data)
        if not can_trade:
            return FEATDecision(
                action="HOLD",
                reasoning=[f"BLACK_SWAN: {rejection}"],
                black_swan_multiplier=lot_mult
            )
        
        # Initialize confidences
        form_conf = 0.0
        space_conf = 0.0
        accel_conf = 0.0
        time_conf = 0.0
        layer_alignment = 0.0
        direction = 0
        
        # 2. FORM (Structure) Analysis
        try:
            from nexus_core.structure_engine import structure_engine, four_layer_ema
            
            if candles is not None and len(candles) >= 20:
                # Get senior structural metrics
                health = structure_engine.get_structural_health(candles)
                zone_status = structure_engine.get_zone_status(candles)
                structural_bias = structure_engine.get_structural_bias(candles)
                
                # Form confidence based on health score
                form_conf = health["health_score"]
                reasoning.append(f"Structure Health: {health['status']} ({health['health_score']:.2f})")
                
                # Direction from structural bias
                if structural_bias == "BULLISH":
                    direction = 1
                elif structural_bias == "BEARISH":
                    direction = -1
                
                # Zone quality boost
                if zone_status["is_in_zone"]:
                    form_conf += zone_status["zone_strength"] * 0.2
                    reasoning.append(f"Inside {zone_status['nearest_zone']} zone (Strength: {zone_status['zone_strength']:.2f})")
                
                # BOS/CHOCH narrative integration
                narrative = structure_engine.get_structural_narrative(candles)
                if "CHOCH" in narrative.get("type", ""):
                    form_conf += 0.1
                    reasoning.append(f"CHOCH Narrative: {narrative['type']}")
                
                # Layer alignment (from four_layer_ema)
                layer_alignment = four_layer_ema.compute_layer_alignment(candles)
                if layer_alignment > 0.7:
                    form_conf += 0.1
                    reasoning.append(f"Layers Aligned: {layer_alignment:.2f}")
                    
        except Exception as e:
            logger.warning(f"[FORM] Analysis error: {e}")
            reasoning.append(f"FORM_ERROR: {e}")
        
        # 3. SPACE (Liquidity) Analysis
        try:
            from app.skills.liquidity_detector import compute_space_confidence
            
            if candles is not None and len(candles) >= 20:
                space_result = compute_space_confidence(candles, current_price)
                space_conf = space_result.overall_space_score
                reasoning.extend(space_result.reasoning)
                
        except Exception as e:
            logger.warning(f"[SPACE] Analysis error: {e}")
            reasoning.append(f"SPACE_ERROR: {e}")
        
        # 4. ACCELERATION Analysis
        try:
            from nexus_core.acceleration import acceleration_engine
            
            if candles is not None and len(candles) >= 20:
                accel_features = acceleration_engine.compute_acceleration_features(candles)
                
                if not accel_features.empty:
                    last = accel_features.iloc[-1]
                    
                    # Initiative candle confidence
                    if last.get('is_initiative', 0) > 0:
                        accel_conf += 0.4
                        reasoning.append("Initiative candle detected")
                    
                    # Acceleration score
                    accel_score = last.get('accel_score', 0)
                    accel_conf += accel_score * 0.3
                    
                    # Trap detection (reduces confidence)
                    if last.get('is_trap', 0) > 0:
                        accel_conf *= 0.5
                        reasoning.append("⚠️ TRAP detected - confidence reduced")
                    
                    # Breakout classification
                    accel_type = last.get('accel_type', 'normal')
                    if accel_type == 'breakout':
                        accel_conf += 0.3
                        reasoning.append("Breakout acceleration")
                    elif accel_type == 'climax':
                        accel_conf += 0.2
                        reasoning.append("Climax volume detected")
                        
        except Exception as e:
            logger.warning(f"[ACCEL] Analysis error: {e}")
            reasoning.append(f"ACCEL_ERROR: {e}")
        
        # 5. TIME Analysis
        try:
            from app.skills.liquidity_detector import get_current_kill_zone
            
            kz = get_current_kill_zone()
            if kz == "NY":
                time_conf = 0.9
                reasoning.append("✅ Inside NY Killzone")
            elif kz == "LONDON":
                time_conf = 0.8
                reasoning.append("✅ Inside London Killzone")
            elif kz == "ASIA":
                time_conf = 0.4
                reasoning.append("Asia session (lower weight)")
            else:
                time_conf = 0.2
                reasoning.append("Outside killzones")
                
        except Exception as e:
            logger.warning(f"[TIME] Analysis error: {e}")
            time_conf = 0.3
            reasoning.append(f"TIME_ERROR: {e}")
        
        # 6. Calculate Composite Score (Weighted Bayesian Combination)
        weights = {"F": 0.30, "S": 0.25, "A": 0.25, "T": 0.20}
        composite = (
            weights["F"] * min(1.0, form_conf) +
            weights["S"] * min(1.0, space_conf) +
            weights["A"] * min(1.0, accel_conf) +
            weights["T"] * min(1.0, time_conf)
        )
        
        # Determine action based on direction and composite
        action = "HOLD"
        if composite >= 0.60:
            if direction > 0:
                action = "BUY"
            elif direction < 0:
                action = "SELL"
            else:
                # No clear direction - check layer ordering
                try:
                    from nexus_core.structure_engine import four_layer_ema, EMALayer
                    if candles is not None:
                        metrics = four_layer_ema.get_all_layer_metrics(candles)
                        micro = metrics.get(EMALayer.MICRO)
                        oper = metrics.get(EMALayer.OPERATIVE)
                        if micro and oper:
                            if micro.avg_value > oper.avg_value:
                                action = "BUY"
                                direction = 1
                            else:
                                action = "SELL"
                                direction = -1
                except:
                    pass
        
        # Build final decision
        decision = FEATDecision(
            form_confidence=min(1.0, form_conf),
            space_confidence=min(1.0, space_conf),
            accel_confidence=min(1.0, accel_conf),
            time_confidence=min(1.0, time_conf),
            composite_score=composite,
            action=action,
            direction=direction,
            reasoning=reasoning,
            black_swan_multiplier=lot_mult,
            layer_alignment=layer_alignment
        )
        
        if decision.is_valid_setup:
            logger.info(f"✅ FEAT SETUP: {action} @ {current_price:.5f} (Score: {composite:.2f})")
        
        return decision

    async def analyze_mtf(
        self,
        candles_by_tf: Dict[str, pd.DataFrame],
        current_price: float
    ):
        """
        Multi-Timeframe Fractal Analysis - SNIPER MODE.
        
        Analyzes all 8 timeframes (W1→D1→H4→H1→M30→M15→M5→M1) and returns
        weighted composite score for precision entry.
        
        Args:
            candles_by_tf: Dict mapping timeframe string to DataFrame
                Example: {
                    "W1": df_weekly,
                    "D1": df_daily,
                    "H4": df_4hour,
                    "H1": df_1hour,
                    "M30": df_30min,
                    "M15": df_15min,
                    "M5": df_5min,
                    "M1": df_1min
                }
            current_price: Current market price
            
        Returns:
            MTFCompositeScore with weighted composite and entry decision
        """
        from nexus_core.mtf_engine import mtf_engine
        
        # Delegate to the fractal MTF engine
        return await mtf_engine.analyze_all_timeframes(candles_by_tf, current_price)


# Global singleton
feat_full_chain_institucional = FEATChain()
