"""
FRACTAL ENGINE - Unified Signal Generation (Vibranium Grade)
==============================================================
The Synapse: Multi-Timeframe Pattern Recognition & Signal Fusion.

[PROJECT ATLAS] Phase 2: THE CEREBRO
Facade that unifies:
- MarketPhysics (Acceleration Layer)
- FEATChain (Structure, Space, Time Layers)
- Vision (Optional Screen Analysis)

Provides a single entry point for market analysis that outputs
a unified FractalState object for downstream consumption.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from datetime import datetime

# Import core engines
from app.skills.market_physics import market_physics, MarketRegime
from app.skills.feat_chain import feat_full_chain_institucional, ValidationResult
from app.core.config import settings

logger = logging.getLogger("feat.cerebro.fractal_engine")


# =============================================================================
# FRACTAL STATE - Unified Output
# =============================================================================

@dataclass
class FractalLayer:
    """Individual layer analysis result."""
    name: str
    is_valid: bool
    confidence: float
    data: Dict = field(default_factory=dict)
    message: str = ""


@dataclass
class FractalState:
    """
    Unified market state from all FEAT layers.
    
    This is the PRIMARY OUTPUT of the Cerebro system.
    All downstream components (Executor, HUD) consume this.
    """
    timestamp: str
    symbol: str
    
    # Layer Results
    form: FractalLayer          # L1: Structure (BOS/CHOCH)
    space: FractalLayer         # L2: Liquidity & POI
    acceleration: FractalLayer  # L3: Momentum Physics
    time: FractalLayer          # L4: Killzone/Session
    
    # Aggregated Signals
    overall_valid: bool = False
    overall_confidence: float = 0.0
    regime: str = "LAMINAR"
    trend: str = "NEUTRAL"
    
    # Raw Physics Data
    physics: Optional[MarketRegime] = None
    
    # Risk Flags
    black_swan_detected: bool = False
    black_swan_reason: str = ""
    
    def to_zmq_payload(self) -> Dict:
        """Convert to ZMQ-compatible payload for MT5 HUD."""
        return {
            "action": "HUD_UPDATE",
            "regime": self.regime,
            "trend": self.trend,
            "ai_confidence": self.overall_confidence * 100,
            "feat_score_val": (self.overall_confidence - 0.5) * 200,  # -100 to +100
            "feat_pvp_price": self.physics.atr if self.physics else 0,
            "session": self.time.data.get("session", "CLOSED"),
            "acceleration": "TRUE" if self.acceleration.is_valid else "FALSE",
            "structure_valid": "TRUE" if self.form.is_valid else "FALSE",
            "ts": datetime.utcnow().timestamp() * 1000
        }


# =============================================================================
# FRACTAL ENGINE - Unified Analysis Facade
# =============================================================================

class FractalEngine:
    """
    The Synapse: Unified entry point for all FEAT analysis.
    
    Orchestrates:
    1. Physics Engine (Acceleration, Velocity, Regime)
    2. Structure Analysis (BOS, CHOCH, Fractals)
    3. Space Analysis (Liquidity, POI, FVG)
    4. Time Analysis (Killzones, Sessions)
    5. Black Swan Protection
    
    Usage:
        engine = FractalEngine()
        state = engine.analyze(tick_data, symbol="BTCUSD")
        if state.overall_valid:
            execute_signal(state)
    """
    
    def __init__(self):
        self._physics = market_physics
        self._chain = feat_full_chain_institucional
        self._analysis_count = 0
        logger.info("[CEREBRO] FractalEngine initialized - Synapse ONLINE")
    
    def analyze(
        self, 
        tick_data: Dict, 
        symbol: str = None,
        current_price: float = None
    ) -> FractalState:
        """
        Perform complete FEAT analysis on incoming tick.
        
        Args:
            tick_data: Raw tick data (bid, ask, volume, etc.)
            symbol: Trading symbol (default from settings)
            current_price: Current price for structure analysis
            
        Returns:
            FractalState: Unified analysis result
        """
        symbol = symbol or settings.SYMBOL
        current_price = current_price or tick_data.get("bid") or tick_data.get("close")
        self._analysis_count += 1
        
        # 1. Physics Analysis (Acceleration Layer)
        physics_result = self._physics.ingest_tick(tick_data)
        
        # 2. FEAT Chain Analysis (Form, Space, Time)
        #    Prepare market_data for chain
        market_data = {
            "symbol": symbol,
            "price": current_price,
            "bid": tick_data.get("bid"),
            "ask": tick_data.get("ask"),
            "volume": tick_data.get("tick_volume", 0),
            **tick_data  # Include any extra fields
        }
        
        chain_result = self._chain.analyze(
            market_data=market_data,
            current_price=current_price,
            precomputed_physics=physics_result
        )
        
        # 3. Build Layer Results
        form_layer = self._extract_layer("Form", chain_result)
        space_layer = self._extract_layer("Space", chain_result)
        time_layer = self._extract_layer("Time", chain_result)
        
        accel_layer = FractalLayer(
            name="Acceleration",
            is_valid=physics_result.is_accelerating if physics_result else False,
            confidence=min(1.0, abs(physics_result.acceleration_score) / 2) if physics_result else 0,
            data={
                "score": physics_result.acceleration_score if physics_result else 0,
                "vol_z": physics_result.vol_z_score if physics_result else 0
            },
            message="Momentum detected" if physics_result and physics_result.is_accelerating else "Laminar flow"
        )
        
        # 4. Aggregate Confidence
        layer_weights = {
            "form": 0.35,
            "space": 0.25,
            "acceleration": 0.25,
            "time": 0.15
        }
        
        weighted_confidence = (
            form_layer.confidence * layer_weights["form"] +
            space_layer.confidence * layer_weights["space"] +
            accel_layer.confidence * layer_weights["acceleration"] +
            time_layer.confidence * layer_weights["time"]
        )
        
        overall_valid = all([
            form_layer.is_valid,
            accel_layer.is_valid,
            # Space and Time can be soft requirements
        ])
        
        # 5. Determine Regime
        regime = "LAMINAR"
        if physics_result and physics_result.is_accelerating:
            regime = "TURBULENT" if physics_result.acceleration_score > 1.5 else "TRENDING"
        
        # 6. Check Black Swan
        black_swan = self._chain.check_black_swan(market_data, physics_result)
        
        # 7. Build Final State
        state = FractalState(
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            form=form_layer,
            space=space_layer,
            acceleration=accel_layer,
            time=time_layer,
            overall_valid=overall_valid and not black_swan.get("halt", False),
            overall_confidence=weighted_confidence,
            regime=regime,
            trend=physics_result.trend if physics_result else "NEUTRAL",
            physics=physics_result,
            black_swan_detected=black_swan.get("halt", False),
            black_swan_reason=black_swan.get("reason", "")
        )
        
        return state
    
    def _extract_layer(self, layer_name: str, chain_result: ValidationResult) -> FractalLayer:
        """Extract layer result from chain validation."""
        # The chain returns the last validation result
        # In a real implementation, we'd track per-layer results
        is_valid = chain_result.is_valid if chain_result else False
        
        return FractalLayer(
            name=layer_name,
            is_valid=is_valid,
            confidence=0.8 if is_valid else 0.3,
            data={},
            message=chain_result.message if chain_result else ""
        )
    
    def get_status(self) -> Dict:
        """Return engine status for monitoring."""
        return {
            "analyses_performed": self._analysis_count,
            "physics_status": self._physics.get_status() if hasattr(self._physics, 'get_status') else {},
            "chain_status": "ACTIVE"
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

fractal_engine = FractalEngine()
