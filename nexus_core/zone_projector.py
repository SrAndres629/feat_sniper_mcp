"""
Zone Projection Engine
======================
Proyecta zonas probables de rebote/ruptura basado en:
- Estructura actual (OB, FVG, Liquidity, BOS/CHOCH)
- Volatilidad (ATR, Killzones)
- Movimientos fractales (impulso/retroceso)

Genera un "Plan de Acción" para visualizar en MT5 y Dashboard.
"""
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("feat.zones")


# =============================================================================
# ZONE TYPES
# =============================================================================

class ZoneType(Enum):
    """Types of projected zones."""
    TARGET = "TARGET"           # Probable destination (expansion target)
    BOUNCE = "BOUNCE"           # Probable reversal zone
    BREAKOUT = "BREAKOUT"       # Zone that may break
    RETRACEMENT = "RETRACEMENT" # Pullback zone
    LIQUIDITY = "LIQUIDITY"     # Stop hunt zone


class VolatilityState(Enum):
    """Market volatility classification."""
    EXTREME = "EXTREME"     # >3x avg ATR (Black Swan territory)
    HIGH = "HIGH"           # 1.5-3x avg ATR (NY/London peak)
    NORMAL = "NORMAL"       # 0.8-1.5x avg ATR
    LOW = "LOW"             # <0.8x avg ATR (Asia/consolidation)


# Fibonacci levels for retracement projection
FIB_LEVELS = {
    "aggressive": [0.236, 0.382],        # High volatility retracements
    "normal": [0.382, 0.500, 0.618],     # Standard retracements
    "deep": [0.618, 0.786],              # Low volatility/accumulation
}

# Extension levels for target projection
EXTENSION_LEVELS = {
    "conservative": [1.0, 1.272],
    "standard": [1.272, 1.618],
    "aggressive": [1.618, 2.0, 2.618],
}


# =============================================================================
# RESULT STRUCTURES
# =============================================================================

@dataclass
class ProjectedZone:
    """A projected price zone with probability."""
    zone_type: ZoneType
    price_high: float
    price_low: float
    probability: float              # 0.0-1.0
    distance_pips: float
    volatility_factor: float        # Current ATR / Avg ATR
    is_high_vol_target: bool
    reasoning: str
    
    # Action plan details
    action_if_reached: str = ""     # "BUY", "SELL", "WAIT", "BREAKOUT_LONG", etc
    suggested_sl: float = 0.0
    suggested_tp: float = 0.0
    
    @property
    def midpoint(self) -> float:
        return (self.price_high + self.price_low) / 2
    
    @property
    def zone_size(self) -> float:
        return self.price_high - self.price_low
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.zone_type.value,
            "high": round(self.price_high, 5),
            "low": round(self.price_low, 5),
            "mid": round(self.midpoint, 5),
            "probability": round(self.probability, 2),
            "distance": round(self.distance_pips, 1),
            "volatility_factor": round(self.volatility_factor, 2),
            "high_vol": self.is_high_vol_target,
            "action": self.action_if_reached,
            "reasoning": self.reasoning
        }


@dataclass
class ActionPlan:
    """Complete action plan with projected zones."""
    current_price: float
    current_structure: str          # "BULLISH", "BEARISH", "RANGING"
    volatility_state: VolatilityState
    in_killzone: bool
    killzone_name: str
    
    # Primary zones
    immediate_target: Optional[ProjectedZone] = None     # Next probable destination
    bounce_zone: Optional[ProjectedZone] = None          # Where to look for reversal
    breakout_level: Optional[ProjectedZone] = None       # Key level to watch
    
    # All projected zones sorted by probability
    all_zones: List[ProjectedZone] = field(default_factory=list)
    
    # Bias and recommendation
    bias: str = "NEUTRAL"           # "LONG", "SHORT", "NEUTRAL"
    recommendation: str = "WAIT"    # "BUY_NOW", "SELL_NOW", "WAIT_ZONE", "WAIT_BREAKOUT"
    confidence: float = 0.0
    reasoning: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "price": self.current_price,
            "structure": self.current_structure,
            "volatility": self.volatility_state.value,
            "killzone": {"active": self.in_killzone, "name": self.killzone_name},
            "immediate_target": self.immediate_target.to_dict() if self.immediate_target else None,
            "bounce_zone": self.bounce_zone.to_dict() if self.bounce_zone else None,
            "breakout_level": self.breakout_level.to_dict() if self.breakout_level else None,
            "all_zones": [z.to_dict() for z in self.all_zones],
            "bias": self.bias,
            "recommendation": self.recommendation,
            "confidence": round(self.confidence, 2),
            "reasoning": self.reasoning
        }


# =============================================================================
# ZONE PROJECTOR ENGINE
# =============================================================================

class ZoneProjector:
    """
    Engine for projecting price zones based on structure and volatility.
    
    Key concepts:
    - Price moves in impulses and retracements (fractal)
    - High volatility = larger moves, shallower retracements
    - Low volatility = smaller moves, deeper retracements
    - Killzones increase probability of expansion
    """
    
    def __init__(self):
        logger.info("[ZoneProjector] Zone projection engine initialized")
    
    def get_volatility_state(self, candles: pd.DataFrame, window: int = 20) -> Tuple[VolatilityState, float]:
        """
        Calculate current volatility state.
        
        Returns:
            (VolatilityState, volatility_factor)
        """
        if len(candles) < window + 5:
            return VolatilityState.NORMAL, 1.0
        
        # Calculate ATR
        tr = np.maximum(
            candles["high"] - candles["low"],
            np.maximum(
                abs(candles["high"] - candles["close"].shift(1)),
                abs(candles["low"] - candles["close"].shift(1))
            )
        )
        atr = tr.rolling(window).mean()
        
        current_atr = atr.iloc[-1]
        avg_atr = atr.iloc[-window:].mean()
        
        if avg_atr == 0:
            return VolatilityState.NORMAL, 1.0
        
        factor = current_atr / avg_atr
        
        if factor > 3.0:
            state = VolatilityState.EXTREME
        elif factor > 1.5:
            state = VolatilityState.HIGH
        elif factor < 0.8:
            state = VolatilityState.LOW
        else:
            state = VolatilityState.NORMAL
        
        return state, factor
    
    def get_current_killzone(self, utc_offset: int = -4) -> Tuple[bool, str]:
        """Check if currently in a killzone."""
        try:
            from app.skills.liquidity_detector import get_current_kill_zone
            kz = get_current_kill_zone(utc_offset)
            return kz is not None, kz or "NONE"
        except:
            return False, "UNKNOWN"
    
    def identify_last_impulse(self, candles: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
        """
        Identify the last significant impulse move.
        
        An impulse is a strong directional move (>1.5x ATR).
        """
        if len(candles) < lookback:
            return {"found": False}
        
        recent = candles.tail(lookback)
        
        # Calculate ATR
        atr = (recent["high"] - recent["low"]).rolling(14).mean().iloc[-1]
        
        # Find largest single-bar move
        moves = abs(recent["close"] - recent["open"])
        max_move_idx = moves.idxmax()
        max_candle = recent.loc[max_move_idx]
        
        is_bullish = max_candle["close"] > max_candle["open"]
        move_size = abs(max_candle["close"] - max_candle["open"])
        
        if move_size < atr * 1.2:
            return {"found": False}
        
        return {
            "found": True,
            "direction": "BULLISH" if is_bullish else "BEARISH",
            "high": float(max_candle["high"]),
            "low": float(max_candle["low"]),
            "size": float(move_size),
            "atr_multiple": float(move_size / atr) if atr > 0 else 0
        }
    
    def calculate_retracement_zones(
        self,
        impulse_high: float,
        impulse_low: float,
        direction: str,
        volatility_state: VolatilityState,
        current_price: float
    ) -> List[ProjectedZone]:
        """
        Calculate Fibonacci retracement zones for the last impulse.
        """
        zones = []
        impulse_range = impulse_high - impulse_low
        
        # Select fib levels based on volatility
        if volatility_state == VolatilityState.HIGH:
            levels = FIB_LEVELS["aggressive"]
            prob_base = 0.75  # Shallow retracements more likely
        elif volatility_state == VolatilityState.LOW:
            levels = FIB_LEVELS["deep"]
            prob_base = 0.70  # Deep retracements more likely
        else:
            levels = FIB_LEVELS["normal"]
            prob_base = 0.65
        
        for i, fib in enumerate(levels):
            if direction == "BULLISH":
                # Retracement goes DOWN from high
                zone_mid = impulse_high - (impulse_range * fib)
                zone_high = zone_mid + (impulse_range * 0.02)
                zone_low = zone_mid - (impulse_range * 0.02)
                action = "BUY" if zone_low < current_price else "WAIT"
            else:
                # Retracement goes UP from low
                zone_mid = impulse_low + (impulse_range * fib)
                zone_high = zone_mid + (impulse_range * 0.02)
                zone_low = zone_mid - (impulse_range * 0.02)
                action = "SELL" if zone_high > current_price else "WAIT"
            
            # Probability decreases for deeper retracements
            prob = prob_base - (i * 0.1)
            
            zones.append(ProjectedZone(
                zone_type=ZoneType.RETRACEMENT,
                price_high=zone_high,
                price_low=zone_low,
                probability=max(0.3, prob),
                distance_pips=abs(current_price - zone_mid) * 10,
                volatility_factor=1.0,
                is_high_vol_target=volatility_state in [VolatilityState.HIGH, VolatilityState.EXTREME],
                reasoning=f"Fib {fib*100:.1f}% retracement",
                action_if_reached=action
            ))
        
        return zones
    
    def calculate_expansion_targets(
        self,
        impulse_high: float,
        impulse_low: float,
        direction: str,
        volatility_state: VolatilityState,
        current_price: float
    ) -> List[ProjectedZone]:
        """
        Calculate expansion targets beyond the impulse.
        """
        zones = []
        impulse_range = impulse_high - impulse_low
        
        # Select extension levels based on volatility
        if volatility_state in [VolatilityState.HIGH, VolatilityState.EXTREME]:
            levels = EXTENSION_LEVELS["aggressive"]
            prob_base = 0.70
        elif volatility_state == VolatilityState.LOW:
            levels = EXTENSION_LEVELS["conservative"]
            prob_base = 0.50
        else:
            levels = EXTENSION_LEVELS["standard"]
            prob_base = 0.60
        
        for i, ext in enumerate(levels):
            if direction == "BULLISH":
                # Extension goes UP
                target = impulse_low + (impulse_range * ext)
                zone_high = target + (impulse_range * 0.01)
                zone_low = target - (impulse_range * 0.01)
                action = "TAKE_PROFIT" if current_price < target else "WAIT"
            else:
                # Extension goes DOWN
                target = impulse_high - (impulse_range * ext)
                zone_high = target + (impulse_range * 0.01)
                zone_low = target - (impulse_range * 0.01)
                action = "TAKE_PROFIT" if current_price > target else "WAIT"
            
            # Probability decreases for further extensions
            prob = prob_base - (i * 0.15)
            
            zones.append(ProjectedZone(
                zone_type=ZoneType.TARGET,
                price_high=zone_high,
                price_low=zone_low,
                probability=max(0.2, prob),
                distance_pips=abs(current_price - target) * 10,
                volatility_factor=1.0,
                is_high_vol_target=volatility_state in [VolatilityState.HIGH, VolatilityState.EXTREME],
                reasoning=f"Fib {ext*100:.1f}% extension target",
                action_if_reached=action
            ))
        
        return zones
    
    def get_structure_zones(self, candles: pd.DataFrame, current_price: float) -> List[ProjectedZone]:
        """
        Get zones from existing structure (OB, FVG, Liquidity).
        """
        zones = []
        
        try:
            from app.skills.liquidity_detector import (
                detect_order_blocks, detect_fvg, detect_liquidity_pools
            )
            
            # Order Blocks
            obs = detect_order_blocks(candles, lookback=50)
            for ob in obs:
                distance = abs(current_price - ob.midpoint)
                is_above = ob.midpoint > current_price
                
                zones.append(ProjectedZone(
                    zone_type=ZoneType.BOUNCE,
                    price_high=ob.top,
                    price_low=ob.bottom,
                    probability=0.7 * ob.strength,
                    distance_pips=distance * 10,
                    volatility_factor=1.0,
                    is_high_vol_target=False,
                    reasoning=f"{ob.zone_type} OrderBlock",
                    action_if_reached="BUY" if "BULLISH" in ob.zone_type else "SELL"
                ))
            
            # Fair Value Gaps
            fvgs = detect_fvg(candles, lookback=30)
            for fvg in fvgs:
                mid = fvg["midpoint"]
                distance = abs(current_price - mid)
                
                zones.append(ProjectedZone(
                    zone_type=ZoneType.BOUNCE,
                    price_high=fvg["top"],
                    price_low=fvg["bottom"],
                    probability=0.65,
                    distance_pips=distance * 10,
                    volatility_factor=1.0,
                    is_high_vol_target=False,
                    reasoning=f"{fvg['type']} FVG",
                    action_if_reached="BUY" if fvg["type"] == "BULLISH" else "SELL"
                ))
            
            # Liquidity pools
            liq = detect_liquidity_pools(candles, lookback=50)
            
            if liq.get("liquidity_above", 0) > 0:
                zones.append(ProjectedZone(
                    zone_type=ZoneType.LIQUIDITY,
                    price_high=liq["liquidity_above"] * 1.001,
                    price_low=liq["liquidity_above"] * 0.999,
                    probability=0.60,
                    distance_pips=abs(current_price - liq["liquidity_above"]) * 10,
                    volatility_factor=1.0,
                    is_high_vol_target=True,  # Liquidity hunts happen in high vol
                    reasoning="Sell-side liquidity pool (highs)",
                    action_if_reached="BREAKOUT_LONG"
                ))
            
            if liq.get("liquidity_below", 0) > 0:
                zones.append(ProjectedZone(
                    zone_type=ZoneType.LIQUIDITY,
                    price_high=liq["liquidity_below"] * 1.001,
                    price_low=liq["liquidity_below"] * 0.999,
                    probability=0.60,
                    distance_pips=abs(current_price - liq["liquidity_below"]) * 10,
                    volatility_factor=1.0,
                    is_high_vol_target=True,
                    reasoning="Buy-side liquidity pool (lows)",
                    action_if_reached="BREAKOUT_SHORT"
                ))
                
        except Exception as e:
            logger.warning(f"[ZoneProjector] Structure zones error: {e}")
        
        return zones
    
    def generate_action_plan(
        self,
        candles: pd.DataFrame,
        current_price: float
    ) -> ActionPlan:
        """
        Generate complete action plan with all projected zones.
        
        This is the main entry point for the zone projector.
        """
        # Get volatility state
        vol_state, vol_factor = self.get_volatility_state(candles)
        
        # Get killzone status
        in_kz, kz_name = self.get_current_killzone()
        
        # Identify last impulse
        impulse = self.identify_last_impulse(candles)
        
        all_zones: List[ProjectedZone] = []
        structure = "RANGING"
        reasoning = []
        
        # Update volatility factor in zones
        def set_vol_factor(zones):
            for z in zones:
                z.volatility_factor = vol_factor
            return zones
        
        if impulse["found"]:
            structure = impulse["direction"]
            reasoning.append(f"Last impulse: {impulse['direction']} ({impulse['atr_multiple']:.1f}x ATR)")
            
            # Get retracement zones
            retracements = self.calculate_retracement_zones(
                impulse["high"], impulse["low"],
                impulse["direction"], vol_state, current_price
            )
            all_zones.extend(set_vol_factor(retracements))
            
            # Get expansion targets
            targets = self.calculate_expansion_targets(
                impulse["high"], impulse["low"],
                impulse["direction"], vol_state, current_price
            )
            all_zones.extend(set_vol_factor(targets))
        else:
            reasoning.append("No clear impulse detected - ranging")
        
        # Add structure zones (OB, FVG, Liquidity)
        structure_zones = self.get_structure_zones(candles, current_price)
        all_zones.extend(set_vol_factor(structure_zones))
        
        # Sort by probability
        all_zones.sort(key=lambda z: z.probability, reverse=True)
        
        # Identify primary zones
        immediate_target = next((z for z in all_zones if z.zone_type == ZoneType.TARGET), None)
        bounce_zone = next((z for z in all_zones if z.zone_type in [ZoneType.BOUNCE, ZoneType.RETRACEMENT]), None)
        breakout_level = next((z for z in all_zones if z.zone_type == ZoneType.LIQUIDITY), None)
        
        # Determine bias
        if structure == "BULLISH":
            bias = "LONG"
        elif structure == "BEARISH":
            bias = "SHORT"
        else:
            bias = "NEUTRAL"
        
        # Determine recommendation
        if vol_state == VolatilityState.EXTREME:
            recommendation = "WAIT"
            reasoning.append("⚠️ EXTREME volatility - wait for calm")
        elif in_kz and bias != "NEUTRAL" and bounce_zone:
            if bounce_zone.distance_pips < 30:  # Close to zone
                recommendation = f"WAIT_ZONE"
                reasoning.append(f"In {kz_name} KZ - wait for {bounce_zone.reasoning}")
            else:
                recommendation = "WAIT"
        else:
            recommendation = "WAIT"
        
        # Confidence based on alignment
        confidence = 0.5
        if in_kz:
            confidence += 0.15
            reasoning.append(f"✅ In {kz_name} Killzone (+15% conf)")
        if vol_state == VolatilityState.HIGH:
            confidence += 0.10
            reasoning.append("High volatility = expansion likely")
        if len([z for z in all_zones if z.probability > 0.6]) >= 3:
            confidence += 0.10
            reasoning.append("Multiple high-prob zones aligned")
        
        return ActionPlan(
            current_price=current_price,
            current_structure=structure,
            volatility_state=vol_state,
            in_killzone=in_kz,
            killzone_name=kz_name,
            immediate_target=immediate_target,
            bounce_zone=bounce_zone,
            breakout_level=breakout_level,
            all_zones=all_zones[:10],  # Top 10 zones
            bias=bias,
            recommendation=recommendation,
            confidence=min(1.0, confidence),
            reasoning=reasoning
        )


# Global singleton
zone_projector = ZoneProjector()
