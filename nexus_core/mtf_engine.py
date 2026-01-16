"""
Multi-Timeframe Fractal Engine
==============================
Análisis jerárquico fractal de 8 timeframes con scoring ponderado.

Jerarquía Fractal:
W1 → D1 → H4 → H1 → M30 → M15 → M5 → M1 (SNIPER)

Cada timeframe hereda contexto del superior y contribuye al score compuesto.
"""
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger("feat.mtf")


# =============================================================================
# TIMEFRAME DEFINITIONS
# =============================================================================

class Timeframe(Enum):
    """Timeframe hierarchy from macro to micro."""
    W1 = "W1"    # Weekly - Régimen
    D1 = "D1"    # Daily - Macro
    H4 = "H4"    # 4-Hour - Bias
    H1 = "H1"    # 1-Hour - Flow
    M30 = "M30"  # 30-Min - Context
    M15 = "M15"  # 15-Min - Structure
    M5 = "M5"    # 5-Min - Timing
    M1 = "M1"    # 1-Min - SNIPER TRIGGER


# Candle counts per timeframe
CANDLE_COUNTS = {
    Timeframe.W1: 52,    # 1 year
    Timeframe.D1: 100,   # ~3 months
    Timeframe.H4: 84,    # 2 weeks
    Timeframe.H1: 168,   # 1 week
    Timeframe.M30: 96,   # 2 days
    Timeframe.M15: 192,  # 2 days
    Timeframe.M5: 288,   # 1 day
    Timeframe.M1: 300,   # 5 hours
}

# Weight per timeframe (must sum to 1.0)
WEIGHTS = {
    Timeframe.W1: 0.05,   # 5% - Régimen (veto power)
    Timeframe.D1: 0.10,   # 10% - Macro estructura
    Timeframe.H4: 0.20,   # 20% - BIAS PRINCIPAL
    Timeframe.H1: 0.20,   # 20% - FLUJO DIRECCIONAL
    Timeframe.M30: 0.10,  # 10% - Contexto
    Timeframe.M15: 0.15,  # 15% - ESTRUCTURA OPERATIVA
    Timeframe.M5: 0.10,   # 10% - Timing
    Timeframe.M1: 0.10,   # 10% - TRIGGER
}


# =============================================================================
# RESULT STRUCTURES
# =============================================================================

@dataclass
class TimeframeScore:
    """Score for a single timeframe."""
    timeframe: str
    score: float                    # 0.0-1.0
    direction: int                  # 1=Bullish, -1=Bearish, 0=Neutral
    trend: str                      # "BULLISH", "BEARISH", "NEUTRAL"
    has_bos: bool = False           # Break of Structure detected
    has_choch: bool = False         # Change of Character detected
    has_fvg: bool = False           # Fair Value Gap present
    has_ob: bool = False            # OrderBlock present
    layer_alignment: float = 0.0    # 4-Layer EMA alignment
    reasoning: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "score": round(self.score, 3),
            "direction": self.direction,
            "trend": self.trend,
            "bos": self.has_bos,
            "choch": self.has_choch,
            "fvg": self.has_fvg,
            "ob": self.has_ob,
            "layer_alignment": round(self.layer_alignment, 3)
        }


@dataclass
class MTFCompositeScore:
    """
    Multi-Timeframe Composite Score for Sniper Entry.
    
    All 8 timeframes are analyzed and weighted to produce
    a single composite score that determines entry.
    """
    # Individual TF scores (0.0-1.0)
    w1_score: float = 0.0
    d1_score: float = 0.0
    h4_score: float = 0.0
    h1_score: float = 0.0
    m30_score: float = 0.0
    m15_score: float = 0.0
    m5_score: float = 0.0
    m1_score: float = 0.0
    
    # Weighted composite
    composite_score: float = 0.0
    
    # Alignment analysis
    all_bullish: bool = False       # All TFs agree bullish
    all_bearish: bool = False       # All TFs agree bearish
    alignment_percentage: float = 0.0  # % of TFs aligned
    primary_direction: int = 0      # 1=Bullish, -1=Bearish, 0=Mixed
    
    # Trading decision
    action: str = "HOLD"            # BUY/SELL/HOLD
    entry_type: str = "NONE"        # MARKET/LIMIT/STOP/NONE
    suggested_entry: float = 0.0
    suggested_sl: float = 0.0
    suggested_tp: float = 0.0
    
    # Metadata
    tf_details: Dict[str, TimeframeScore] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)
    
    # Thresholds
    THRESHOLD_SIGNAL: float = 0.65
    THRESHOLD_STRONG: float = 0.75
    THRESHOLD_SNIPER: float = 0.80
    
    @property
    def is_valid_setup(self) -> bool:
        """True if composite score exceeds minimum threshold."""
        return self.composite_score >= self.THRESHOLD_SIGNAL and self.action != "HOLD"
    
    @property
    def is_strong_setup(self) -> bool:
        """True if this is a high-confidence setup."""
        return self.composite_score >= self.THRESHOLD_STRONG
    
    @property
    def is_sniper_entry(self) -> bool:
        """True if this is maximum confidence sniper entry."""
        return self.composite_score >= self.THRESHOLD_SNIPER and self.entry_type == "MARKET"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scores": {
                "W1": round(self.w1_score, 3),
                "D1": round(self.d1_score, 3),
                "H4": round(self.h4_score, 3),
                "H1": round(self.h1_score, 3),
                "M30": round(self.m30_score, 3),
                "M15": round(self.m15_score, 3),
                "M5": round(self.m5_score, 3),
                "M1": round(self.m1_score, 3),
            },
            "composite_score": round(self.composite_score, 3),
            "alignment": {
                "all_bullish": self.all_bullish,
                "all_bearish": self.all_bearish,
                "percentage": round(self.alignment_percentage, 1),
                "direction": self.primary_direction
            },
            "decision": {
                "action": self.action,
                "entry_type": self.entry_type,
                "entry": self.suggested_entry,
                "sl": self.suggested_sl,
                "tp": self.suggested_tp
            },
            "is_valid": self.is_valid_setup,
            "is_strong": self.is_strong_setup,
            "is_sniper": self.is_sniper_entry,
            "reasoning": self.reasoning
        }


# =============================================================================
# FRACTAL MTF ENGINE
# =============================================================================

class FractalMTFEngine:
    """
    Motor de análisis multitemporal fractal.
    
    Cada TF hereda contexto del superior y contribuye score.
    M1 es el trigger final para entrada Sniper.
    
    Usage:
        engine = FractalMTFEngine()
        result = await engine.analyze_all_timeframes(candles_by_tf, current_price)
        if result.is_sniper_entry:
            execute_trade(result.action, result.suggested_entry)
    """
    
    def __init__(self):
        logger.info("[MTF] Fractal Multi-Timeframe Engine initialized")
        
        # Lazy imports to avoid circular dependencies
        self._structure_engine = None
        self._four_layer_ema = None
        self._acceleration_engine = None
    
    def _get_structure_engine(self):
        if self._structure_engine is None:
            from nexus_core.structure_engine import structure_engine
            self._structure_engine = structure_engine
        return self._structure_engine
    
    def _get_four_layer_ema(self):
        if self._four_layer_ema is None:
            from nexus_core.structure_engine import four_layer_ema
            self._four_layer_ema = four_layer_ema
        return self._four_layer_ema
    
    def _get_acceleration_engine(self):
        if self._acceleration_engine is None:
            from nexus_core.acceleration import acceleration_engine
            self._acceleration_engine = acceleration_engine
        return self._acceleration_engine
    
    async def analyze_all_timeframes(
        self,
        candles_by_tf: Dict[str, pd.DataFrame],
        current_price: float
    ) -> MTFCompositeScore:
        """
        Analyze all timeframes and compute composite score.
        
        Args:
            candles_by_tf: Dict mapping TF string to DataFrame
                Example: {"W1": df_w1, "D1": df_d1, ..., "M1": df_m1}
            current_price: Current market price
            
        Returns:
            MTFCompositeScore with weighted composite and entry decision
        """
        result = MTFCompositeScore()
        result.reasoning = []
        
        # Track directions for alignment calculation
        directions = []
        tf_scores = {}
        
        # Inherited context from higher TFs
        inherited_context = {
            "regime": "NEUTRAL",
            "macro_trend": "NEUTRAL",
            "bias": 0,
            "higher_bos": False,
            "higher_choch": False
        }
        
        # Process each timeframe in hierarchical order
        for tf in Timeframe:
            tf_key = tf.value
            candles = candles_by_tf.get(tf_key)
            
            if candles is None or len(candles) < 10:
                result.reasoning.append(f"⚠️ {tf_key}: Insufficient data")
                continue
            
            # Analyze this timeframe with context from higher TFs
            tf_score = self._analyze_timeframe(tf, candles, current_price, inherited_context)
            
            # Store score
            tf_scores[tf_key] = tf_score
            result.tf_details[tf_key] = tf_score
            
            # Update inherited context for lower TFs
            if tf_score.has_choch:
                inherited_context["higher_choch"] = True
            if tf_score.has_bos:
                inherited_context["higher_bos"] = True
            
            if tf == Timeframe.W1:
                inherited_context["regime"] = tf_score.trend
            elif tf == Timeframe.D1:
                inherited_context["macro_trend"] = tf_score.trend
            elif tf == Timeframe.H4:
                inherited_context["bias"] = tf_score.direction
            
            directions.append(tf_score.direction)
            
            # Map TF to result attribute
            setattr(result, f"{tf_key.lower()}_score", tf_score.score)
        
        # Calculate weighted composite score
        total_weight = 0.0
        weighted_sum = 0.0
        
        for tf in Timeframe:
            if tf.value in tf_scores:
                weight = WEIGHTS[tf]
                score = tf_scores[tf.value].score
                weighted_sum += weight * score
                total_weight += weight
        
        if total_weight > 0:
            result.composite_score = weighted_sum / total_weight
        
        # Calculate alignment
        if directions:
            bullish_count = sum(1 for d in directions if d > 0)
            bearish_count = sum(1 for d in directions if d < 0)
            total = len(directions)
            
            result.all_bullish = bullish_count == total
            result.all_bearish = bearish_count == total
            result.alignment_percentage = max(bullish_count, bearish_count) / total * 100
            
            if bullish_count > bearish_count:
                result.primary_direction = 1
            elif bearish_count > bullish_count:
                result.primary_direction = -1
            else:
                result.primary_direction = 0
        
        # Determine action and entry type
        result.action, result.entry_type = self._determine_entry(result, tf_scores, current_price)
        
        # Calculate entry levels if valid
        if result.action != "HOLD":
            result.suggested_entry, result.suggested_sl, result.suggested_tp = \
                self._calculate_entry_levels(result, tf_scores, current_price)
        
        # Log result
        if result.is_valid_setup:
            emoji = "🎯" if result.is_sniper_entry else ("💪" if result.is_strong_setup else "✅")
            logger.info(
                f"{emoji} MTF SIGNAL: {result.action} | "
                f"Composite: {result.composite_score:.2f} | "
                f"Alignment: {result.alignment_percentage:.0f}% | "
                f"Entry: {result.entry_type}"
            )
        
        return result
    
    def _analyze_timeframe(
        self,
        tf: Timeframe,
        candles: pd.DataFrame,
        current_price: float,
        context: Dict[str, Any]
    ) -> TimeframeScore:
        """
        Analyze a single timeframe with inherited context.
        
        Each TF contributes:
        - Trend direction
        - BOS/CHOCH detection
        - FVG/OB presence
        - Layer alignment
        """
        result = TimeframeScore(
            timeframe=tf.value,
            score=0.0,
            direction=0,
            trend="NEUTRAL"
        )
        
        reasoning = []
        score = 0.0
        
        try:
            structure = self._get_structure_engine()
            ema_layers = self._get_four_layer_ema()
            
            # 1. Structural analysis (BOS/CHOCH)
            narrative = structure.get_structural_narrative(candles)
            
            if "CHOCH" in narrative.get("type", ""):
                result.has_choch = True
                score += 0.25
                reasoning.append(f"CHOCH: {narrative['type']}")
            elif "BOS" in narrative.get("type", ""):
                result.has_bos = True
                score += 0.15
                reasoning.append(f"BOS: {narrative['type']}")
            
            # 2. MAE Pattern
            mae = structure.mae_recognizer.detect_mae_pattern(candles)
            if mae.get("is_expansion"):
                score += 0.20
                result.direction = mae.get("direction", 0)
                reasoning.append(f"MAE Expansion: {mae['status']}")
            elif mae.get("phase") == "ACCUMULATION":
                score += 0.05
                reasoning.append("Accumulation phase")
            
            # 3. Layer alignment
            alignment = ema_layers.compute_layer_alignment(candles)
            result.layer_alignment = alignment
            if alignment > 0.7:
                score += 0.20
                reasoning.append(f"Strong layer alignment: {alignment:.2f}")
            elif alignment > 0.5:
                score += 0.10
            
            # 4. Determine trend
            position = ema_layers.get_price_position(candles)
            if position.get("position") == "ABOVE_STRUCTURE":
                result.trend = "BULLISH"
                if result.direction == 0:
                    result.direction = 1
            elif position.get("position") == "BELOW_STRUCTURE":
                result.trend = "BEARISH"
                if result.direction == 0:
                    result.direction = -1
            else:
                result.trend = "NEUTRAL"
            
            # 5. Context bonus/penalty
            if context.get("higher_choch") and result.has_bos:
                score += 0.15
                reasoning.append("BOS confirms higher CHOCH")
            
            if context.get("bias") != 0:
                if context["bias"] == result.direction:
                    score += 0.10
                    reasoning.append("Aligned with H4 bias")
                elif result.direction != 0:
                    score -= 0.05
                    reasoning.append("⚠️ Against H4 bias")
            
            # 6. FVG/OB detection (for M15 and below)
            if tf in [Timeframe.M15, Timeframe.M5, Timeframe.M1]:
                try:
                    from app.skills.liquidity_detector import detect_fvg, detect_order_blocks
                    
                    fvgs = detect_fvg(candles, lookback=20)
                    if fvgs:
                        result.has_fvg = True
                        score += 0.10
                        reasoning.append(f"FVG detected ({len(fvgs)})")
                    
                    obs = detect_order_blocks(candles, lookback=30)
                    if obs:
                        result.has_ob = True
                        score += 0.10
                        reasoning.append(f"OrderBlock detected ({len(obs)})")
                except:
                    pass
            
            result.score = min(1.0, max(0.0, score))
            result.reasoning = reasoning
            
        except Exception as e:
            logger.warning(f"[{tf.value}] Analysis error: {e}")
            result.reasoning.append(f"ERROR: {e}")
        
        return result
    
    def _determine_entry(
        self,
        result: MTFCompositeScore,
        tf_scores: Dict[str, TimeframeScore],
        current_price: float
    ) -> Tuple[str, str]:
        """
        Determine action (BUY/SELL/HOLD) and entry type (MARKET/LIMIT/STOP).
        """
        if result.composite_score < result.THRESHOLD_SIGNAL:
            return "HOLD", "NONE"
        
        # Determine action from direction
        if result.primary_direction > 0:
            action = "BUY"
        elif result.primary_direction < 0:
            action = "SELL"
        else:
            return "HOLD", "NONE"
        
        # Determine entry type
        m1_score = tf_scores.get("M1")
        m5_score = tf_scores.get("M5")
        
        # MARKET entry: Very high confidence + M1 confirmed
        if result.composite_score >= result.THRESHOLD_SNIPER:
            if m1_score and m1_score.score >= 0.6:
                return action, "MARKET"
        
        # STOP entry: Waiting for BOS confirmation
        if m1_score and m1_score.has_bos:
            return action, "STOP"
        
        # LIMIT entry: Waiting for price to reach OB/FVG zone
        if m5_score and (m5_score.has_ob or m5_score.has_fvg):
            return action, "LIMIT"
        
        # Default to MARKET if strong enough
        if result.composite_score >= result.THRESHOLD_STRONG:
            return action, "MARKET"
        
        return action, "LIMIT"
    
    def _calculate_entry_levels(
        self,
        result: MTFCompositeScore,
        tf_scores: Dict[str, TimeframeScore],
        current_price: float
    ) -> Tuple[float, float, float]:
        """
        Calculate suggested entry, stop loss, and take profit levels.
        """
        entry = current_price
        
        # Get M15 data for zone-based SL
        m15 = tf_scores.get("M15")
        
        # Estimate ATR from price (rough approximation)
        atr_estimate = current_price * 0.002  # 0.2% of price
        
        if result.action == "BUY":
            sl = entry - (atr_estimate * 2.0)
            tp = entry + (atr_estimate * 4.0)  # 2:1 RR target
        else:
            sl = entry + (atr_estimate * 2.0)
            tp = entry - (atr_estimate * 4.0)
        
        return entry, sl, tp


# Global singleton
mtf_engine = FractalMTFEngine()
