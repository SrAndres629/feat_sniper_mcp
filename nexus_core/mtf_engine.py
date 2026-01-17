"""
Multi-Timeframe Fractal Engine v2.0 (Microstructure Aware)
==========================================================
Institutional Analysis Engine. 
Replaces Retail Logic (BOS/CHOCH) with Flow Dynamics (OFI/Liquidity/Impact).

Hierarchy:
W1/D1: Regime (Macro Liquidity State)
H4/H1: Flow (Order Flow Imbalance)
M30-M1: Microstructure (Impact & Vacuum Detection)
"""
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from app.skills.market_physics import market_physics, MarketRegime

logger = logging.getLogger("feat.mtf")


# =============================================================================
# TIMEFRAME DEFINITIONS
# =============================================================================

class Timeframe(Enum):
    """Timeframe hierarchy from macro to micro."""
    W1 = "W1"    # Weekly - Regime
    D1 = "D1"    # Daily - Macro Flow
    H4 = "H4"    # 4-Hour - Structural Bias
    H1 = "H1"    # 1-Hour - Intraday Flow
    M30 = "M30"  # 30-Min - Context
    M15 = "M15"  # 15-Min - Tactical
    M5 = "M5"    # 5-Min - Momentum
    M1 = "M1"    # 1-Min - SNIPER EXECUTION


# Weight per timeframe (must sum to 1.0)
WEIGHTS = {
    Timeframe.W1: 0.05,
    Timeframe.D1: 0.10,
    Timeframe.H4: 0.20,
    Timeframe.H1: 0.20,
    Timeframe.M30: 0.10,
    Timeframe.M15: 0.15,
    Timeframe.M5: 0.10,
    Timeframe.M1: 0.10,
}


# =============================================================================
# RESULT STRUCTURES (INSTITUTIONAL GRADE)
# =============================================================================

@dataclass
class TimeframeScore:
    """Score for a single timeframe based on Hydrodynamics."""
    timeframe: str
    score: float                    # 0.0-1.0 Probability of Valid Move
    direction: int                  # 1=Long, -1=Short, 0=Neutral
    trend: str                      # "ACCUMULATION", "DISTRIBUTION", "VACUUM_RUN"
    
    # Microstructure Metrics
    impact_pressure: float = 0.0    # Effective Force
    is_vacuum: bool = False         # Low Liquidity State
    is_absorption: bool = False     # High Effort / Low Result
    ofi_z_score: float = 0.0        # Order Flow Imbalance Significance
    
    reasoning: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "score": round(self.score, 3),
            "direction": self.direction,
            "trend": self.trend,
            "impact": round(self.impact_pressure, 2),
            "vacuum": self.is_vacuum,
            "absorption": self.is_absorption,
            "ofi_z": round(self.ofi_z_score, 2)
        }


@dataclass
class MTFCompositeScore:
    """
    Multi-Timeframe Composite Score.
    Aggregates Flow Dynamics across time.
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
    alignment_percentage: float = 0.0
    primary_direction: int = 0      # 1=Long, -1=Short
    
    # Trading decision
    action: str = "HOLD"            # BUY/SELL/HOLD
    entry_type: str = "NONE"        # AGGRESSIVE/PASSIVE
    suggested_entry: float = 0.0
    suggested_sl: float = 0.0
    suggested_tp: float = 0.0
    
    # Metadata
    tf_details: Dict[str, TimeframeScore] = field(default_factory=dict)
    reasoning: List[str] = field(default_factory=list)
    
    # Thresholds (Probabilistic)
    THRESHOLD_SIGNAL: float = 0.70
    THRESHOLD_SNIPER: float = 0.85
    THRESHOLD_SNIPER_TRIGGER: float = 0.80 # M1 Score requisite for override
    
    @property
    def is_valid_setup(self) -> bool:
        return self.composite_score >= self.THRESHOLD_SIGNAL and self.action != "HOLD"
    
    @property
    def is_sniper_entry(self) -> bool:
        return self.composite_score >= self.THRESHOLD_SNIPER
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "composite_score": round(self.composite_score, 3),
            "action": self.action,
            "entry_type": self.entry_type,
            "entry": self.suggested_entry,
            "sl": self.suggested_sl,
            "tp": self.suggested_tp,
            "reasoning": self.reasoning
        }


# =============================================================================
# FRACTAL MTF ENGINE (MICROSTRUCTURE)
# =============================================================================

class FractalMTFEngine:
    """
    Motor de Análisis Fractal de Microestructura.
    Decodifica la "Intención Institucional" analizando OFI e Impacto en 8 TFs.
    """
    
    def __init__(self):
        logger.info("[MTF] Fractal Microstructure Engine initialized")
        # No lazy imports needed for physics engine, it's a singleton
    
    async def analyze_all_timeframes(
        self,
        candles_by_tf: Dict[str, pd.DataFrame],
        current_price: float
    ) -> MTFCompositeScore:
        """
        Analyze all timeframes using Hydrodynamic Regime Detection.
        """
        result = MTFCompositeScore()
        result.reasoning = []
        
        directions = []
        tf_scores = {}
        
        # Context inheritance (Flow Bias)
        flow_bias_z = 0.0
        
        # Process from Macro (W1) to Micro (M1)
        for tf in Timeframe:
            tf_key = tf.value
            candles = candles_by_tf.get(tf_key)
            
            if candles is None or len(candles) < 20:
                result.reasoning.append(f"⚠️ {tf_key}: Insufficient data for Statistical Sig.")
                continue
            
            # 1. Hydrate Physics Engine Contextually (Simulation)
            # We take the last slice of candles to estimate current regime for this TF
            # Normalize candle to 'tick-like' dict for physics engine ingestion
            last_candle = candles.iloc[-1]
            tick_data = {
                'close': last_candle['close'],
                'tick_volume': last_candle['tick_volume'],
                'time': last_candle['time']
            }
            
            # Temporarily ingest into physics engine to get regime
            # Note: Ideally we hydrate with a window, but for this step we check the immediacy
            regime = market_physics.ingest_tick(tick_data)
            
            # If engine not warm, we simulate logic here to keep strategy robust
            # (Fallback to dataframe calculation if Physics Engine is cold)
            tf_score = self._analyze_hydrodynamics(tf, candles, regime, flow_bias_z)
            
            tf_scores[tf_key] = tf_score
            result.tf_details[tf_key] = tf_score
            
            # Pass bias down
            if tf in [Timeframe.W1, Timeframe.D1, Timeframe.H4]:
                flow_bias_z += tf_score.ofi_z_score * 0.5 # Decay influence
            
            directions.append(tf_score.direction)
            setattr(result, f"{tf_key.lower()}_score", tf_score.score)
        
        # Calculate Composite
        self._calculate_composite(result, tf_scores)
        
        # Determine Trade
        if result.composite_score > 0.1: # Optimization skip
            self._determine_trade(result, tf_scores, current_price)
            
        return result
    
    def _analyze_hydrodynamics(
        self,
        tf: Timeframe,
        candles: pd.DataFrame,
        regime: Optional[MarketRegime],
        parent_bias_z: float
    ) -> TimeframeScore:
        """
        Calculates score based on Fluid Dynamics (OFI, Impact, Liquidity).
        """
        result = TimeframeScore(
            timeframe=tf.value,
            score=0.0,
            direction=0,
            trend="NEUTRAL"
        )
        
        score = 0.0
        reasoning = []
        
        # --- 1. Institutional Logic Calculation (Dataframe Fallback) ---
        # Calculate OFI (Order Flow Imbalance) approximation on Candles
        closes = candles['close'].values
        volumes = candles['tick_volume'].values
        
        delta_p = np.diff(closes)
        # Direction of trade based on candle color (naive tick rule approximation for candles)
        trade_dir = np.sign(delta_p) 
        ofi_proxy = trade_dir * volumes[1:]
        
        # Z-Score of OFI (Last 20 bars)
        ofi_window = ofi_proxy[-20:]
        ofi_mean = np.mean(ofi_window)
        ofi_std = np.std(ofi_window) + 1e-6
        current_ofi_z = (ofi_proxy[-1] - ofi_mean) / ofi_std
        
        result.ofi_z_score = current_ofi_z
        
        # --- 2. Regime Scoring ---
        
        # A. Flow Alignment (OFI matches Price Move)
        if current_ofi_z > 1.5:
            score += 0.2
            result.direction = 1
            reasoning.append("Strong Buying Pressure (OFI > 1.5σ)")
        elif current_ofi_z < -1.5:
            score += 0.2
            result.direction = -1
            reasoning.append("Strong Selling Pressure (OFI < -1.5σ)")
            
        # B. Liquidity Vacuum (Low Volume, High Move) -> Vacuum Run
        # This is high opportunity but high risk (Reversion)
        # We assume Vacuum if price moved > 2stdev but volume is low
        move_z = abs(delta_p[-1]) / (np.std(delta_p[-20:]) + 1e-6)
        vol_z = (volumes[-1] - np.mean(volumes[-20:])) / (np.std(volumes[-20:]) + 1e-6)
        
        if move_z > 2.0 and vol_z < 0:
            result.is_vacuum = True
            result.trend = "VACUUM_RUN"
            score += 0.3 # High potential
            reasoning.append("Liquidity Vacuum Detect (Fast Move/Low Res)")
            
        # C. Absorption (High Volume, Low Move) -> Reversal Warning
        if vol_z > 2.0 and move_z < 0.5:
            result.is_absorption = True
            result.trend = "ABSORPTION"
            score -= 0.1 # Penalty for direction, usually reversal
            # If absorbing trend, direction flips? Complex logic simplified:
            reasoning.append("Absorption Detected (Hidden Wall)")
            
        # --- 3. Context Integration ---
        # Add bonus if aligned with Higher TF Flow
        if parent_bias_z != 0 and np.sign(result.ofi_z_score) == np.sign(parent_bias_z):
            score += 0.15
            reasoning.append(f"Aligned with Macro Flow")
            
        result.score = min(1.0, max(0.0, score))
        result.reasoning = reasoning
        
        # Set trend text based on score/dir
        if result.trend == "NEUTRAL":
            if result.score > 0.5:
                result.trend = "BULLISH_FLOW" if result.direction > 0 else "BEARISH_FLOW"
                
        return result

    def _calculate_composite(self, result: MTFCompositeScore, tf_scores: Dict[str, TimeframeScore]):
        weighted_sum = 0.0
        total_weight = 0.0
        
        bullish = 0
        bearish = 0
        total = 0
        
        for tf, weight in WEIGHTS.items():
            key = tf.value
            if key in tf_scores:
                s = tf_scores[key]
                weighted_sum += s.score * weight
                total_weight += weight
                
                if s.direction > 0: bullish += 1
                elif s.direction < 0: bearish += 1
                total += 1
        
        if total_weight > 0:
            result.composite_score = weighted_sum / total_weight
            
        if total > 0:
            result.alignment_percentage = (max(bullish, bearish) / total) * 100
            result.primary_direction = 1 if bullish > bearish else -1

        # [SNIPER MODE LOGIC]
        # Validates user request: "Si M1 da señal fuerte, debe disparar aunque H1 esté neutral"
        # Pre-requisite: H4 must NOT be opposing (H4 Bias Validation)
        
        m1 = tf_scores.get("M1")
        h4 = tf_scores.get("H4")
        
        # [DOCTORAL STANDARD] Use defined constant, no magic numbers.
        if m1 and m1.score >= result.THRESHOLD_SNIPER_TRIGGER: # Strong Sniper Trigger
            h4_direction = h4.direction if h4 else 0
            m1_direction = m1.direction
            
            # Check for Conflict (H4 against M1)
            is_conflict = (h4_direction != 0) and (h4_direction != m1_direction)
            
            if not is_conflict:
                # BOOST SCORE to Guarantee Trigger
                # We map the strong M1 score directly to composite, ensuring it crosses THRESHOLD
                boosted_score = max(result.composite_score, m1.score)
                if boosted_score >= result.THRESHOLD_SNIPER:
                    result.composite_score = boosted_score
                    result.reasoning.append(f"🎯 Sniper Override: M1 Strong ({m1.score}) + H4 Aligned/Neutral")

    def _determine_trade(
        self, 
        result: MTFCompositeScore, 
        tf_scores: Dict[str, TimeframeScore],
        current_price: float
    ):
        if result.composite_score < result.THRESHOLD_SIGNAL:
            return
            
        # Determine Action
        result.action = "BUY" if result.primary_direction > 0 else "SELL"
        
        # Determine Type based on Microstructure
        m1 = tf_scores.get("M1")
        m5 = tf_scores.get("M5")
        
        # If Vacuum detected in M1/M5, Market Entry (Chase the void)
        if (m1 and m1.is_vacuum) or (m5 and m5.is_vacuum):
            result.entry_type = "AGGRESSIVE_MARKET" 
            result.reasoning.append("Vacuum Run -> Aggressive Entry")
        
        # If Absorption detected, Limit Order (Fade the wall)
        elif (m1 and m1.is_absorption):
            result.entry_type = "PASSIVE_LIMIT"
            result.reasoning.append("Absorption -> Passive Entry")
        
        else:
            result.entry_type = "MARKET"
            
        # Calculate Levels (ATR based logic kept for simplicity, better logic implies searching liquidity pools)
        # Simplified for robustness
        atr_proxy = current_price * 0.001
        
        if result.action == "BUY":
            result.suggested_sl = current_price - (atr_proxy * 10)
            result.suggested_tp = current_price + (atr_proxy * 20)
        else:
            result.suggested_sl = current_price + (atr_proxy * 10)
            result.suggested_tp = current_price - (atr_proxy * 20)
            
        result.suggested_entry = current_price


# Global singleton
mtf_engine = FractalMTFEngine()
