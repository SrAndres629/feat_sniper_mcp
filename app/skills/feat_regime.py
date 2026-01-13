"""
FEAT REGIME STATE MACHINE (FSM)
===============================
Classifies the market into functional regimes based on Physics + Chrono inputs.

Regimes:
1. ACCUMULATION (Low Entropy, Value Building)
2. MANIPULATION (Phase Transition, Trap)
3. EXPANSION_REAL (High Entropy, Valid Trend)
4. EXPANSION_FAKE (Failed Breakout, Absorption)
5. DISTRIBUTION (High Entropy, Value Shift)
6. REVERSAL (Structural Shift)

Output: Probability Distribution of States (e.g., {"EXPANSION": 0.85})
"""

import logging
from typing import Dict, Any, List
from enum import Enum
import math

logger = logging.getLogger("FEAT.RegimeFSM")

class MarketRegime(Enum):
    ACCUMULATION = "ACCUMULATION"
    MANIPULATION = "MANIPULATION"
    EXPANSION_REAL = "EXPANSION_REAL"
    EXPANSION_FAKE = "EXPANSION_FAKE"
    DISTRIBUTION = "DISTRIBUTION"
    REVERSAL = "REVERSAL"
    NEUTRAL = "NEUTRAL"

class MarketRegimeFSM:
    """
    Probabilistic Finite State Machine for FEAT NEXUS.
    Uses Bayesian-like heuristics to score each regime.
    """
    
    def __init__(self):
        # Base probabilities (priors)
        self.priors = {r: 0.15 for r in MarketRegime}
        self.priors[MarketRegime.NEUTRAL] = 0.25

    def detect_regime(
        self,
        physics_metrics: Dict[str, Any],
        temporal_features: Dict[str, Any],
        structure_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detects current market regime.
        
        Args:
            physics_metrics: Output from market_physics (PVP, MCI, etc.)
            temporal_features: Output from killzone_intelligence (Heat, State)
            structure_context: Output from feat_forma (Trend, BOS)
        
        Returns:
            Dict with 'best_regime', 'probabilities', 'confidence'
        """
        scores = {r: 0.0 for r in MarketRegime}
        
        # Extract features
        poc_displacement = physics_metrics.get("poc_displacement", 0.0) # To be calc from prev cycle
        mci_score = physics_metrics.get("mci", {}).get("mci_score", 0.0)
        mci_type = physics_metrics.get("mci", {}).get("mci_type", "NEUTRAL")
        va_width_pct = physics_metrics.get("pvp", {}).get("va_volume_pct", 0.70) # Proxy for width if width not direct
        
        session_heat = temporal_features.get("session_heat", 0.0)
        killzone_state = temporal_features.get("liquidity_state", "OUTSIDE")
        
        trend = structure_context.get("trend", "NEUTRAL")
        has_sweep = structure_context.get("has_sweep", False)
        
        # --- SCORING HEURISTICS ---
        
        # 1. ACCUMULATION SCORING
        # - Low Displacement
        # - Narrow Value Area
        # - Low Session Heat (often Asian)
        if abs(poc_displacement) < 0.0010: # Small shift
            scores[MarketRegime.ACCUMULATION] += 0.4
        if session_heat < 0.3:
            scores[MarketRegime.ACCUMULATION] += 0.3
        if killzone_state == "ACCUMULATION":
            scores[MarketRegime.ACCUMULATION] += 0.5

        # 2. MANIPULATION SCORING
        # - Sweep detected
        # - High Volatility but Low Value Shift
        # - Killzone Start
        if has_sweep or "SWEEP" in mci_type:
            scores[MarketRegime.MANIPULATION] += 0.6
        if killzone_state == "SWEEP_HUNT":
            scores[MarketRegime.MANIPULATION] += 0.5
        if mci_type == "BEARISH_SWEEP" or mci_type == "BULLISH_SWEEP":
             scores[MarketRegime.MANIPULATION] += 0.4

        # 3. EXPANSION (REAL) SCORING
        # - Significant POC Displacement
        # - High Session Heat
        # - Trend Alignment
        # - MCI Confirmation (Displacement > Sweep)
        if abs(poc_displacement) > 0.0020:
             scores[MarketRegime.EXPANSION_REAL] += 0.5
        if session_heat > 0.7:
             scores[MarketRegime.EXPANSION_REAL] += 0.3
        if trend != "NEUTRAL":
             scores[MarketRegime.EXPANSION_REAL] += 0.2
        if mci_type == "CONTINUATION":
             scores[MarketRegime.EXPANSION_REAL] += 0.4

        # 4. EXPANSION (FAKE) SCORING
        # - Price moves but POC stays (Divergence)
        # - High MCI Rejection Ratio
        if abs(poc_displacement) < 0.0010 and session_heat > 0.8: # Price moving, volume not?
            scores[MarketRegime.EXPANSION_FAKE] += 0.6
        if "SWEEP" in mci_type and mci_score > 2.0: # Huge rejection
            scores[MarketRegime.EXPANSION_FAKE] += 0.5

        # 5. REVERSAL SCORING
        # - Structure Shift (CHoCH)
        # - Value Shift in opposite direction
        if structure_context.get("has_choch", False):
            scores[MarketRegime.REVERSAL] += 0.6
            
        # --- FRACTAL OVERRIDE (Physics) ---
        # "Mathematics doesn't lie. Structures do."
        hurst = physics_metrics.get("hurst_exponent", 0.5)
        entropy = physics_metrics.get("shannon_entropy", 0.0)
        
        # TRENDING STATE (Hurst > 0.6, Low Entropy)
        if hurst > 0.6 and entropy < 0.8:
            scores[MarketRegime.EXPANSION_REAL] += 0.5
            scores[MarketRegime.ACCUMULATION] -= 0.3
            
        # MEAN REVERTING (Hurst < 0.4) "Pink Noise"
        elif hurst < 0.4:
            scores[MarketRegime.ACCUMULATION] += 0.4
            scores[MarketRegime.MANIPULATION] += 0.3
            scores[MarketRegime.EXPANSION_REAL] -= 0.5
            
        # RANDOM WALK (Hurst ~ 0.5, High Entropy) "White Noise"
        # Danger Zone - The market is confused.
        elif 0.45 <= hurst <= 0.55 and entropy > 0.9:
            scores[MarketRegime.NEUTRAL] += 0.8
            scores[MarketRegime.EXPANSION_REAL] -= 1.0 # Veto trade
            scores[MarketRegime.EXPANSION_FAKE] += 0.4 # More likely to be fake
        
        # --- PROBABILITY CALCULATION ---
        
        # Normalize scores to probabilities
        total_score = sum(scores.values()) + 0.01
        probabilities = {k.value: v / total_score for k, v in scores.items()}
        
        # Find winner
        best_regime = max(probabilities, key=probabilities.get)
        confidence = probabilities[best_regime]
        
        # Apply thresholds
        final_regime = best_regime
        if confidence < 0.35:
            final_regime = MarketRegime.NEUTRAL.value
            
        return {
            "regime": final_regime,
            "confidence": round(confidence, 2),
            "probabilities": {k: round(v, 2) for k, v in probabilities.items()},
            "is_trap": final_regime in ["MANIPULATION", "EXPANSION_FAKE"],
            "is_trending": final_regime in ["EXPANSION_REAL", "DISTRIBUTION"],
            "fractal_state": {
                "hurst": hurst,
                "entropy": entropy,
                "interpretation": "TRENDING" if hurst > 0.6 else ("REVERTING" if hurst < 0.4 else "RANDOM")
            }
        }

# Singleton
regime_fsm = MarketRegimeFSM()
